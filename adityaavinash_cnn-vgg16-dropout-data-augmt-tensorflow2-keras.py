# Make sure to use tensorflow.keras and not keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
# Paths to train and test images
train_path = '../input/intel-image-classification/seg_train/seg_train'
test_path = '../input/intel-image-classification/seg_test/seg_test'
target_size = (224, 224)
colormode = 'rgb'
seed = 666
batch_size = 64

test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
# Splits train data into train and validation sets with 0.9 / 0.1 proportion
train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                             validation_split = 0.1,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             shear_range=0.2,
                             zoom_range=0.2)

# Creating training, validation and test generators
train_generator = train_datagen.flow_from_directory(directory = train_path, 
                                             target_size = target_size, 
                                             color_mode = colormode, 
                                             batch_size = batch_size,
                                             class_mode = 'categorical',
                                             shuffle = True,
                                             seed = seed,
                                             subset = 'training')

valid_generator = train_datagen.flow_from_directory(directory = train_path,
                                             target_size = target_size,
                                             color_mode = colormode,
                                             batch_size = batch_size,
                                             class_mode = 'categorical',
                                             shuffle = True,
                                             seed = seed,
                                             subset = 'validation')

test_generator = test_datagen.flow_from_directory(directory = test_path,
                                            target_size = target_size,
                                            color_mode = colormode,
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            shuffle = False, 
                                            seed = seed)

# Define number of steps for fit_generator function
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
base_model = keras.applications.VGG16(include_top=False, weights='imagenet',input_shape = (224,224,3))
x = keras.layers.Flatten() (base_model.output)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.1)(x)
output = keras.layers.Dense(6, activation="softmax")(x)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# The newly added layers are initialized with random values.
# Make sure based model remain unchanged until newly added layers weights get reasonable values.
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Uncomment these lines to see the final model architecture:
# model.summary()
# See all the layers with index.
# for index, layer in enumerate(base_model.layers):
#    print(index, layer.name)
# Defining checkpoint callback
checkpoint = ModelCheckpoint('../working/best_model.hdf5', verbose = 1, monitor = 'val_categorical_accuracy', save_best_only = True)

# Fit model to get reasonable weights for newly added layers.
history = model.fit_generator(generator = train_generator,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_data = valid_generator,
                             validation_steps = STEP_SIZE_VALID,
                             epochs = 5, callbacks = [checkpoint])
# Now let's train the full model and update all weights.
for layer in base_model.layers:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate (This ensures the base model weights do not change a lot)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['categorical_accuracy'])
# Fit model
history = model.fit_generator(generator = train_generator,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_data = valid_generator,
                             validation_steps = STEP_SIZE_VALID,
                             epochs = 10, callbacks = [checkpoint])
saved_model = keras.models.load_model('../working/best_model.hdf5')
validation_set_performance = saved_model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)
test_set_performance = saved_model.evaluate_generator(generator=test_generator,
steps=STEP_SIZE_TEST)
print("Validation set accuracy in %: " + str(validation_set_performance[1]*100))
print("Test set accuracy in %: " + str(test_set_performance[1]*100))