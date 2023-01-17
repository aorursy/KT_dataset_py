# Make sure to use tensorflow.keras and not keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
# Paths to train and test images
train_path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train'
val_path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val'
test_path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test'
target_size = (224, 224)
colormode = 'rgb'
seed = 666
batch_size = 64

# Training ImageDataGenerator will have data augumentation parameters.
train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                             rotation_range=5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.1)

val_datagen = ImageDataGenerator(rescale = 1.0/255.0)
test_datagen = ImageDataGenerator(rescale = 1.0/255.0)


# Creating training, validation and test generators
train_generator = train_datagen.flow_from_directory(directory = train_path, 
                                             target_size = target_size, 
                                             color_mode = colormode, 
                                             batch_size = batch_size,
                                             class_mode = 'binary',
                                             shuffle = True,
                                             seed = seed)

valid_generator = val_datagen.flow_from_directory(directory = val_path,
                                             target_size = target_size,
                                             color_mode = colormode,
                                             batch_size = batch_size,
                                             class_mode = 'binary',
                                             shuffle = True,
                                             seed = seed)

test_generator = test_datagen.flow_from_directory(directory = test_path,
                                            target_size = target_size,
                                            color_mode = colormode,
                                            batch_size = 1,
                                            class_mode = 'binary',
                                            shuffle = False, 
                                            seed = seed)

# Define number of steps for fit_generator function
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
base_model = keras.applications.VGG16(include_top=False, weights='imagenet',input_shape = (224,224,3))
x = keras.layers.Flatten() (base_model.output)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.25)(x)
output = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# The newly added layers are initialized with random values.
# Make sure based model remain unchanged until newly added layers weights get reasonable values.
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# Uncomment these lines to see the final model architecture:
# model.summary()
# See all the layers with index.
# for index, layer in enumerate(base_model.layers):
#  
# Defining checkpoint callback
checkpoint = ModelCheckpoint('../working/best_model.hdf5', verbose = 1, monitor = 'val_binary_accuracy', save_best_only = True)

# Fit model to get reasonable weights for newly added layers.
history = model.fit_generator(generator = train_generator,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_data = valid_generator,
                             validation_steps = STEP_SIZE_VALID,
                             epochs = 5, callbacks = [checkpoint])
fig, ax = plt.subplots()
ax.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy as training progresses')
plt.plot(history.history['binary_accuracy'],'r--', label = "Training Accuracy" , linewidth=4.0)
plt.plot(history.history['val_binary_accuracy'], 'b--', label = "Validation Accuracy",  linewidth=4.0)
plt.legend()
plt.annotate('High Accuracy - Weights of top layers acceptable now', xy=(3.5, .93), xytext=(4.5, 0.90),
             arrowprops=dict(facecolor='green', shrink=0.05),
             )
plt.show()

fig, ax = plt.subplots()
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss as training progresses')
plt.plot(history.history['loss'], 'r--', label = "Train loss", linewidth=4.0)
plt.plot(history.history['val_loss'], 'b--', label = "Val loss", linewidth=4.0)
plt.legend()
plt.annotate('Very low loss - Weights of top layers acceptable now', xy=(3.5, .2), xytext=(4.5, 0.3),
             arrowprops=dict(facecolor='green', shrink=0.05),
             )
plt.show()

# Now let's train the full model and update all weights.
for layer in base_model.layers:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate (This ensures the base model weights do not change a lot)
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['binary_accuracy'])
# Fit model
history = model.fit_generator(generator = train_generator,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_data = valid_generator,
                             validation_steps = STEP_SIZE_VALID,
                             epochs = 10, callbacks = [checkpoint])
fig, ax = plt.subplots()
ax.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy as training progresses')
plt.plot(history.history['binary_accuracy'],'r--', label = "Training Accuracy" , linewidth=4.0)
plt.plot(history.history['val_binary_accuracy'], 'b--', label = "Validation Accuracy",  linewidth=4.0)
plt.legend()
plt.annotate('High Accuracy only for training set - Potential overfitting', xy=(8.5, .974), xytext=(9.5, 0.96),
             arrowprops=dict(facecolor='green', shrink=0.05),
             )
plt.show()

fig, ax = plt.subplots()
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss as training progresses')
plt.plot(history.history['loss'], 'r--', label = "Train loss", linewidth=4.0)
plt.plot(history.history['val_loss'], 'b--', label = "Val loss", linewidth=4.0)
plt.legend()
plt.annotate('Low loss only for training set - Potential overfitting', xy=(8, .07), xytext=(9.5, 0.07),
             arrowprops=dict(facecolor='green', shrink=0.05),
             )
plt.show()

saved_model = keras.models.load_model('../working/best_model.hdf5')
validation_set_performance = saved_model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)
test_set_performance = saved_model.evaluate_generator(generator=test_generator,
steps=STEP_SIZE_TEST)
print("Validation set accuracy in %: " + str(validation_set_performance[1]*100))
print("Test set accuracy in %: " + str(test_set_performance[1]*100))