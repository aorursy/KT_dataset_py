# First we trained a ResNet50 network which achieved 79% accuracy
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from imagenet_utils import preprocess_input
# parameters
img_width, img_height = 200, 200
train_data_dir = "images/train"
validation_data_dir = "images/dev"
nb_train_samples = 32488
nb_validation_samples = 5723
batch_size = 32
epochs = 50
# load the original ResNet without top layer
model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
# adding custom Layers 
x = model.output
x = Flatten(name='flatten')(x)
out = Dense(18, activation="softmax", name='output_layer')(x)

# create the final model
model_final = Model(inputs=model.input, outputs=out)
# freeze layers to stage 3d
for layer in model_final.layers[:79]:
    layer.trainable = False
def print_layer_trainable(model_name):
    for layer in model_name.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
print_layer_trainable(model_final)
# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])
# initiate the train and validation generators with data augumentation
train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input,
        fill_mode='nearest')

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

# save the model according to the conditions
checkpoint = ModelCheckpoint("models/ResNet50_trained_f3d.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
model_final.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    callbacks = [checkpoint])
# Second we trained a deep neural network of hand crafted feature extraction
model = Sequential()
model.add(Dense(2048, input_shape=(2014,)))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(18, activation='softmax'))
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])
nb_train_samples = 32488
nb_validation_samples = 5723
batch_size = 32
epochs = 1000
from resnets_utils import*

# initiate the train and validation generators with data augumentation
X_train, Y_train_orig, X_test, Y_test_orig = load_dataset()

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 18)
Y_test = convert_to_one_hot(Y_test_orig, 18)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# save the model according to the conditions
checkpoint = ModelCheckpoint("models/feature_network.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
H = model.fit(X_train, Y_train, 
          epochs=epochs, batch_size=batch_size, 
          callbacks = [checkpoint], 
          validation_data=(X_test, Y_test))
# Third we trained a bottleneck network by combining both ResNet50 and feature network
train_img = np.load('bottleneck_img_train.npy')
train_feature = np.load('bottleneck_feature_train.npy')

train_data = np.concatenate((train_img, train_feature), axis=1)
train_labels = load_label('label_img_train.h5')
print(train_data.shape)
print(train_labels.shape)

val_img = np.load('bottleneck_img_val.npy')
val_feature = np.load('bottleneck_feature_val.npy')

val_data = np.concatenate((val_img, val_feature), axis=1)
val_labels = load_label('label_img_val.h5')
print(val_data.shape)
print(val_labels.shape)
model_top = Sequential()
model_top.add(Dense(18, activation='softmax', input_shape=train_data.shape[1:], name='shopee_output'))

model_top.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])
# save the model according to the conditions
checkpoint = ModelCheckpoint("models/bottleneck.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto')
H = model_top.fit(train_data, train_labels,
          epochs=epochs, batch_size=batch_size, 
          callbacks = [checkpoint, early],
          validation_data=(val_data, val_labels))
# Finally the combined model is used for prediction