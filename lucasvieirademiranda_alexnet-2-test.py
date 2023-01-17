from keras.models import Sequential, load_model
from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
data_dir = "../input/asl-alphabet/asl-alphabet/asl_alphabet_train_full/"
#data_dir = "../input/asl_alphabet_edges_train_full/asl_alphabet_edges_train_full"
target_size = (200, 200)
target_dims = (200, 200, 3) # add channel for RGB
n_classes = 29
validation_percentage = 0.1
batch_size = 128

data_augmentor = ImageDataGenerator(samplewise_center = True, 
                                    samplewise_std_normalization = True,
                                    validation_split = validation_percentage)

training_generator = data_augmentor.flow_from_directory(data_dir, target_size = target_size, batch_size = batch_size, shuffle = True, subset = "training")

validation_generator = data_augmentor.flow_from_directory(data_dir, target_size = target_size, batch_size = batch_size, shuffle = True, subset = "validation")
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters = 96, input_shape = (200, 200, 3), kernel_size = (11, 11), strides = (4, 4), padding = 'valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Batch Normalization
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Batch Normalization
model.add(BatchNormalization())

# 3rd Convolutional Layer

model.add(Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))

# Batch Normalization
model.add(BatchNormalization())

# 4th Convolutional Layer

model.add(Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))

# Batch Normalization
model.add(BatchNormalization())

# 5th Convolutional Layer

model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Batch Normalization
model.add(BatchNormalization())

# Transforming in to a dense layer
model.add(Flatten())

# 1st Dense Layer
model.add(Dense(4096, input_shape = (200 * 200 * 3,)))
model.add(Activation('relu'))

# Dropout
model.add(Dropout(0.5))

# Batch Normalization
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))

# Dropout
model.add(Dropout(0.5))

# Batch Normalization
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(29))
model.add(Activation('softmax'))

model.summary()
stopping = EarlyStopping(monitor = 'val_acc', patience = 3, mode = 'max', restore_best_weights = True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(training_generator, initial_epoch = 0, epochs = 100, steps_per_epoch = len(training_generator), validation_data = validation_generator, validation_steps = len(validation_generator), callbacks = [stopping])
scores = model.evaluate_generator(validation_generator, steps = len(validation_generator))
print("CNN Accuracy: %.2f" % scores[1])
print("CNN Error: %.2f%%" % (1 - scores[1]))
data_dir = "../input/asl-alphabet/asl-alphabet/asl_alphabet_test_full/"
#data_dir = "../input/asl_alphabet_edges_test_full/asl_alphabet_edges_test_full"
target_size = (200, 200)
target_dims = (200, 200, 3) # add channel for RGB
n_classes = 29
batch_size = 128

data_augmentor = ImageDataGenerator(samplewise_center = True, 
                                    samplewise_std_normalization = True)

test_generator = data_augmentor.flow_from_directory(data_dir, target_size = target_size, batch_size = batch_size, shuffle = True)
scores = model.evaluate_generator(test_generator, steps = len(test_generator))
print("CNN Accuracy: %.2f" % scores[1])
print("CNN Error: %.2f%%" % (1 - scores[1]))
model.save("alexnet.h5")
model = load_model("alexnet.h5")

scores = model.evaluate_generator(test_generator, steps = len(test_generator))
print("CNN Accuracy: %.2f" % scores[1])
print("CNN Error: %.2f%%" % (1 - scores[1]))