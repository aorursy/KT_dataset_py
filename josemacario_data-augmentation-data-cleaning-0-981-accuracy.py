import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

%matplotlib inline
train_dataset = pd.read_csv("../input/train.csv")
test_dataset = pd.read_csv("../input/test.csv")
print(train_dataset.shape, test_dataset.shape)
train_dataset.describe()
test_dataset.describe()
LABELS = len(train_dataset["label"].value_counts())
print("Distinct labels:", LABELS)
train_dataset["label"].value_counts()
y_train = to_categorical(train_dataset["label"], num_classes = LABELS)
print(y_train)
x_train = train_dataset.drop(labels = ["label"], axis = 1).values
print(x_train.shape, y_train.shape)
FEATURES = len(x_train[0])
print(FEATURES)
x_train = x_train.reshape(-1, 28, 28, 1)
print(x_train.shape, y_train.shape)
RANDOM_SEED = 3
BATCH_SIZE = 500
# Passing parameters to Keras' generator function
datagen = ImageDataGenerator(featurewise_center = False,
                             samplewise_center = False,
                             featurewise_std_normalization = False,
                             samplewise_std_normalization = False,
                             zca_whitening = False,
                             rotation_range = 10,
                             zoom_range = 0.1,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             horizontal_flip = False,
                             vertical_flip = False)

# Feeding the generator with images we already have
datagen.fit(x_train)

# Getting new images
train_gen = datagen.flow(x_train, y_train, batch_size = BATCH_SIZE, seed = RANDOM_SEED)

# Adding new images (and their recpective labels) to our training datasets
x_train = np.concatenate((x_train, train_gen.x))
y_train = np.concatenate((y_train, train_gen.y))
print(x_train.shape, y_train.shape)
x_train = x_train.reshape(-1, FEATURES)
print(x_train.shape, y_train.shape)
# Defining a list of features (columns) we'll dropp out
dropped_columns = []

# Discovering which features will be dropped out from training
for column in range(len(x_train[0])):
    if x_train[:,column].max() == x_train[:,column].min():
        dropped_columns.append(column)

# Can the features we found be removed from the tests as well?
x_test = test_dataset.values
for column in dropped_columns:
    if not x_test[:,column].max() == x_test[:,column].min():
        dropped_columns.remove(column)

# Dropping out those irrelevant features
x_train = np.delete(x_train, dropped_columns, axis = 1)
x_test = np.delete(x_test, dropped_columns, axis = 1)

print(x_train.shape, y_train.shape, x_test.shape)
FEATURES = len(x_train[0])
print(FEATURES)
x_train = x_train / 255.0
x_test = x_test / 255.0
HIDDEN_L1 = 256
DROPOUT_L1 = 0.33
HIDDEN_L2 = 128
DROPOUT_L2 = 0.33
model_NN = Sequential()
model_NN.add(Dense(HIDDEN_L1, input_shape = (FEATURES,)))
model_NN.add(Activation('relu'))
model_NN.add(Dropout(DROPOUT_L1))
model_NN.add(Dense(HIDDEN_L2))
model_NN.add(Activation('relu'))
model_NN.add(Dropout(DROPOUT_L2))
model_NN.add(Dense(LABELS))
model_NN.add(Activation('softmax'))
model_NN.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
model_NN.summary()
EPOCHS = 30
VALIDATION_SIZE = 0.1
training_history = model_NN.fit(x_train,
                                y_train,
                                batch_size = BATCH_SIZE,
                                epochs = EPOCHS,
                                verbose = 2,
                                validation_split = VALIDATION_SIZE)
fig, ax = plt.subplots(2, 1)
ax[0].plot(training_history.history['loss'], color = 'b', label = "Training loss")
ax[0].plot(training_history.history['val_loss'], color = 'r', label = "validation loss", axes = ax[0])
legend = ax[0].legend(loc = 'best', shadow = True)

ax[1].plot(training_history.history['acc'], color = 'b', label = "Training accuracy")
ax[1].plot(training_history.history['val_acc'], color = 'r', label = "Validation accuracy")
legend = ax[1].legend(loc = 'best', shadow = True)
predictions = np.argmax(model_NN.predict(x_test), axis = 1)
predictions = pd.Series(predictions, name = "Label")

submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), predictions], axis = 1)

submission.to_csv("submission.csv", index = False)
