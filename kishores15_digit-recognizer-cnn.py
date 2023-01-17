# Importing the required Libraries

import pandas as pd

import numpy as np

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Loading the train and test files

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Checking the shape of train and test data

print(train_data.shape)

print(test_data.shape)
train_data.head()
test_data.head()
# Splitting label and features from train_data

label = train_data["label"]

features = train_data.drop(labels = "label", axis = 1)
print(features.shape)

print(label.shape)
from sklearn.preprocessing import StandardScaler



scaler1 = StandardScaler()

scaler2 = StandardScaler()



features = scaler1.fit_transform(features)

test_data = scaler2.fit_transform(test_data)
# One - Hot Encodings of Labels

labels = tf.keras.utils.to_categorical(label)
labels.shape
# Reshaping the features

features = features.reshape(-1, 28, 28, 1)
print(features.shape)
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size = 0.1)
print("X_Train Shape = ", x_train.shape)

print("Y_Train Shape = ", y_train.shape)

print("X_Val Shape = ", x_val.shape)

print("Y_Val Shape = ", y_val.shape)
# Model

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = "Same", activation = "relu", input_shape = (28, 28, 1)))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = "Same", activation = "relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Conv2D(filters = 128, padding = "Same", kernel_size = 3, activation = "relu"))

model.add(tf.keras.layers.Conv2D(filters = 256, padding = "Same", kernel_size = 3, activation = "relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Conv2D(filters = 64, padding = "Same", kernel_size = 3, activation = "relu"))

model.add(tf.keras.layers.Conv2D(filters = 32, padding = "Same", kernel_size = 3, activation = "relu"))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Conv2D(filters = 32, padding = "Same", kernel_size = 3, activation = "relu"))

model.add(tf.keras.layers.Conv2D(filters = 32, padding = "Same", kernel_size = 3, activation = "relu"))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 64, activation = "relu"))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(units = 10, activation = "softmax"))
# Optimizer

opt = tf.keras.optimizers.Adam(lr = 0.01)



# Compiling the Model

model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_accuracy",

                                                               patience = 3,

                                                               verbose = 1,

                                                               factor = 0.5,

                                                               min_lr = 0.00001)
model.summary()
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 10,

                                                          zoom_range = 0.1,

                                                          width_shift_range = 0.1,

                                                          height_shift_range = 0.1)



datagen.fit(features)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32), epochs = 25, validation_data = (x_val, y_val), callbacks = [learning_rate_reduction])
plt.plot(range(1, 26), history.history["loss"], label = "Loss")

plt.plot(range(1, 26), history.history["val_loss"], label = "Validation Loss")

plt.xlim([1, 25])

plt.xlabel("Epochs")

plt.ylim([0, 1])

plt.ylabel("Loss")

plt.title("CNN Loss")

plt.legend()

plt.show()
plt.plot(range(1, 26), history.history["accuracy"], label = "Accuracy")

plt.plot(range(1, 26), history.history["val_accuracy"], label = "Validation Accuracy")

plt.xlim([1, 25])

plt.xlabel("Epochs")

plt.ylim([0.7, 1])

plt.ylabel("Accuaracy")

plt.title("CNN Accuracy")

plt.legend()

plt.show()
y_val.shape
from sklearn.metrics import accuracy_score



y_pred = model.predict(x_val)

y_pred = np.argmax(y_pred, axis = 1)



y_val = np.argmax(y_val, axis = 1)



print("Accuracy = {:.3f}".format(accuracy_score(y_val, y_pred) * 100))
test_data = test_data.reshape(-1, 28, 28, 1)
preds = model.predict(test_data)

preds = np.argmax(preds, axis = 1)

preds = pd.Series(preds, name = "Label")
img_id = pd.Series(range(1, 28001), name = "ImageId")
# Concatenation

submission = pd.concat([img_id, preds], axis = 1)
# Saving the file in csv format without index

submission.to_csv("Digit_Recognizer_CNN_Datagen_1.csv", index = False)
from sklearn.metrics import confusion_matrix

import itertools



# Plotting the Confusion Matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_val, y_pred) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 