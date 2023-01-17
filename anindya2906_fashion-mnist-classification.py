import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

from keras.utils import to_categorical
fashion_train_df = pd.read_csv('../input/fashion-mnist-datasets/fashion-mnist_train.csv', sep=',')

fashion_test_df = pd.read_csv('../input/fashion-mnist-datasets/fashion-mnist_test.csv', sep=',')
fashion_train_df.shape   # Shape of the dataset
fashion_train_df.columns   # Name of the columns of the DataSet.
print(set(fashion_train_df['label']))
print([fashion_train_df.drop(labels='label', axis=1).min(axis=1).min(), 

      fashion_train_df.drop(labels='label', axis=1).max(axis=1).max()])
fashion_train_df.head()
fashion_test_df.shape
fashion_test_df.head()
# Convert the dataframe ti numpy array

training = np.asarray(fashion_train_df, dtype='float32')



# Lets show multiple images in a 15x15 grid

height = 10

width = 10



fig, axes = plt.subplots(nrows=width, ncols=height, figsize=(17,17))

axes = axes.ravel()  # this flattens the 15x15 matrix into 225

n_train = len(training)



for i in range(0, height*width):

    index = np.random.randint(0, n_train)

    axes[i].imshow(training[index, 1:].reshape(28,28))

    axes[i].set_title(int(training[index, 0]), fontsize=8)

    axes[i].axis('off')

    

plt.subplots_adjust(hspace=0.5)
# convert to numpy arrays and reshape

training = np.asarray(fashion_train_df, dtype='float32')

X_train = training[:, 1:].reshape([-1,28,28,1])

X_train = X_train/255   # Normalizing the data

y_train = training[:, 0]



testing = np.asarray(fashion_test_df, dtype='float32')

X_test = testing[:, 1:].reshape([-1,28,28,1])

X_test = X_test/255    # Normalizing the data

y_test = testing[:, 0]
# Split the training set into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=12345)    # TODO : change the random state to 5
# Lets check the shape of all three datasets

print(X_train.shape, X_val.shape, X_test.shape)

print(y_train.shape, y_val.shape, y_test.shape)
cnn_model = Sequential()

cnn_model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size = (2,2)))

cnn_model.add(Dropout(rate=0.3))

cnn_model.add(Flatten())

cnn_model.add(Dense(units=32, activation='relu'))

cnn_model.add(Dense(units=10, activation='sigmoid'))
cnn_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.summary()
cnn_model.fit(x=X_train, y=y_train, batch_size=512, epochs=50, validation_data=(X_val, y_val))
eval_result = cnn_model.evaluate(X_test, y_test)

print("Accuracy : {:.3f}".format(eval_result[1]))
y_pred = cnn_model.predict_classes(x=X_test)
height = 10

width = 10



fig, axes = plt.subplots(nrows=width, ncols=height, figsize=(20,20))

axes = axes.ravel()

for i in range(0, height*width):

    index = np.random.randint(len(y_pred))

    axes[i].imshow(X_test[index].reshape((28,28)))

    axes[i].set_title("True Class : {:0.0f}\nPrediction : {:d}".format(y_test[index],y_pred[index]))

    axes[i].axis('off')

plt.subplots_adjust(hspace=0.9, wspace=0.5)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,5))

sbn.heatmap(cm, annot=True)
num_classes = 10

class_names = ["class {}".format(i) for i in range(num_classes)]

cr = classification_report(y_test, y_pred, target_names=class_names)

print(cr)