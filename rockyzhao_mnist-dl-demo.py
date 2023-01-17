import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
train_file = "../input/train.csv"
test_file = "../input/test.csv"
output_file = "submission.csv"
# train set
train = pd.read_csv("../input/train.csv")
train_Y = train["label"]
train_X = train.drop(labels = ["label"],axis = 1)
# test set
test = pd.read_csv(test_file)
fig = sns.countplot(train_Y)
train_Y.value_counts()
test.isnull().any().describe()
# Normalization
train_X = train_X / 255.
# the image should be width = height = 28, channel = 1
# reshape
train_X = train_X.values.reshape(-1, 28, 28, 1)
print(train_X.shape)
# lable encoding to one-hot key
# example 2 -> [0,0,1,0,0,0,0,0,0,0]
train_Y = to_categorical(train_Y, num_classes=10)
train_Y[:5]
seed = 2
# split the train set into train set into train and dev set
train_X, dev_X, train_Y, dev_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=seed)
fig = plt.imshow(train_X[0][:,:,0])
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu',
                input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

# fully connection
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(zoom_range=0.1, height_shift_range=0.1, width_shift_range=0.1, rotation_range=10)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = model.fit_generator(datagen.flow(train_X, train_Y, batch_size=16),
                           steps_per_epoch=500,
                           epochs=20, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(dev_X[:400,:], dev_Y[:400,:]), #For speed
                           callbacks=[annealer])
final_loss, final_acc = model.evaluate(dev_X, dev_Y, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()
mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
x_test = mnist_testset.astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_hat,axis=1)
with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(y_pred)) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))








