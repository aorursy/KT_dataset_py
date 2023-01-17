import numpy as np
import keras
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator 
import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import sklearn.utils as shuffle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.callbacks import LearningRateScheduler
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
plt.figure(figsize = (15,15))
sns.catplot(x = 'label', kind = 'count' ,data = train, palette = "pastel")
plt.title("Distribution according to label")
plt.show()
y = train["label"]
train.drop(["label"], axis = 1, inplace = True)
train.head()
y.head()
y.unique()
y = np_utils.to_categorical(y, 10)
y.shape
train.shape
def image_show(train):
    fig = plt.figure(figsize = (20,20))
    fig.suptitle("Few Images from the dataset")
    for i in range(15):
        index = np.random.randint(train.shape[0])
        plt.subplot(10,10,i+1)
        plt.imshow(train[index][:,:, 0])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.show()  
train = train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
image_show(train)
train = train / 255
test = test / 255
image_generator= ImageDataGenerator(rotation_range = 10,zoom_range = 0.10,width_shift_range=0.1,height_shift_range=0.1)

model = [0] * 10
for i in range(10):
    model[i] = Models.Sequential()
    model[i].add(Layers.Conv2D(64, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Dropout(0.4))
    model[i].add(Layers.Conv2D(128, kernel_size = 3, activation='relu'))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Conv2D(128, kernel_size = 3, activation='relu'))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Dropout(0.4))
    model[i].add(Layers.Conv2D(256, kernel_size = 4, activation='relu'))
    model[i].add(Layers.BatchNormalization())
    model[i].add(Layers.Flatten())
    model[i].add(Layers.Dense(512, activation = 'relu'))
    model[i].add(Layers.Dropout(0.4))
    model[i].add(Layers.Dense(10, activation='softmax'))
    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
call_back =  LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * 10
epochs = 30
for i in range(10):
    train_x, val_x, train_y, val_y = train_test_split(train, y, test_size = 0.1)
    history[i] = model[i].fit_generator(image_generator.flow(train_x,train_y, batch_size= 64),
        epochs = 30, steps_per_epoch = (train_x.shape[0]// 64) ,  
        validation_data = (val_x,val_y), callbacks=[call_back], verbose= 0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        i+1,epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))

results = np.zeros((test.shape[0],10)) 
for i in range(10):
    results = results + model[i].predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST.csv",index=False)