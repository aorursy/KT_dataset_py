# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.layers import *

from keras.activations import *

from keras.models import *

from keras.optimizers import *

from keras.initializers import *

from keras.callbacks import *



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# data class

class MNIST:

    def __init__(self):

        

        #load data

        self.train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

        self.test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

         

        self.width = 28

        self.height = 28

        self.depth = 1

        self.test_size = 0.2

        self.num_classes = 10



        #train

        self.pd_x_train = self.train.drop(["label"], axis = 1)

        self.pd_y_train = self.train["label"].values



        self.np_x_train = np.array(self.pd_x_train)

        self.np_x_train = self.np_x_train.reshape(self.pd_x_train.shape[0], self.width, self.height, self.depth)



        self.np_y_train = np.array(self.pd_y_train)



        #test

        self.pd_x_test = self.test

        self.np_x_test = np.array(self.test)

        self.np_x_test = self.np_x_test.reshape(self.pd_x_test.shape[0], self.width, self.height, self.depth)

        

        # train test split

        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(self.np_x_train, self.np_y_train, test_size = self.test_size)

        self.x_test = self.np_x_test

        

        #Normalization

        #self.x_train

        #self.x_validation

        #self.x_test

        

        #One hot encoding

        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)

        self.y_validation = to_categorical(self.y_validation, num_classes=self.num_classes)

        

    def info(self):

        print("x_train: ", self.x_train.shape)

        print("x_validation: ", self.x_validation.shape)

        print("x_test: ", self.x_test.shape)

        print("y_train: ", self.y_train.shape)

        print("y_validation: ", self.y_validation.shape)
# Define the CNN

def create_model(optimizer, lr, width, height, depth, num_classes):

    

    input_img = Input(shape=(width, height, depth))



    x = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding="same")(input_img)

    x = BatchNormalization()(x)

    x = Activation("relu")(x) 

    x = MaxPool2D()(x)

    

    x = Conv2D(filters=16, kernel_size=6, strides=(1, 1), padding="same")(x)

    x = BatchNormalization()(x)

    x = Dropout(rate = 0.1)(x)

    x = Activation("relu")(x) 

    x = MaxPool2D()(x)

    

    x = Conv2D(filters=16, kernel_size=9, strides=(1, 1), padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x) 

    x = Dropout(rate = 0.1)(x)

    x = MaxPool2D()(x)

    

    x = Flatten()(x)

    x = Dense(128)(x)

    x = Activation("relu")(x)

    x = Dense(64)(x)

    x = Activation("relu")(x)

    x = Dense(num_classes)(x)

    output_pred = Activation("softmax")(x)

   

    optimizer = optimizer(lr=lr)

    model = Model(inputs=input_img, outputs=output_pred)

    model.compile(

        loss="categorical_crossentropy", 

        optimizer=optimizer, 

        metrics=["accuracy"])

    model.summary()

    

    return model
data = MNIST()

data.info()
#show example images



rows = 5

cols = 5



fig, axs = plt.subplots(rows,cols, figsize = (25,25))



for i in range(rows):

    for j in range(cols):      

        axs[i,j].imshow(data.x_train[rows*i+j].reshape(28,28))

        axs[i,j].set_title(np.argmax(data.y_train[rows*i+j]))

fig.show()
# hyperparameter

lr = 1e-2

optimizer = Adam

batch_size = 32

epochs = 15
# create model

model = create_model(optimizer, lr, data.width, data.height, data.depth, data.num_classes)
# define learning rate sheduler

def schedule(epoch):

    lr = 0.01/(epoch+1)



    return lr



lrs = LearningRateScheduler(

    schedule=schedule,

    verbose=1)
# training

history = model.fit(

    x=data.x_train, 

    y=data.y_train, 

    verbose=1, 

    #batch_size=batch_size, 

    epochs=epochs, 

    validation_data=(data.x_validation, data.y_validation),

    callbacks=[lrs])
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# store wrong predictions



pred = model.predict(data.x_validation)



# init

wrong = np.array([])



for k in range(pred.shape[0]):

    ClassId_pred = np.argmax(pred[k])

    ClassId_true= np.argmax(data.y_validation[k])

    if ClassId_pred != ClassId_true: 

        wrong = np.append(wrong, k)

        

print("Number of wrong predictions: ", wrong.size)

print("Percentage if wrong predictions: {0:.3f}".format((wrong.size/pred.shape[0])*100), "%")
#show examples of wrong predictions



rows = 5

cols = 5



label_true = "true"



fig, axs = plt.subplots(rows,cols, figsize = (25,25))



for i in range(rows):

    for j in range(cols):      

        axs[i,j].imshow(data.x_validation[int(wrong[rows*i+j])].reshape(28,28))

        axs[i,j].set_title("true: "+str(np.argmax(data.y_validation[int(wrong[rows*i+j])]))+"\n pred: "+str(np.argmax(pred[int(wrong[rows*i+j])])))

fig.show()
# predict results of x_test

results = model.predict(data.x_test)

results = np.argmax(results,axis = 1)



#create data frame

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.head(15)
submission.to_csv('submission.csv', index = False)