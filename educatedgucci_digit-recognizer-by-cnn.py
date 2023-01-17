# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from keras.utils import to_categorical

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



x = train.drop(labels = ["label"],axis = 1)

y = train["label"]
x = np.array(x)

e =x[10000]

image = e.reshape(28,28)

plt.imshow(image, cmap = plt.cm.binary,

           interpolation = 'nearest')

plt.axis('off')

plt.show()

print(y[10000])
plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(x[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()



ans = []

for i in range(30):

    ans.append(y[i])

print(ans)
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 123)



xtrain.shape, xtest.shape, ytrain.shape, ytest.shape
xtrain = xtrain.reshape(-1, 28, 28, 1)

xtest = xtest.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1,28,28,1)
xtrain = xtrain.astype("float32")/255

xtest = xtest.astype("float32")/255

test = test.astype("float32")/255

ytrain = to_categorical(ytrain, num_classes=10)

ytest = to_categorical(ytest, num_classes=10)

# CNN MODEL



model = Sequential()



model.add(Conv2D(filters = 32,

                 kernel_size = (3,3),padding = 'Same', activation ='relu', 

                 input_shape = (28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Dense(10, activation = "softmax"))
# Model Visulization

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
# Optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )

# Compiling the model

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()




from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False,  

        featurewise_std_normalization=False, 

        samplewise_std_normalization=False, 

        zca_whitening=False,  

        rotation_range=10, 

        zoom_range = 0.1,

        width_shift_range=0.1,  

        height_shift_range=0.1, 

        horizontal_flip=False,  

        vertical_flip=False) 

datagen.fit(xtrain)

# Fitting the model

batch_size = 64

epochs = 50

reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)



history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size = batch_size), epochs = epochs, 

                              validation_data = (xtest, ytest), verbose=2, 

                              steps_per_epoch=xtrain.shape[0] // batch_size,

                              callbacks = [reduce_lr])

model.evaluate(xtest, ytest)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Test'])

plt.show()
testprediction=np.argmax(model.predict(test),axis=1)

testimage=[]

for i in range (len(testprediction)):

    testimage.append(i+1)

final={'ImageId':testimage,'Label':testprediction}

submission=pd.DataFrame(final)

submission.to_csv('submssion.csv',index=False)