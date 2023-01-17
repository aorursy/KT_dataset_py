from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import pandas as pd

from keras.datasets import mnist

import numpy as np

np.random.seed(0)
x_train = TODO

labels = TODO



x_test = TODO
x_train = x_train.values

y_train = labels.values

x_test = x_test.values
# TODO
print("the number of training examples = %i" % (TODO) )

print("the number of classes = %i" % (TODO) )

print("Flattened Image dimentions = %d x %d  " % (TODO)  )



#This line will allow us to know the number of occurrences of each specific class in the data

print("The number of occurances of each class in the dataset = %s " % TODO, "\n" )
X_train = TODO
# UNCOMMENT THE FOLLOWING SEGMENT

# images_and_labels = list(zip(X_train,  y_train))

# for index, (image, label) in enumerate(images_and_labels[:12]):

#     plt.subplot(5, 4, index + 1)

#     plt.axis('off')

#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

#     plt.title('label: %i' % label)
from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers import Dropout, Conv2D

from keras import regularizers



from keras.utils import np_utils



#reshape the inputs

# I will change the size of the training and testing sets to be able to use ImageDataGenerator wich accepts inputs in the following shape

x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)



print(x_train.shape )

print(x_train.shape )





#Makine the outputs 1-hot vector of 10 elements

y_train = TODO



model = Sequential()

# The first layer doesn't have significant importance in the code.

# THe conv layer is used only to get the 3d images from the fit generator in the 2d format and flatten it using flatten layer

# THe layer will not affect the layer since i am only using feature Pooling _ 1*1 convolution with only 1 feature map

model.add(Conv2D(1, kernel_size=1, padding="same",input_shape=(28, 28, 1), activation = 'relu'))

model.add(Flatten())



# model.add(Dense(units=800, activation='relu', input_dim= 784 ,  kernel_regularizer=regularizers.l2(0.001) ) )



model.add(Dense(units=100, activation='relu'  ))



#and now the output layer which will have 10 units to

#output a 1-hot vector to detect one of the 10 classes

model.add(Dense(units=10, activation='softmax'))
from keras import optimizers



# optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

x_train2 = np.array(x_train, copy=True) 

y_train2 = np.array(y_train, copy=True) 



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=10

    )





datagen.fit(x_train)



# # fits the model on batches with real-time data augmentation:

history = model.fit_generator(datagen.flow(x_train2, y_train2, batch_size=32),

                    steps_per_epoch=len(x_train) / 32, epochs = 25)
import matplotlib.pyplot as plt



plt.plot(TODO)
res = model.predict(x_test)

res = np.argmax(res,axis = 1)

res = pd.Series(res, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)

submission.to_csv("solution.csv",index=False)

submission.head(10)