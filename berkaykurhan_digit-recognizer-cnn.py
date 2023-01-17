

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import random



df = pd.read_csv('../input/digit-recognizer/train.csv')



#Explore the data and split

print(df.columns)

print(df.head)

print(df.dtypes)

# We can see that our data's first column is the label and pixels the rest.

#There are 784 pixels(28*28) and all of them contains only int64 values.

X = df.iloc[:,1:]

y = df.iloc[:,0]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#We need to see whether our data contains null values.

print(pd.isnull(X_train[:]).any().describe())

print(pd.isnull(X_test[:]).any().describe())

#We don't have any null values.
#We need to reshape our data to 3D

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

print(X_train.shape) #We have 37800 images for training which are 28*28 pixels
#Let's make some visualisation to understand the data better.

plt.bar(y_train.value_counts().index,y_train.value_counts()) #we can see that our data disturbuted well.

plt.title('Disturbution of our labels')

plt.show()

plt.subplot(3,3,1)

plt.imshow(X_train[random.randint(0, 37800)].reshape(28,28))

plt.title('Random Image ')

plt.show()
# We got to Normalize our data. Since we have grayscale data our pixel values between 0-255 so, dividing 255 would be enough to normalize

X_train = X_train/255

X_test= X_test/255
#We need to convert our labels into categorical data:

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)
# We need to do Data Augmentation in order to avoid overfitting. It means we will  increase diversity by playing our images(shifting,zoom,rotate,etc.)

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





datagen.fit(X_train)
# We succesfully pre processed our data. Now we can build our CNN model.

#Our model will be like  [Conv2D->relu -> MaxPool2D -> Dropout] -> Flatten -> Dense -> Dropout -> Out

from tensorflow import keras

from tensorflow.keras import layers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

#Initialising the CNN

model = Sequential()

model.add(layers.Conv2D(filters=256,kernel_size=3,activation='relu',input_shape=[28,28,1]))  # 1 is our canal number it is just 1 because we use grayscale data

model.add(layers.Conv2D(filters=256,kernel_size=3,activation='relu'))



#I preffered relu activation because image data sets are not linear mostly.

#32 feature detector will be enough for this model. And size of the each will be 3(3,3)

#Pooling

model.add(layers.MaxPool2D(pool_size=2,strides=2)) #I preffered Max Pooling for this model

model.add(Dropout(0.2))



#Second Layer

model.add(layers.Conv2D(filters=256,kernel_size=3,activation='relu'))

model.add(layers.Conv2D(filters=256,kernel_size=3,activation='relu'))

model.add(layers.MaxPool2D(pool_size=2,strides=2))

model.add(Dropout(0.2))







#Flattening and bulding ANN



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) #Our output layer occurs 10 neurons since we have 10 categorical variables. And we use softmax activation

# Now we need to choose loss function, optimizer and compile the model

from keras.optimizers import RMSprop # I preferred RMSprop because of it's speed but you can use Stochastic Gradient Descent instead.

model.compile(optimizer=RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0),loss='categorical_crossentropy',metrics=['accuracy'])

#
# Now we fit our model

model.fit_generator(datagen.flow(X_train,y_train, batch_size=86),epochs = 100, validation_data = (X_test,y_test),verbose = 1,steps_per_epoch=len(X_train) // 50,callbacks=None)

#Making Predictions on Test data

predicted = model.predict(X_test)

y_head = predicted.argmax(axis=1).reshape(-1,1)
#Evaluation(accuracy and confusion matrix)

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

print(accuracy_score(y_test.argmax(axis=1), y_head))

cm = confusion_matrix(y_test.argmax(axis=1),y_head)

sns.heatmap(cm, annot=True)
# Submission Preaparation

sample_set = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

test_set = pd.read_csv('../input/digit-recognizer/test.csv')

test = test_set.values.reshape(-1,28,28,1)

# Making Predictions on Test Data

y_head_test = model.predict(test)

result_test = y_head_test.argmax(axis=1)

#Visualising predictions

for i in range(1,5):

    index = np.random.randint(1,28001)

    plt.subplot(3,3,i)

    plt.imshow(test[index].reshape(28,28))

    plt.title("Predicted Label : {} ".format(result_test[index]))

plt.subplots_adjust(hspace = 1.2, wspace = 1.2)

plt.show()
result_test = pd.Series(result_test,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result_test],axis = 1)

submission.to_csv("submission.csv",index=False)