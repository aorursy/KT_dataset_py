# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the datasets

import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test_set = pd.read_csv("../input/digit-recognizer/test.csv")

training_set = pd.read_csv("../input/digit-recognizer/train.csv")
# Displaying the shape and datatype of each attribute



print(training_set.shape)

training_set.dtypes
# Histogram Visualisation for Output attribute



import seaborn as sb

sb.distplot(training_set['label'])

# Displaying Null values info in each column



training_set.info()
# Displaying the sum count of null or empty values in each count..Due too many columns unable to view

training_set.isna().sum()
# Now we going to display only those column which have null value and remaining columns wont display

training_set.isnull().any().describe()
test_set.isnull().any().describe()
y_train=training_set['label'].values

x_train=training_set.drop(['label'],axis=1)
x_train=x_train/255.0

test_set=test_set/255.0
x_train=x_train.values.reshape(-1,28,28,1)

x_test=test_set.values.reshape(-1,28,28,1)

del test_set

del training_set
# Encoding numerics in one hot encoder vector

'''from sklearn.preprocessing import OneHotEncoder

onehotencoder=OneHotEncoder(Categorical_features=[0])

y_train=onehotencoder.fit_transform(y_train)



TypeError: __init__() got an unexpected keyword argument 'Categorical_features' some api issue there

'''

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



y_train=to_categorical(y_train,num_classes=10)
# split the training set into train and test set



seed=5

train_size=0.80

test_size=0.20



from sklearn.model_selection import train_test_split

x1_train,x1_test,y1_train,y1_test=train_test_split(x_train,y_train,train_size=train_size,test_size=test_size,random_state=seed)
# Building CNN layers



# intialisaing the sequence of layers

from keras.models import Sequential

cnn=Sequential()



# Building fist Convolutional Layer

from keras.layers import Convolution2D

from keras.layers import Dropout

cnn.add(Convolution2D(input_shape=(28,28,1),activation='relu',filters=32,kernel_size=(5,5)))

cnn.add(Dropout(0.2))



# Building first pooling Layer

from keras.layers import MaxPooling2D

cnn. add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Dropout(0.2))



# Building Second Colvolution and pooling layers

cnn.add(Convolution2D(kernel_size=(5,5),filters=32,activation='relu'))

cnn.add(Dropout(0.2))

cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Dropout(0.2))



# Building Third Convolution and pooling layer

cnn.add(Convolution2D(kernel_size=(3,3),filters=32,activation='relu'))

cnn.add(Dropout(0.2))

cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Dropout(0.2))



# Building Flatten layer

from keras.layers import Flatten

cnn.add(Flatten())



# Building Fully Connected Layers

from keras.layers import Dense

# First fully connected hidden layer

cnn.add(Dense(256,activation='relu'))

cnn.add(Dropout(0.2))

# Second fully connected hidden layer

cnn.add(Dense(256,activation='relu'))

cnn.add(Dropout(0.2))

# Third Fully connected hidden layer

cnn.add(Dense(128,activation='relu'))

cnn.add(Dropout(0.2))



# Output layer with 10 neurons

cnn.add(Dense(10,activation='softmax'))
# compile the CNN Model

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
result=cnn.fit(x1_train, y1_train, batch_size = 32, epochs = 10,validation_data = (x1_test,y1_test), verbose = 2)
# Create Data Augmentation Generator

from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,# randomly shift images

                          horizontal_flip=False,vertical_flip=False,# randomly flip images

                          rotation_range=10,# randomly rotate images in the range (degrees, 0 to 180)

                          #brightness_range=[0.1,1.0],# randomly brightning images

                          zoom_range=0.1,# Randomly zoom image

                          zca_whitening=False,# apply ZCA whitening

                          featurewise_center=False,  # set input mean to 0 over the dataset

                          samplewise_center=False,  # set each sample mean to 0

                          featurewise_std_normalization=False,  # divide inputs by std of the dataset

                          samplewise_std_normalization=False,  # divide each input by its std

                          )



datagen.fit(x1_train)
batch_size=86

# makeing iteration flow

sample=datagen.flow(x1_train,y1_train,batch_size=batch_size)

# fit and generate the outcome

train_predictions=cnn.fit_generator(sample,epochs=10,validation_data=(x1_test,y1_test))
# predicting the training set test accuracy

import numpy as np

y_trainpred=cnn.predict(x1_test)

# Convert predictions classes to one hot vectors 

y_pred_one=np.argmax(y_trainpred,axis=1)

# Convert validation observations to one hot vectors

y1_test_one=np.argmax(y1_test,axis=1)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y1_test_one,y_pred_one)
print(accuracy)
# Predict the result for test set

y_pred=cnn.predict(x_test)
# argmax() is used to decode the onehotencoder value to numerical value

result=np.argmax(y_pred,axis=1)

# storing those value as a column name label

result=pd.Series(result,name='label')
# Submission 

submission=pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)

submission.to_csv("My_Submission.csv",index=False)

print("Submission Successfully")