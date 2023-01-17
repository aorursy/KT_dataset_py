# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers.core import Lambda, Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Conv2D  , MaxPooling2D

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC  

from sklearn import tree
print(os.listdir("../input"))

print(os.getcwd())

#read the data and turn it into a dataframe

X_data = pd.read_csv("../input/train.csv") 

T_df = pd.read_csv("../input/test.csv") 

Y_df = X_data["label"]

X_df = X_data.drop("label", axis=1)

#%%

#extract just the values so it works with numpy

X = X_df.values

X = X/255 #We also take this moment to normalize the data. 

Y = Y_df.values



np.random.seed(100) #seet the seed so it is reproducible



#split our data into testing and training, for validation

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)





#Logistic regression and Bayes want Y to have a certain shape. 

Y_train_r = np.ravel(Y_train) 

Y_test_r = np.ravel(Y_test)



#our neural network wants Y to be categorical. 

Y_train_c = np_utils.to_categorical(Y_train, 10)

Y_test_c = np_utils.to_categorical(Y_test, 10)
#create classifier

KNN = KNeighborsClassifier(n_neighbors=1)

#fit classifier

KNN.fit(X_train,Y_train)

#score classifier

KNN.score(X_test, Y_test)



#create classifier

KNN_2 = KNeighborsClassifier(n_neighbors=2)

#fit classifier

KNN_2.fit(X_train,Y_train)

#score classifier

KNN_2.score(X_test, Y_test)
#SVC



svclassifier = SVC(kernel="linear")  

svclassifier.fit(X_train, Y_train)  



Y_pred = svclassifier.predict(X_test)

accuracy_score(Y_test, Y_pred)
svclassifier_rbf = SVC(kernel="rbf", gamma = "auto")  

svclassifier_rbf.fit(X_train, Y_train)  



Y_pred = svclassifier_rbf.predict(X_test)

accuracy_score(Y_test, Y_pred)
#Logistic regression. 



#first we create a list, that we will populate with fitted models 

Lreg = list()

for i in range(-3,4):

    c = 10**i #(0.001 ,0.01, 0.1, 1, 10, 100, 1000)

    #create our model

    dummy = LogisticRegression(multi_class="multinomial",solver="lbfgs", max_iter = 4000, C=c)

    #fit our model and add it to the list

    Lreg.append(dummy.fit(X_train,Y_train_r))    



#create a list of scores

Lreg_scores = list()

for i in range(0,6):

    #score our model and add it to the list. 

    Lreg_scores.append(Lreg[i].score(X_test, Y_test)) 



#print out list

print(Lreg_scores)
neural1 = Sequential() 

neural1.add(Dense(16, input_dim=784 , activation='relu')) 

neural1.add(Dense(10, activation='softmax'))



neural2 = Sequential() 

neural2.add(Dense(32, input_dim=784 , activation='relu')) 

neural2.add(Dense(10, activation='softmax'))



neural3 = Sequential() 

neural3.add(Dense(64, input_dim=784 , activation='relu')) 

neural3.add(Dense(10, activation='softmax'))



neural4 = Sequential() 

neural4.add(Dense(128, input_dim=784 , activation='relu')) 

neural4.add(Dense(10, activation='softmax'))



neural5 = Sequential() 

neural5.add(Dense(256, input_dim=784 , activation='relu')) 

neural5.add(Dense(10, activation='softmax'))



neural6 = Sequential() 

neural6.add(Dense(512, input_dim=784 , activation='relu')) 

neural6.add(Dense(10, activation='softmax'))



neural7 = Sequential() 

neural7.add(Dense(1024, input_dim=784 , activation='relu')) 

neural7.add(Dense(10, activation='softmax'))



neural8 = Sequential() 

neural8.add(Dense(2048, input_dim=784 , activation='relu')) 

neural8.add(Dense(10, activation='softmax'))



neural1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural6.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural7.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

neural8.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



neural1.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural2.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural3.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural4.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural5.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural6.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural7.fit(X_train, Y_train_c, epochs=50, batch_size=200)

neural8.fit(X_train, Y_train_c, epochs=50, batch_size=200)



scores1 = neural1.evaluate(X_test, Y_test_c)

scores2 = neural2.evaluate(X_test, Y_test_c)

scores3 = neural3.evaluate(X_test, Y_test_c)

scores4 = neural4.evaluate(X_test, Y_test_c)

scores5 = neural5.evaluate(X_test, Y_test_c)

scores6 = neural6.evaluate(X_test, Y_test_c)

scores7 = neural7.evaluate(X_test, Y_test_c)

scores8 = neural8.evaluate(X_test, Y_test_c)



plt.plot([1,2,3,4,5,6,7,8],

         [scores1[0],scores2[0],scores3[0],scores4[0],scores5[0],scores6[0],

                            scores7[0],scores8[0]],'-o')
plt.plot([1,2,3,4,5,6,7,8],[scores1[1],scores2[1],scores3[1],scores4[1],scores5[1],scores6[1],

                            scores7[1],scores8[1]],'-o')



X_train_2d = X_train.reshape(X_train.shape[0], 28, 28,1)

X_train_2d.shape

X_test_2d = X_test.reshape(X_test.shape[0], 28, 28,1)

X_test_2d.shape





model_1= Sequential()

model_1.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu', input_shape=(28,28,1)))

model_1.add(MaxPooling2D(pool_size=(2, 2)))

model_1.add(Flatten())

model_1.add(Dense(512, activation='relu'))

model_1.add(Dense(10, activation='softmax'))



model_2= Sequential()

model_2.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu', input_shape=(28,28,1)))

model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Conv2D(filters = 64, kernel_size = (5,5),activation ='relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Flatten())

model_2.add(Dense(512, activation='relu'))

model_2.add(Dense(10, activation='softmax'))



model_3= Sequential()

model_3.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu', input_shape=(28,28,1)))

model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Conv2D(filters = 64, kernel_size = (5,5),activation ='relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Conv2D(filters = 128, kernel_size = (3,3),activation ='relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Flatten())

model_3.add(Dense(512, activation='relu'))

model_3.add(Dense(10, activation='softmax'))



model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_1.fit(X_train_2d, Y_train_c, epochs=30, batch_size=200)

model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_2.fit(X_train_2d, Y_train_c, epochs=30, batch_size=200)

model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_3.fit(X_train_2d, Y_train_c, epochs=30, batch_size=200)



scores1 = model_1.evaluate(X_test_2d, Y_test_c)

scores2 = model_2.evaluate(X_test_2d, Y_test_c)

scores3 = model_3.evaluate(X_test_2d, Y_test_c)

print(scores1)

print(scores2)

print(scores3)
model_2= Sequential()

model_2.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu', input_shape=(28,28,1)))

model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Dropout(0.4))

model_2.add(Conv2D(filters = 64, kernel_size = (5,5),activation ='relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Dropout(0.4))

model_2.add(Flatten())

model_2.add(Dense(128, activation='relu'))

model_2.add(Dropout(0.4))

model_2.add(Dense(10, activation='softmax'))



model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_2.fit(X_train_2d, Y_train_c, epochs=30, batch_size=200)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train_2d)
history = model_2.fit_generator(datagen.flow(X_train_2d,Y_train_c, batch_size=200),

                              epochs = 25, validation_data = (X_test_2d,Y_test_c),

                              verbose = 2, steps_per_epoch=X_train.shape[0]/200)

scores2 = model_2.evaluate(X_test_2d, Y_test_c)

print(scores2)

T = T_df.values

T = T/255

T = T.reshape(T.shape[0], 28, 28,1)



results = model_2.predict(T)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")







submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission_kaggle_2D.csv",index=False)