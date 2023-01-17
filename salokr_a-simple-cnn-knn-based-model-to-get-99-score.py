# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd
#Let's import some important Libraries

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

#Now, let's load the input data and visualize them for illustration

train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')

'''

Let's observe the training_data for understanding, from the column labels we can see that the 

first column actually represents the labels for the training data and rest are the pixel values

'''

train_data.head()
#Similarly, we have test_data but without labels

test_data.head()

#Let's also see the dimension of the data

print('Training Data : '+str(train_data.shape))

print('Testing Data : '+ str(test_data.shape))
#load Train_Data, features are from 1 to rest of columns

X_train=train_data.values.astype('float32')[:,1:]

#the first column is label

y_train=train_data.values.astype('int32')[:,:1]

#the Test Data

X_test=test_data.values.astype('float32')

#Plot First Few Instances

for i in range(0,3):

    plt.subplot(330+i+1)

    #REshape to 2D for plotting, let's plot Image Number: 100,101,102

    test_img=X_train[100+i].reshape(28,28)

    plt.imshow(test_img, cmap=plt.get_cmap('gray'))

    plt.title('Label'+str(y_train[100+i]))

#Preprocessing Step (https://en.wikipedia.org/wiki/Feature_scaling#Standardization)

std=np.std(X_train)

mean=np.mean(X_train)



X_train-=mean

X_test-=mean

X_train/=std

X_test/=std
#Preparing the labels, converting 'em to one-hot vectors

y_train=np_utils.to_categorical(y_train)

#Number of features

cols=X_train.shape[1]

#Number of labels

labels=y_train.shape[1]
#Building a very simple model

from keras.models import Sequential

from keras.layers import Dense

model=Sequential()

model.add(Dense(cols,input_shape=(cols,),activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(256,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))#try linear, tanh and other combinations here





'''

#Install and import pydot to viualize the model

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

'''



model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=33,batch_size=512,validation_split=0.33)

#Let's plot the curves for study

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('#f Iterations')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
#From the curve we are looking OK, not overfitting, Let's see the loss plot

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('#f Iterations')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

#By submitting the above model we can achieve a good score, but can we do better ?

'''

#To get the first set of results

predictions = model.predict_classes(X_test,verbose=0)

submissions=pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),"Label":predictions})

submissions.to_csv("results.csv",index=False,header=True)

#Upload to get our first score YEEEEEE!!!!

'''
'''

Create a simple CNN model

Ref: CNN

http://colah.github.io/posts/2014-07-Understanding-Convolutions/

http://cs231n.github.io/convolutional-networks/

http://neuralnetworksanddeeplearning.com/chap6.html#introducing_convolutional_networks

http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

http://colah.github.io/posts/tags/convolutional_neural_networks.html

https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/

https://brohrer.github.io/how_convolutional_neural_networks_work.html

https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/

'''
from keras.layers import *

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

#Image Dimensions

width=28

height=28

channels=1



model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(width,height,channels)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))

model.add(Activation('relu'))

#We are naming this layer, as we will extract features from it in future, JUST WAIT !!!

model.add(MaxPooling2D(pool_size=(2,2),name='feature_layer'))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(labels))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#Let's prepare the data for 2D-CNN

X_Train=X_train.reshape(X_train.shape[0],28,28,1)

X_Test=X_test.reshape(X_test.shape[0],28,28,1)

history=model.fit(X_Train,y_train,epochs=33,batch_size=512,validation_split=0.33)



#Again, let's see the Accuracy Plot again, I could have created a function for these,but let's do it this way

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('#f Iterations')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
#Again we are GOOD!!! , Let's see the loss plot, the curve seems to be a smoother one

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('#f Iterations')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

#By submitting the above model we can achieve a good score, but can we do better ?

'''

#Submit the results

predictions = model.predict_classes(X_test,verbose=0)

submissions=pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),"Label":predictions})

submissions.to_csv("results.csv",index=False,header=True)

#Upload to get a better score YEEEEEE!!!!

'''
'''

Now, we are going to extract the features extracted from the CNN, the last layer of

CNN is supposed to learn a good representation of image and accordingly could have adjusted the weights

why should not obtain these features, and feed them to some other Machine Learning Models

like SVM, KNN etc.

Let's see our model first, the last CNN layer which we named it as 'feature_layer'(before Reshape layer)

we are going to pull it out, and then obtain the input representations from it

'''

model.summary()
#Extract the hidden state representations(Just a fancy name, call it embeddings or whatever you want)

from keras.models import Model

new_model=Model(inputs=model.input,outputs=model.get_layer('feature_layer').output)

#Let's obtain the Input Representations

train_x=new_model.predict(X_Train)

X_test=X_test.reshape(X_test.shape[0],28,28,1)

test_x=new_model.predict(X_test)

#Convert back the labels

train_y=[ np.where(r==1)[0][0] for r in y_train ]

#We are now going to have a single row for each example

train_x=train_x.reshape(42000,5*5*64)

test_x=test_x.reshape(28000,5*5*64)

'''

#Let's Try SVM First

from sklearn.svm import SVC

svm=SVC()

svm.fit(train_x,train_y)

svm.score(train_x,train_y)

svm_predict=svm.predict(test_x)

submissions=pd.DataFrame({"ImageId":list(range(1,len(knn_predict)+1)),"Label":svm_predict})

submissions.to_csv("results_svm.csv",index=False,header=True)

#The SVM requires lot's of parameter tuning and optimizations (due to noise), so we may not get the best results expected here

'''



#K_nearest Classifier



from sklearn.neighbors import KNeighborsClassifier

knc=KNeighborsClassifier()

knc.fit(train_x,train_y)

knc.score(train_x,train_y)

knn_predict=knc.predict(test_x)

submissions=pd.DataFrame({"ImageId":list(range(1,len(knn_predict)+1)),"Label":knn_predict})

submissions.to_csv("results.csv",index=False,header=True)



#NOTE: The codes are commented because they take very long time to execute, feel free to run them

'''

The results for me were the best out of all the results I got(But too much time consumed to 

get the results), you may have to modify

the network inorder to get the better results like changing the number of CNN layers, modify the architecture in the

way you want.

Tha main objective of my post was to illustrate how we can remove the last Dense layer

and replace it with, other machine learning algorithms.

'''

#Let's plot The curves here



train_sizes, train_scores, test_scores = learning_curve(knc, train_x, train_, n_jobs=-1, cv=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



plt.figure()

plt.title("K-NN")

plt.legend(loc="best")

plt.xlabel("Training examples")

plt.ylabel("Score")

plt.gca().invert_yaxis()



plt.grid()



plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")



plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")



plt.ylim(-.1,1.1)

plt.show()
