import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint



from sklearn import preprocessing

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve, auc, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras import optimizers



import tensorflow as tf



import h5py
import os

os.listdir('../input/image-detection-using-nn')
file_path='../input/image-detection-using-nn/SVHN_single_grey1.h5';



inp_data = h5py.File(file_path,'r')



data=list(inp_data)

data
print(type(inp_data))
# The text is converted to array

X_test=np.array(inp_data['X_test'])

y_test=np.array(inp_data['y_test'])



X_train=np.array(inp_data['X_train'])

y_train=np.array(inp_data['y_train'])



X_val=np.array(inp_data['X_val'])

y_val=np.array(inp_data['y_val'])
print(X_test.shape)

print(y_test.shape)



print(X_train.shape)

print(y_train.shape)



print(X_val.shape)

print(y_val.shape)
# Plotting the image 



plt.imshow(inp_data['X_train'][20])
plt.figure(figsize=(18,10))

plt.subplot(2,2,1)

sns.countplot(np.array(inp_data['y_test']))

plt.subplot(2,2,2)

sns.countplot(np.array(inp_data['y_train']))

plt.subplot(2,2,3)

sns.countplot(np.array(inp_data['y_val']))
#Converting the y values to categorical column



y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)

y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)

y_val=tf.keras.utils.to_categorical(y_val,num_classes=10)

print(y_train.shape,y_train[4])
# Reshaping input to 1D vector



X_train=X_train.reshape(42000,1024)

X_test=X_test.reshape(18000,1024)

X_val=X_val.reshape(60000,1024)
print(X_train.shape)

print(X_test.shape)

print(X_val.shape)

print(y_train.shape)

print(y_test.shape)

print(y_val.shape)
#RandomizedSearchCV - KNN

#Implement Hyperparameter



def hyper_params_knn(X,y):



    # Create the random grid

    random_grid = {'n_neighbors':[7,10],

                  'leaf_size':[2],

                  'weights':['distance'],

                  'algorithm':['auto'],

                  'n_jobs':[-1]}



    pprint(random_grid)

    return random_grid



def randomizedsearch_knn(X_train,X_test,y_train,y_test):

# Use the random grid to search for best hyperparameters

# First create the base model to tune

    knn = KNeighborsClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

    rf_random = RandomizedSearchCV(estimator = knn, param_distributions = hyper_params_knn(X_train,X_test), n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)

    print("Best Hyper Parameters:",rf_random.best_params_)

    

    pred=rf_random.predict(X_test)

    score=rf_random.score(X_test,y_test)

    print("The model prediction is " + str(score*100) + "%")

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test, pred))

    print("the Classification report is")

    print(metrics.classification_report(y_test, pred))
# Since the X_train and X_val count is more, we take only 10k values.

randomizedsearch_knn(X_train[1:10000],X_val[1:10000], y_train[1:10000], y_val[1:10000])
neural_model=Sequential()



neural_model.add(Dense(250, input_shape = (1024,), activation = 'relu'))

neural_model.add(BatchNormalization())

neural_model.add(Dropout(0.5))

neural_model.add(Dense(10, activation = 'sigmoid'))



obj = optimizers.Adam(lr = 0.01 , beta_1=0.9 , decay =0)

neural_model.compile(optimizer = obj, loss = 'binary_crossentropy', metrics=['accuracy'])

neural_model.summary()



neural_model.fit(X_train, y_train, batch_size = 1000, epochs = 10, verbose = 1,validation_data=(X_val, y_val))

Y_pred_cls = neural_model.predict_classes(X_test, batch_size=200, verbose=0)



print('Accuracy') 

print( str(neural_model.evaluate(X_test,y_test)[1]) )

label=np.argmax(y_test.T, axis=0)

print(confusion_matrix(label, Y_pred_cls))

print(classification_report(Y_pred_cls, label))  
model = Sequential()



model.add(Dense(250, input_shape = (1024,), activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(125, input_shape = (250,), activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'sigmoid'))



sgd = optimizers.Adam(lr = 0.01 , beta_1=0.9 , decay =0)

model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])

model.summary()



model.fit(X_train, y_train, batch_size = 1000, epochs = 10, verbose = 1,validation_data=(X_val, y_val))

Y_pred_cls = model.predict_classes(X_test, batch_size=200, verbose=0)



print('Accuracy') 

print( str(model.evaluate(X_test,y_test)[1]) )



label=np.argmax(y_test.T, axis=0)

print(confusion_matrix(label, Y_pred_cls))

print(classification_report(Y_pred_cls, label))    

model = Sequential()



model.add(Dense(250, input_shape = (1024,), activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(250, input_shape = (250,), activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(125, input_shape = (250,), activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(10, activation = 'sigmoid'))

sgd = optimizers.Adam(lr = 0.01 , beta_1=0.9 , decay =0)



model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, batch_size = 1000, epochs = 50, verbose = 1,validation_data=(X_val, y_val))

Y_pred_cls = model.predict_classes(X_test, batch_size=200, verbose=0)



print('Accuracy') 

print( str(model.evaluate(X_test,y_test)[1]) )





label=np.argmax(y_test.T, axis=0)

print(confusion_matrix(label, Y_pred_cls))

print(classification_report(Y_pred_cls, label))    
plt.imshow(X_test[56].reshape(32,32))

print(label[56])

print(Y_pred_cls[56])
plt.imshow(X_test[66].reshape(32,32))

print(label[66])

print(Y_pred_cls[66])