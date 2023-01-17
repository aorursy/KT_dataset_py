import numpy as np

import keras as K

import tensorflow as tf

import pandas as pd

import math



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.externals import joblib



import xgboost as xgb

from sklearn.model_selection import train_test_split

seed = 78

test_size = 0.3

from sklearn.metrics import accuracy_score



from numpy import loadtxt

from xgboost import XGBClassifier



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras import optimizers



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras import optimizers

import keras as K

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

seed = 78

test_size = 0.33

import os





train = pd.read_csv("/kaggle/input/datamaestro2020/astro_train.csv")

test = pd.read_csv("/kaggle/input/datamaestro2020/astro_test.csv")

sample = pd.read_csv("/kaggle/input/datamaestro2020/sample_submission.csv")
del train["id"]

del test["id"]

Y = train["class"]

del train["class"]



del train["rerun"]

del test["rerun"]

del train["skyVersion"]

del test["skyVersion"]

del train["run"]

del test["run"]

del train["camCol"]

del test["camCol"]



### values donot change
train['dered_err_i'] = train["dered_i"] * train["err_i"]

train['dered_err_z'] = train["dered_z"] * train["err_z"]

train['dered_err_u'] = train["dered_u"] * train["err_u"]

train['dered_err_g'] = train["dered_g"] * train["err_g"]

train['dered_err_r'] = train["dered_r"] * train["err_r"]



test['dered_err_i'] = test["dered_i"] * test["err_i"]

test['dered_err_z'] = test["dered_z"] * test["err_z"]

test['dered_err_u'] = test["dered_u"] * test["err_u"]

test['dered_err_g'] = test["dered_g"] * test["err_g"]

test['dered_err_r'] = test["dered_r"] * test["err_r"]
#train.describe()
#test complete :- no NULL enteries
print(np.shape(train))

print(np.shape(test))
from sklearn import preprocessing

train = preprocessing.MinMaxScaler().fit_transform(train)

test = preprocessing.MinMaxScaler().fit_transform(test)



X_train, X_test, y_train, y_test = train_test_split(train, Y, test_size=0.3, random_state=2)



model = XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.4,

                      subsample = 1,

                      objective='multi:softmax', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=5, 

                      gamma=0.9,

                      num_class = 3)



model.fit(X_train, y_train)



# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

#print(np.shape(predictions))



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
y_pred = model.predict(test)

predictions = [round(value) for value in y_pred]





sample_submission = pd.read_csv("/kaggle/input/datamaestro2020/sample_submission.csv")



submission_df = pd.DataFrame(columns=['id', 'class'])

submission_df['id'] = sample_submission['id']

submission_df['class'] = predictions

submission_df.to_csv('XGB.csv', header=True, index=False)

submission_df.head(100)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

y_train = preprocessing.label_binarize(y_train, classes=[0, 1, 2])
random_state=42

gbr=GradientBoostingRegressor(random_state=random_state)

param_grid={ 

     "learning_rate": [0.01, 0.05 , 0.1 , 1],

    "max_depth":[5, 11, 17, 25, 37, 53],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "n_estimators":[10, 50, 200, 600, 1000]

    }

           

grid=GridSearchCV(gbr,param_grid = param_grid ,  verbose = 1, n_jobs = -1)

grid.fit(X_train,y_train)



print("Best Score:" + str(grid.best_score_))

print("Best Parameters: " + str(grid.best_params_))
from keras.utils import to_categorical



Y = to_categorical(Y, num_classes = 3)
all_data=[train,test]



for dataset in all_data:

    

    dataset['dered_i_range'] = pd.cut(dataset['dered_i'], bins=[8.013940, 19.922500, 21.040150, 21.970625, 34.928800 ], labels=[0,1,2,3])

    

    dataset['dered_z_range'] = pd.cut(dataset['dered_z'], bins=[7.091240, 19.526400, 20.567300, 21.746125, 30.933600 ], labels=[0,1,2,3])

    

    dataset['dered_u_range'] = pd.cut(dataset['dered_u'], bins=[10.007800, 21.839075, 22.834600, 24.217250	, 32.930400 ], labels=[0,1,2,3])

    

    dataset['dered_g_range'] = pd.cut(dataset['dered_g'], bins=[8.067590, 21.439600, 22.576600, 23.739500, 34.980600 ], labels=[0,1,2,3])



    dataset['dered_r_range'] = pd.cut(dataset['dered_r'], bins=[7.067860, 20.527900, 21.713250, 22.691825, 33.911000 ], labels=[0,1,2,3])

    

    dataset['err_i_range'] = pd.cut(dataset['err_i'], bins=[0.000070, 0.100398, 0.258745, 0.519050, 2550.904800 ], labels=[0,1,2,3])

    

    dataset['err_z_range'] = pd.cut(dataset['err_z'], bins=[0.000030, 0.230547, 0.520850, 1.011320, 752.907850 ], labels=[0,1,2,3])

    

    dataset['err_u_range'] = pd.cut(dataset['err_u'], bins=[ 0.000260, 0.531480, 0.933175, 1.578488, 302297.250000 ], labels=[0,1,2,3])

    

    dataset['err_g_range'] = pd.cut(dataset['err_g'], bins=[0.000180,0.166480, 0.419025	, 0.819457, 3688.869000 ], labels=[0,1,2,3])

    

    dataset['err_r_range'] = pd.cut(dataset['err_r'], bins=[0.000000, 0.106455, 0.288775, 0.582720, 2421.863800 ], labels=[0,1,2,3])

    

   

    
train = pd.get_dummies(train, columns = ["dered_i_range","dered_z_range","dered_u_range","dered_g_range","dered_r_range", "err_i_range", "err_z_range","err_u_range","err_g_range","err_r_range"],

                             prefix=[ "dered_i","dered_z","dered_u","dered_g","dered_r","err_i","err_z","err_u","err_g","err_r"])



test = pd.get_dummies(test, columns = ["dered_i_range","dered_z_range","dered_u_range","dered_g_range","dered_r_range", "err_i_range", "err_z_range","err_u_range","err_g_range","err_r_range"],

                             prefix=[ "dered_i","dered_z","dered_u","dered_g","dered_r","err_i","err_z","err_u","err_g","err_r"])
del train["dered_i"]

del test["dered_i"]

del train["dered_z"]

del test["dered_z"]

del train["dered_u"]

del test["dered_u"]

del train["dered_g"]

del test["dered_g"]

del train["dered_r"]

del test["dered_r"]



del train["err_i"]

del test["err_i"]

del train["err_z"]

del test["err_z"]

del train["err_u"]

del test["err_u"]

del train["err_g"]

del test["err_g"]

del train["err_r"]

del test["err_r"]
model = Sequential()



model.add(Dense(activation="relu", input_dim=26, units=32, kernel_initializer="uniform"))

#model.add(Dropout(0.2))



model.add(Dense(activation="relu", units= 24, kernel_initializer="uniform"))

#model.add(Dropout(0.2))



#model.add(Dense(activation="relu", units=18, kernel_initializer="uniform"))

#model.add(Dropout(0.2))



model.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))

#model.add(Dropout(0.2))



#model.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#model.add(Dropout(0.2))



model.add(Dense(activation="relu", units=3, kernel_initializer="uniform"))

#model.add(Dropout(0.2))





model.add(Dense(activation="softmax", units=3, kernel_initializer="uniform"))
#K.optimizers.Adamax(learning_rate=0.0002, beta_1=0.9, beta_2=0.999)



model.compile(loss='categorical_crossentropy',

              optimizer='adamax',

              metrics=['accuracy'])



from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,  factor=0.5,  min_lr=0.00001)



model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=8) #epochs = 500 for 70%
predictions = model.predict(test)

predict_class = np.argmax(predictions, axis=1)

predict_class = predict_class.tolist()





sample_submission = pd.read_csv("/kaggle/input/datamaestro2020/sample_submission.csv")



submission_df = pd.DataFrame(columns=['id', 'class'])

submission_df['id'] = sample_submission['id']

submission_df['class'] = predict_class

submission_df.to_csv('submissionsfinal.csv', header=True, index=False)

submission_df.head(10)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train
np.multiply(np.isnan(X_train))
# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 42)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

predictions = [np.round(value) for value in y_pred]

#print(np.shape(predictions))



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 42)

classifier.fit(train, Y)



y_pred = classifier.predict(test)

predictions = [np.round(value) for value in y_pred]



sample_submission = pd.read_csv("/kaggle/input/datamaestro2020/sample_submission.csv")



submission_df = pd.DataFrame(columns=['id', 'class'])

submission_df['id'] = sample_submission['id']

submission_df['class'] = predictions

submission_df.to_csv('submissionsfinal1.csv', header=True, index=False)

submission_df.head(10)
from sklearn.model_selection import validation_curve

param_range = np.arange(1, 250, 2)

train_scoreNum, test_scoreNum = validation_curve(

                                RandomForestClassifier(),

                                X = X_train, y = y_train, 

                                param_name = 'n_estimators', 

                                param_range = param_range, cv = 3)
from keras.models import Sequential 

from keras.layers import Dense, Activation 

output_dim = nb_classes = 10 

model = Sequential() 

model.add(Dense(3, input_dim=16, activation='softmax')) 

batch_size = 128 

nb_epoch = 200
model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy']) 

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3000,verbose=1, validation_data=(X_test, y_test)) 

score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])
from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

gnb = GaussianNB()



#Train the model using the training sets

gnb.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = gnb.predict(X_test)



#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 300, oob_score = True, n_jobs = -1,max_features = "auto", min_samples_leaf = 10)
model.fit(X_train,y_train)


pred=np.round(model.predict(X_test))

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

pred
print("Accuracy:",metrics.accuracy_score(y_test, pred))