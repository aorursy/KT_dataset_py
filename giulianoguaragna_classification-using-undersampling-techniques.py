# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from sklearn import neighbors

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.model_selection import KFold



from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/bank-marketing-term-deposit/bank_customer_survey.csv")
data.head()
data.isnull().sum()
data.describe()
data[data['y']==0].describe()
data[data['y']==1].describe()
data[ data['age'] > 90 ].head()
data[ data['duration'] < 8 ]['y'].count()
from sklearn.preprocessing import LabelEncoder



newdata = data

le = LabelEncoder()

for col in newdata.columns:

    if(newdata[col].dtype == 'object'):

        newdata.loc[:,col] = le.fit_transform(newdata.loc[:,col])



newdata.head()
corr = newdata.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
X = newdata.iloc[:,:-1].values

y = newdata.iloc[:,-1].values

#If you want to try KNeighborsClassifier uncomment lines 13,21,and comment  14 and 22



scoresAc = []

scoresF1 = []



preds = []

actual_labels = []

# Initialise the 5-fold cross-validation

kf = KFold(n_splits=10,shuffle=True)





for i in range(1,15):

  #model1= neighbors.KNeighborsClassifier(n_neighbors = i)

  model2= RandomForestClassifier(max_depth=i ,n_estimators = 200,n_jobs= 5) 

  aux1 =[]

  aux2 = []

  for train_index,test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



    #estimator = model1.fit(X_train,y_train)

    estimator = model2.fit(X_train,y_train)

    

    predictions = estimator.predict(X_test)

    scoreF1 = metrics.f1_score(y_test,predictions)

    accuracy = metrics.accuracy_score(y_test,predictions)

    aux1.append(accuracy)

    aux2.append(scoreF1)

  

  scoresAc.append(np.average(aux1))

  scoresF1.append(np.average(aux2))



#print("F1 score: {0}".format((scoresF1)))



#print("accuracy score: {0}".format((scoresAc)))

report = classification_report(y_test, predictions)

print(report)



plt.plot(range(1,15), scoresAc, label="training accuracy")

plt.plot(range(1,15), scoresF1, label="F1 accuracy") 

plt.ylabel("score")

plt.xlabel("max_depth")

plt.legend()

#If you want to try SVC uncomment lines 21,22,and comment  18 and 19







scoresAc = []

scoresF1 = []



preds = []



# Initialise the 5-fold cross-validation

kf = KFold(n_splits=10,shuffle=True)



for train_index,test_index in kf.split(X):

  X_train, X_test = X[train_index], X[test_index]

  y_train, y_test = y[train_index], y[test_index]



  

  model3 = GaussianNB()

  estimator = model3.fit(X_train, y_train)



  #model4 = SVC(C=1000, kernel = "rbf")

  #estimator = model4.fit(X_train, y_train)



  predictions = estimator.predict(X_test)

  scoreF1 = metrics.f1_score(y_test,predictions)

  accuracy = metrics.accuracy_score(y_test,predictions)



  

scoresAc.append(accuracy)

scoresF1.append(scoreF1)



print("F1 score: {0}".format((np.average(scoreF1))))



print("accuracy score: {0}".format(np.average(scoresAc)))



report = classification_report(y_test, predictions)

print(report)

sizeY = data['y'].count()

print ("number of observations:",sizeY)



#using undersampling

sizeClass0=data[data['y']==0]['y'].count()

print ("number of observations with class 0:",sizeClass0)



sizeClass1=data[data['y']==1]['y'].count()

print ("number of observations with class 1:",sizeClass1)





#preprocesing

newdata= data

le = LabelEncoder()

for col in newdata.columns:

    if(newdata[col].dtype == 'object'):

        newdata.loc[:,col] = le.fit_transform(newdata.loc[:,col])





# We are going to create dt with class1 and 5000 elements of class0

# in testclass0 We save the remaining elements of class0

dataClass0 = newdata[newdata['y']==0]

dataClass1 = newdata[newdata['y']==1]



perm = np.random.permutation(sizeClass0)



# underSampling  with sizeClass1 



split_point = sizeClass1

#split_point = int(np.ceil(sizeClass0*0.5))



dataClass0 = dataClass0.values

dataClass1 = dataClass1.values





class0ForTrain = dataClass0[perm[:split_point].ravel(),:] 



testClass0 = dataClass0[perm[split_point:].ravel(),:] 



dt = np.concatenate((class0ForTrain,dataClass1))





print('length of dt contains class1 and 5000 elements of class0', len(dt))

print('length of remaining elements of class0 :', len(testClass0))



X = dt[:,:-1]

y = dt[:,-1]



XTestClass0 = testClass0[:,:-1]

yTestClass0 = testClass0[:,-1]

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

scoresAc = []

scoresF1 = []



preds = []



kf = KFold(n_splits=10,shuffle=True)



for train_index,test_index in kf.split(X):

  X_train, X_test = X[train_index], X[test_index]

  y_train, y_test = y[train_index], y[test_index]



  model1 = GaussianNB()

  estimator = model1.fit(X_train, y_train)



  #model1 = SVC(C=30, kernel = "rbf")

  #estimator = model1.fit(X_train, y_train)



  #model1= RandomForestClassifier(max_depth=16 ,n_estimators = 200) 

  #estimator = model1.fit(X_train, y_train)



  #model1 = neighbors.KNeighborsClassifier(n_neighbors = 2)

  #estimator = model1.fit(X_train, y_train)





  XfinalTest = np.concatenate((X_test,XTestClass0))

  yfinalTest = np.concatenate((y_test,yTestClass0))



  predictions = model1.predict(XfinalTest)



  scoreF1 = metrics.f1_score(yfinalTest,predictions)

  accuracy = metrics.accuracy_score(yfinalTest,predictions)



  #predictions = estimator.predict(X_test)

  #scoreF1 = metrics.f1_score(y_test,predictions)

  #accuracy = metrics.accuracy_score(y_test,predictions)



  

scoresAc.append(accuracy)

scoresF1.append(scoreF1)



print("F1 score: {0}".format((np.average(scoreF1))))



print("accuracy score: {0}".format(np.average(scoresAc)))



report = classification_report(yfinalTest, predictions)

print(report)
sizeY = data['y'].count()

print ("number of observations:",sizeY)



#undersampling

sizeClass0=data[data['y']==0]['y'].count()

print ("number of observations with class 0:",sizeClass0)



sizeClass1=data[data['y']==1]['y'].count()

print ("number of observations with class 1:",sizeClass1)





#preprocesing

newdata= data

le = LabelEncoder()

for col in newdata.columns:

    if(newdata[col].dtype == 'object'):

        newdata.loc[:,col] = le.fit_transform(newdata.loc[:,col])





# We are going to create dt with class1 and 19961 elements of class0

# in testclass0 We save the remaining elements of class0

dataClass0 = newdata[newdata['y']==0]

dataClass1 = newdata[newdata['y']==1]



perm = np.random.permutation(sizeClass0)



# underSampling  with class0ForTrain = sizeClass0 / 2





split_point = int(np.ceil(sizeClass0*0.5))



dataClass0 = dataClass0.values

dataClass1 = dataClass1.values





class0ForTrain = dataClass0[perm[:split_point].ravel(),:] 



testClass0 = dataClass0[perm[split_point:].ravel(),:] 



dt = np.concatenate((class0ForTrain,dataClass1))





print('length of dt contains class1 and 19961 elements of class0', len(dt))

print('length of remaining elements of class0 :', len(testClass0))



X = dt[:,:-1]

y = dt[:,-1]



XTestClass0 = testClass0[:,:-1]

yTestClass0 = testClass0[:,-1]
scoresAc = []

scoresF1 = []



preds = []



kf = KFold(n_splits=10,shuffle=True)



for train_index,test_index in kf.split(X):

  X_train, X_test = X[train_index], X[test_index]

  y_train, y_test = y[train_index], y[test_index]



  #model1 = GaussianNB()

  #estimator = model1.fit(X_train, y_train)



  #model1 = SVC(C=30, kernel = "rbf")

  #estimator = model1.fit(X_train, y_train)



  model1= RandomForestClassifier(max_depth=16 ,n_estimators = 200,n_jobs= 5) 

  estimator = model1.fit(X_train, y_train)



  #model1 = neighbors.KNeighborsClassifier(n_neighbors = 2)

  #estimator = model1.fit(X_train, y_train)





  XfinalTest = np.concatenate((X_test,XTestClass0))

  yfinalTest = np.concatenate((y_test,yTestClass0))



  predictions = model1.predict(XfinalTest)



  scoreF1 = metrics.f1_score(yfinalTest,predictions)

  accuracy = metrics.accuracy_score(yfinalTest,predictions)



  #predictions = estimator.predict(X_test)

  #scoreF1 = metrics.f1_score(y_test,predictions)

  #accuracy = metrics.accuracy_score(y_test,predictions)



  

scoresAc.append(accuracy)

scoresF1.append(scoreF1)



print("F1 score: {0}".format((np.average(scoreF1))))



print("accuracy score: {0}".format(np.average(scoresAc)))



report = classification_report(yfinalTest, predictions)

print(report)


from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical



X = newdata.iloc[:,:-1].values

y = newdata.iloc[:,-1].values



perm = np.random.permutation(y.size)



PRC = 0.80

split_point = int(np.ceil(y.shape[0]*PRC))

X_train = X[perm[:split_point].ravel(),:] 

y_train = y[perm[:split_point].ravel()]



X_test = X[perm[split_point:].ravel(),:]



y_test = y[perm[split_point:].ravel()]





y1 = to_categorical(y)



y_train1 = y1[perm[:split_point].ravel()]

y_test1 = y1[perm[split_point:].ravel()]



#We standardize features by removing the mean and scaling to unit variance

sc = StandardScaler()

X_trainS = sc.fit_transform(X_train)

X_testS = sc.fit_transform(X_test)


model = Sequential()



#get number of columns in training data

n_cols_2 = X.shape[1]



#add layers to model

model.add(Dense(250, activation='relu', input_shape=(n_cols_2,)))

model.add(Dense(250, activation='relu'))

model.add(Dense(250, activation='relu'))

model.add(Dense(2, activation='softmax'))





#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train1, epochs=20, batch_size=32)



predictions = model.predict(X_test)





y_pred1 = predictions[:,0] < 0.5



k = classification_report(y_test, y_pred1)

print (k)
#Same neural network but after standardize their features.

model.fit(X_trainS, y_train1, epochs=20, batch_size=32)



predictions = model.predict(X_testS)





y_pred1 = predictions[:,0] < 0.5



k = classification_report(y_test, y_pred1)

print (k)