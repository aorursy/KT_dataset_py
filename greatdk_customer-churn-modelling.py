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
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
churn=pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
churn.head()
churn.info()
churn.describe()
sns.set_style('whitegrid')
sns.countplot(churn['Geography'])
sns.countplot(churn['Gender'])
sns.distplot(churn['Age'],bins=50)
plt.title('Distribution of Age')
sns.countplot(churn['NumOfProducts'])
sns.countplot(churn['HasCrCard'])
sns.countplot(churn['IsActiveMember'])
churn.isnull().sum()
churn.duplicated().sum()
geography=pd.get_dummies(churn['Geography'],drop_first=True)
gender=pd.get_dummies(churn['Gender'],drop_first=True)
churn.drop(['Geography','Gender'],axis=1,inplace=True)
churn=pd.concat([churn,geography,gender],axis=1)
churn.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)
churn.head()
X=churn.drop(['Exited'],axis=1).values
X
y=churn['Exited'].values
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
# Import required models and layers and otheer important libraries
from keras.models import Sequential
from keras.layers import Dense
# Initializing the ANN
Classifier=Sequential()
# Adding layers
#input layer
Classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#hidden layers
Classifier.add(Dense(6, kernel_initializer = 'uniform',activation='relu'))
#Output layer
Classifier.add(Dense(1, kernel_initializer = 'uniform',activation='sigmoid'))
# compiling the model
Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
hist=Classifier.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=10,epochs=100)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['loss','val_loss'], loc='upper right')
plt.show()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy','val_accuracy'], loc='upper right')
plt.show()
predictions=Classifier.predict(X_test)
predictions
predictions=(predictions>0.5)
predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    Classifier=Sequential()
    Classifier.add(Dense(6,activation='relu'))
    Classifier.add(Dense(6,activation='relu'))
    Classifier.add(Dense(1,activation='sigmoid'))
    Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return Classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std()
mean
variance
from keras.layers import Dropout
Classifier=Sequential()
Classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
Classifier.add(Dropout(p=0.1))
Classifier.add(Dense(6, kernel_initializer = 'uniform',activation='relu'))
Classifier.add(Dropout(p=0.1))
Classifier.add(Dense(1, kernel_initializer = 'uniform',activation='sigmoid'))
Classifier.add(Dropout(p=0.1))
Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, kernel_initializer = 'uniform',activation='relu'))
    classifier.add(Dense(1, kernel_initializer = 'uniform',activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters={'batch_size':[25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator =classifier,param_grid = parameters,scoring = 'accuracy',cv= 10)
grid_search =grid_search.fit(X_train,y_train)
best_parameters =grid_search.best_params_
best_parameters
best_accuracy =grid_search.best_score_
best_accuracy
