# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/churn-predictions-personal/Churn_Predictions.csv")
df.head()
df.columns
df.info
df.describe()
df.columns
# Independent variables

X = df.iloc[:,3:13]

# Dependent variable

y = df.iloc[:,13]
y
print(X.shape)

print(y.shape)
X.head()
df.columns
dummy=pd.get_dummies(X[[ 'Geography','Gender']])

X=pd.concat([X,dummy],axis=1)





X.drop([ 'Geography','Gender'],axis = 1,inplace = True)
X.head()
%time

# Splitting the dataset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=1)
print("X_train :-{} \nX_test :-{}\ny_train :-{}\ny_test :-{}".format(X_train.shape,

                                                             X_test.shape,y_train.shape,y_test.shape))
X_train.astype(int)
y_train
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu',input_dim =13))

classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu'))

classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=1,init ='uniform',activation = 'sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=100,batch_size=10)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
(2296+77)/(2296+77+419+208)
cm
(2308+194)/(2308+65+433+194)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

def buildclassifier():

    classifier = Sequential()

    classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu',input_dim =13))

    classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu'))

    classifier.add(Dense(output_dim=1,init ='uniform',activation = 'sigmoid'))

    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return classifier
classifier = KerasClassifier(build_fn=buildclassifier,epochs=100,batch_size=10)
accuracies = cross_val_score(estimator=classifier,X = X_train,y =y_train,cv=10,n_jobs=-1)
accuracies
mean = accuracies.mean()

mean
std = accuracies.std()

std
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

def buildclassifier(optimize):

    classifier = Sequential()

    classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu',input_dim =13))

    classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu'))

    classifier.add(Dense(output_dim=1,init ='uniform',activation = 'sigmoid'))

    classifier.compile(optimizer=optimize,loss='binary_crossentropy',metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=buildclassifier)

parameters = { 'batch_size':[25,34],

             'epochs':[100,500],

             'optimize': ['adam','rmsprop']}

gridsearch = gridsearch.fit(X_train,y_train)



gridsearch = GridSearchCV(estimator=classifier,

                         param_grid=parameters,

                         cv =10,

                         scoring='accuracy') 

best_param = gridsearch.best_params_

best_accuracy = gridsearch.best_score_