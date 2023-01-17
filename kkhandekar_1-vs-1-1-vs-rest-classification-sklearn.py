#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



#SK Learn Libraries

import sklearn

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier   #1vs1 & 1vsRest Classifiers

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score



import gc
#Load Data

url = '../input/iris/Iris.csv'

data = pd.read_csv(url, header='infer')

data.drop('Id',axis=1,inplace=True)
#Records

print("Total Records: ", data.shape[0])
#Records per Species

data.Species.value_counts()
#Stat Summary

data.describe().transpose()
#Inspect

data.head()
#Encoding Species columns (to numerical values)

data['Species'] = data['Species'].astype('category').cat.codes
#Feature & Target Selection

features = data.select_dtypes('float').columns

target = ['Species']



# Feature& Target  Dataset

X = data[features]

y = data[target]
#Split Parameters

test_size = 0.1



#Dataset Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0) 



#Feature Scaling

#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)

#X_test = sc.transform(X_test)



#Reset Index

X_test = X_test.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)
#SVC Model

model = SVC(gamma='scale',random_state=0)



#Define 1-vs-1 Strategy / Classifier

ovo = OneVsOneClassifier(model)



#fit model to training data

ovo.fit(X_train, y_train)



#Predications

ovo_pred = ovo.predict(X_test)



#Adding Predictions to Test Dataset

ovo_df = X_test.copy()

ovo_df.insert(4,"Actual",y_test, True)

ovo_df.insert(5,"Predicted",ovo_pred, True)
#Inspect Test Dataset

ovo_df.head()
#Define 1-vs-Rest Strategy / Classifier

ovr = OneVsRestClassifier(model)



#fit model to training data

ovr.fit(X_train, y_train)



#Predications

ovr_pred = ovr.predict(X_test)



#Adding Predictions to Test Dataset

ovr_df = X_test.copy()

ovr_df.insert(4,"Actual",y_test, True)

ovr_df.insert(5,"Predicted",ovr_pred, True)
#Inspect

ovr_df.head()