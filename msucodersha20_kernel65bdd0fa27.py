# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn.metrics as sm

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/diabetes/diabetes.csv')
dataset

dataset.describe()
print((dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]==0).sum())
dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NAN)
print(dataset.isnull().sum())
dataset.fillna(dataset.mean(),inplace=True)
print(dataset.isnull().sum())
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
dataTransform=dataset.copy()
for data in dataset.columns:

    dataTransform[data]=labelencoder.fit_transform(dataset[data])
dataTransform

X = dataTransform.drop(['Outcome'], axis=1)
X

Y = dataTransform['Outcome']
Y

datafeatures=list(X.columns)
datafeatures

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=41)
X_train.shape

Y_train.shape
from sklearn.ensemble import RandomForestClassifier
randomforestclassifier=RandomForestClassifier(n_estimators=1200)
randomforestclassifier.fit(X_train,Y_train)
predicition_y=randomforestclassifier.predict(X_test)
predicition_y
experiment_accuracy=sm.accuracy_score(Y_test,predicition_y)

print('Accuracy Score:',str(experiment_accuracy))
from sklearn import metrics

print("Classification Report: ",metrics.classification_report(predicition_y,Y_test,target_names=["HAVE","NOT HAVE"]))
from sklearn.metrics import confusion_metrix
from sklearn.metrics import confusion_metrix
import seaborn as sb
sb.set()
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib','inline')
import matplotlib.pyplot as pt
confusionmt=confusion_matrix(Y_test,predicition_y)
sb.heatmap(confusionmt.T,square=True,annot=True,fmt='d',cbar=False)

pt.xlabel('true class axis')

pt.ylabel('predicted class axis')
pt.xlabel('true class axis')

pt.ylabel('predicted class axis')