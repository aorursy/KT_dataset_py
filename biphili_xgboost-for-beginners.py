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
import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore') 
dataset=pd.read_csv('../input/ann-churn-modelling/Churn_Modelling.csv')

dataset.head()
X=dataset.iloc[:,3:13].values

#X

y=dataset.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X_1=LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) #Encoding the values of column Country

labelencoder_X_2=LabelEncoder()

X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])

X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

#X
from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#X_train
from xgboost import XGBClassifier

classifier=XGBClassifier()

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix 

cm=confusion_matrix(y_test,y_pred)

cm
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)

print(accuracies.mean())

print(accuracies.std())