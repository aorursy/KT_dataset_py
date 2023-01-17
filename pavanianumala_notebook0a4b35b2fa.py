# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

y_out = pd.read_csv("../input/titanic/gender_submission.csv")

print(test)
train = train.drop(['Name','Ticket','Cabin'], axis = 1)

test = test.drop(['Name','Ticket','Cabin'], axis = 1)





print(train.head())

print(test.head())

x = train.iloc[:,2:].values

y = train.iloc[:,1].values

X_test = test.iloc[:,1:].values

X_test1 = test.iloc[:,0].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent')

imputer.fit(x[:,[2]])

x[:,[2]] = imputer.transform(x[:,[2]])

imputer.fit(x[:,[6]])

x[:,[6]] = imputer.transform(x[:,[6]])

imputer.fit(x[:,[2]])

x[:,[2]] = imputer.transform(x[:,[2]])

imputer.fit(x[:,[5]])

x[:,[5]] = imputer.transform(x[:,[5]])





from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1,-1])] , remainder = "passthrough")

x = ct.fit_transform(x)



X_test = ct.transform(X_test)
from sklearn.model_selection import train_test_split

x_train , x_test ,y_train , y_test = train_test_split(x,y, test_size = 0.2,random_state = 42)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

print(y_pred)

from sklearn.metrics import accuracy_score,confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)

accuracy_score(y_test,y_pred)
print(X_test1)
Y_pred = classifier.predict(X_test)

print(Y_pred)


dict = {'PassengerId': X_test1, 'Survived': Y_pred}  

     

df = pd.DataFrame(dict) 

df.to_csv('titanicCSV.csv',index=False)