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
# Logistic Regression



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

x = dataset.iloc[:, [2, 4,5,6,7]].values

y = dataset.iloc[:, 1].values





#taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x[:, 2:3])

x[:, 2:3] = imputer.transform(x[:, 2:3])



#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()

x[:, 1] = labelencoder_x.fit_transform(x[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])

x = onehotencoder.fit_transform(x).toarray()

'''labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)'''





# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.20, random_state = 0)





# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_valid = sc.transform(x_valid)



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(x_valid)



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train,y_train)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_pred)



#############predicting the given problem############



#importing the dataset



test_set = pd.read_csv('/kaggle/input/titanic/test.csv')

x1 = test_set.iloc[:, [1, 3,4,5,6]].values



#taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x1[:, 2:3])

x1[:, 2:3] = imputer.transform(x1[:, 2:3])



#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x1 = LabelEncoder()

x1[:, 1] = labelencoder_x1.fit_transform(x1[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])

x1 = onehotencoder.fit_transform(x1).toarray()



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x1 = sc.fit_transform(x1)



# making predictions

y_test = xgb.predict(x1)



final_result = pd.DataFrame(test_set.iloc[:, [0]].values)

final_result['survival'] = y_test

final_result.columns = ['PassengerId','Survived']

final_result.to_csv('submission2020.csv', index= False)
