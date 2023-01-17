root = "../input/"

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", root]).decode("utf8"))
train_data = pd.read_csv(root + 'train.csv')

pre_y_test = pd.read_csv(root + 'gendermodel.csv')['Survived']

pre_x_test = pd.read_csv(root + 'test.csv')

pre_x_test = pre_x_test[pre_x_test.columns.difference(['Ticket','PassengerId','Name','Cabin'])]

pre_x_train = train_data[train_data.columns.difference(['Survived','Ticket','PassengerId','Name','Cabin'])]

pre_y_train = train_data['Survived']
from sklearn.preprocessing import Imputer

imputer = Imputer()

pre_x_train[['Age']] = imputer.fit_transform(pre_x_train[['Age']])

pre_x_test[['Age']] = imputer.transform(pre_x_test[['Age']])

pre_x_test[['Fare']] = imputer.fit_transform(pre_x_test[['Fare']])
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

labelEncoder = LabelEncoder()

pre_x_train.iloc[:,1] = labelEncoder.fit_transform(pre_x_train.iloc[:,1].astype(str))

pre_x_test.iloc[:,1] = labelEncoder.transform(pre_x_test.iloc[:,1].astype(str))

pre_x_train.iloc[:,5] = labelEncoder.fit_transform(pre_x_train.iloc[:,5])

pre_x_test.iloc[:,5] = labelEncoder.transform(pre_x_test.iloc[:,5])

oneHotEncoder = OneHotEncoder(categorical_features=[1,5],sparse=False)

pre_x_train = oneHotEncoder.fit_transform(pre_x_train)

pre_x_test = oneHotEncoder.fit_transform(pre_x_test)