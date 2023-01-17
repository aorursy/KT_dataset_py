# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import Imputer
#from sklear import impute.SimpleImputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
train_file_path = '../input/train.csv'
test_file_path = '../input/test.csv'


train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
print(test_data)

train_y = train_data.Survived
titanic_predictor = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_x = train_data[titanic_predictor]
train_x = pd.get_dummies(train_x)
test_x = test_data[titanic_predictor]
test_x = pd.get_dummies(test_x)
output = test_data.PassengerId

##################################################################################
# make copy to avoid changing original data (when Imputing)
new_train_data = train_x.copy()
new_test_data = test_x.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_train_data.columns 
                                 if new_train_data[col].isnull().any())
for col in cols_with_missing:
    new_train_data[col + '_was_missing'] = new_train_data[col].isnull()
    new_test_data[col + '_was_missing'] = new_test_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_train_data = my_imputer.fit_transform(new_train_data)
new_test_data = my_imputer.fit_transform(new_test_data)


######################################################################################

titanic_model = RandomForestClassifier()
titanic_model.fit(new_train_data,train_y)
prediction = titanic_model.predict(new_test_data)
prediction = pd.DataFrame(data=prediction,columns=['Survived'])

#print(prediction)

final_output = pd.concat([output,prediction],axis=1)
final_output.to_csv('output.csv',index=False)
# Any results you write to the current directory are saved as output.


