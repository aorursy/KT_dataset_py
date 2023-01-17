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
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.describe()

train_data.head(10)
train_data.columns
train_data.shape
#Data processing for train data



missing_val_count_by_column = (train_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#dropping the column Cabin as it is more than 70 % missing in both test and train

train_data=train_data.drop(["Cabin"],axis=1)

train_data.shape
#dealing with Age and embarked columns for missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')



imputer=imputer.fit(train_data[['Embarked']])

train_data[['Embarked']]=imputer.transform(train_data[['Embarked']])
#dealing with Age and embarked columns for missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')



imputer=imputer.fit(train_data[['Age']])

train_data[['Age']]=imputer.transform(train_data[['Age']])
#as name and ticket are going to different for everyone there is no benefit in keeping them

train_data=train_data.drop(['Name','Ticket'],axis=1)
#filtering the categorical columns from the data set

categorical_feature_mask = train_data.dtypes==object
#transfering categorical columns into a list

categorical_cols = train_data.columns[categorical_feature_mask].tolist()

train_data[categorical_cols].head(10)
#Encoding the categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_train=LabelEncoder()

train_data[categorical_cols] = train_data[categorical_cols].apply(lambda col: labelencoder_train.fit_transform(col))
train_data.head(10)
X=train_data.iloc[:,[0,2,3,4,5,6,7,8]]

y=train_data.iloc[:,1]
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X,y)

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.describe()
test_data.columns
test_data.shape
missing_val_count_by_column = (test_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#dropping the column Cabin as it is more than 70 % missing in both test and train

test_data=test_data.drop(["Cabin"],axis=1)

#dealing with Age and embarked columns for missing data

from sklearn.impute import SimpleImputer

imputer_t = SimpleImputer(missing_values = np.nan, strategy = 'mean')



imputer_t=imputer_t.fit(test_data[['Age','Fare']])

test_data[['Age','Fare']]=imputer_t.transform(test_data[['Age','Fare']])
#as name and ticket are going to different for everyone there is no benefit in keeping them

test_data=test_data.drop(['Name','Ticket'],axis=1)
#filtering the categorical columns from the data set

categorical_feature_mask = test_data.dtypes==object
#transfering categorical columns into a list

categorical_cols = test_data.columns[categorical_feature_mask].tolist()

test_data[categorical_cols].head(10)
#Encoding the categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_test=LabelEncoder()

test_data[categorical_cols] = test_data[categorical_cols].apply(lambda col: labelencoder_test.fit_transform(col))
#predicting new result



train_predictions=regressor.predict(X)
test_predictions=regressor.predict(test_data)
output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':test_predictions})
output.to_csv('submission_1.csv',index=False)