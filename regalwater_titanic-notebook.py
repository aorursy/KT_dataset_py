# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.describe())
print("_"*40)
print(train_df.head())
print("_"*40)
print(train_df.columns)
print("_"*40)
missing_value_count_by_column_train_df = train_df.isnull().sum()

print(missing_value_count_by_column_train_df)
missing_cabins = train_df["Cabin"].isnull()
train_df["Cabin"][missing_cabins] = "Z0"
train_df["CabinLetter"] = train_df["Cabin"].astype(str).str[0]


def count_cabins(S): 
    stringstyle = '\D\d+'
    stringstyler = re.compile(stringstyle)
    stringsfound = stringstyler.findall(S)    
    return len(stringsfound)


train_df["NumberCabins"] = train_df["Cabin"].apply(count_cabins)
cabinzeros = train_df["NumberCabins"] == 0
train_df["NumberCabins"][cabinzeros] = 1

train_df["PricePerCabin"] = train_df["Fare"]/train_df["NumberCabins"]
train_df
#count_cabins(train_df["Cabin"][5])
#count_cabins(train_df["Cabin"].astype(str))
test_df["Fare"] = test_df.Fare.fillna(test_df.Fare.mean())


missing_cabins = test_df["Cabin"].isnull()
test_df["Cabin"][missing_cabins] = "Z0"
test_df["CabinLetter"] = test_df["Cabin"].astype(str).str[0]


# def count_cabins(S): 
#     stringstyle = '\D\d+'
#     stringstyler = re.compile(stringstyle)
#     stringsfound = stringstyler.findall(S)    
#     return len(stringsfound)


test_df["NumberCabins"] = test_df["Cabin"].apply(count_cabins)
cabinzeros = test_df["NumberCabins"] == 0
test_df["NumberCabins"][cabinzeros] = 1

test_df["PricePerCabin"] = test_df["Fare"]/test_df["NumberCabins"]
test_df
#count_cabins(train_df["Cabin"][5])
#count_cabins(train_df["Cabin"].astype(str))
train_df["AgeImputed"] = train_df.Age.fillna(train_df.Age.mean())

train_df.head()
test_df["AgeImputed"] = test_df.Age.fillna(test_df.Age.mean())

#train_df.head()
# from sklearn.impute import SimpleImputer

# my_imputer = SimpleImputer()

# train_df_imputed = my_imputer.fit_transform(train_df.as_matrix)

#print(train_df_imputed)
clean_train = train_df.columns.values.tolist()
clean_train.remove('Name')
clean_train.remove('Ticket')
clean_train.remove('Cabin')
clean_train.remove('Age')

clean_train_df = train_df[clean_train]

one_hot_encoded_train_df = pd.get_dummies(clean_train_df)
one_hot_encoded_train_df.head()
clean_test = test_df.columns.values.tolist()
clean_test.remove('Name')
clean_test.remove('Ticket')
clean_test.remove('Cabin')
clean_test.remove('Age')

clean_test_df = test_df[clean_test]

one_hot_encoded_test_df = pd.get_dummies(clean_test_df)
one_hot_encoded_test_df['CabinLetter_T'] = 0
one_hot_encoded_test_df.head()

#one_hot_encoded_train_df.PricePerCabin.max()
feature_names = one_hot_encoded_train_df.columns.values.tolist()
feature_names.remove("Survived")

print(feature_names)

train_y = one_hot_encoded_train_df.Survived
train_X = one_hot_encoded_train_df[feature_names]

feature_names_test = one_hot_encoded_test_df.columns.values.tolist()

test_X = one_hot_encoded_test_df[feature_names_test]
print(test_X.columns)

missing_value_count_by_column = one_hot_encoded_train_df.isnull().sum()

print(missing_value_count_by_column)


missing_value_count_by_column_test = one_hot_encoded_test_df.isnull().sum()

print(missing_value_count_by_column_test)
print(train_X.columns)
print(len(train_X.columns))
print(test_X.columns)
print(len(test_X.columns))
print(type(test_X))
print(test_X.shape)
titanic_model = RandomForestRegressor(random_state = 0)
titanic_model.fit(train_X,train_y)
probs = titanic_model.predict(test_X)
survivallist = np.rint(probs)
print(survivallist)
submission = pd.DataFrame()
submission["PassengerId"] = test_X["PassengerId"]
submission["Survived"] = survivallist.astype(int)

print(submission.head())

submission.to_csv('submission.csv', index=False)
