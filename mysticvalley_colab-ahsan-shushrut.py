#importing modules

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from statistics import mean

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np

import warnings
#creating dataframe

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_test
#wrangle data

train_X = df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']] #columns selected to train X

test_X = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']] #columns selected to Test X

train_y = df_train[['Survived']] #To predict survival



#for train_X

train_X = pd.get_dummies(train_X) #one hot encoding for conversion of categorical data to numeric

train_X = train_X.drop(['Sex_male'], axis=1) #sex_male is useless

train_X.columns = train_X.columns.str.replace('Sex_female', 'Sex')#setting sex_female as sex



#for test_X

test_X = pd.get_dummies(test_X) #one hot encoding for conversion of categorical data to numeric

test_X = test_X.drop(['Sex_male'], axis=1) #sex_male is useless

test_X.columns = test_X.columns.str.replace('Sex_female', 'Sex')#setting sex_female as sex



test_X
#Imputing for missing values

imputer = SimpleImputer()



#for train_X

train_X = imputer.fit_transform(train_X)

train_X = pd.DataFrame(train_X)



#for test_X

test_X = imputer.fit_transform(test_X)

test_X = pd.DataFrame(test_X)
##Using XGBoost, for running multiple times...

warnings.filterwarnings("ignore")#ignoring warnings

    

#classification & pred

clf_xgb = XGBClassifier(learning_rate=0.05, n_jobs=8, max_depth=7)

clf_xgb.fit(train_X, train_y)

pred = clf_xgb.predict(test_X)



#Create csv file to upload to Kaggle

submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred})

submission.head()



#Convert dataframe to csv file 

filename = 'titanix_pred.csv'

submission.to_csv(filename, index = False)
# #Using RandomForest, for running multiple times...

# lis_acc = []

# for i in range(50):

#     #spliting in train and test data

#     train_x, test_x, train_y, test_y = train_test_split(X_imputed, y_orginal, test_size = 20)

    

#     warnings.filterwarnings("ignore")#ignoring warnings

    

#     #classification & pred

#     clf_rfc = RandomForestClassifier()

#     clf_rfc.fit(train_x, train_y)

#     pred = clf_rfc.predict(test_x)

    

#     #checking accuracy

#     lis_acc.append(accuracy_score(test_y, pred))

# print(mean(lis_acc)*100, "%")