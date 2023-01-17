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
pd.set_option('display.max_rows', None)

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_df.head()
null_vals = pd.DataFrame(round ( 100 * (train_df.isnull().sum() / len(train_df.index)), 3))

null_vals.columns = ['% null values']

null_vals
#Let me check the null values for embarked column

train_df.loc[train_df['Embarked'].isnull()]
#Let me impute the values with mode values for the embarked

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)  

test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)  

train_df.loc[train_df['Embarked'].isnull()]
print(train_df['Cabin'].mode()[0])

print(test_df['Cabin'].mode()[0])
#Let me check the unique cabins

train_df['Cabin'].unique()
train_df['Cabin'].fillna('N0', inplace=True)

test_df['Cabin'].fillna('N0', inplace=True)

print('Train DF::', train_df['Cabin'].unique())

print('Test DF::', test_df['Cabin'].unique())
train_df['Age'].fillna(train_df['Age'].mode()[0], inplace=True)  

test_df['Age'].fillna(test_df['Age'].mode()[0], inplace=True)  
print('Train::', train_df['Age'].unique())



print('Test::',train_df['Age'].unique())
null_vals = pd.DataFrame(round ( 100 * (train_df.isnull().sum() / len(train_df.index)), 3))

null_vals.columns = ['% null values']

null_vals
null_vals = pd.DataFrame(round ( 100 * (test_df.isnull().sum() / len(test_df.index)), 3))

null_vals.columns = ['% null values']

null_vals
test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace=True)  

null_vals = pd.DataFrame(round ( 100 * (test_df.isnull().sum() / len(test_df.index)), 3))

null_vals.columns = ['% null values']

null_vals
train_df.info()
test_df.info()
print('Cabin::', train_df['Cabin'].unique())

print('Embarked::', train_df['Embarked'].unique())

print('Sex::', train_df['Sex'].unique())
test_passengerid = test_df['PassengerId']

test_df.drop(['Name', 'Ticket', 'PassengerId'], axis = 1, inplace=True)

train_df.drop(['Name', 'Ticket', 'PassengerId'], axis = 1, inplace=True)
train_df['Cabin_Type'] = train_df['Cabin'].astype(str).str[0]

test_df['Cabin_Type'] = test_df['Cabin'].astype(str).str[0]

train_df['Cabin_Seq'] = train_df['Cabin'].astype(str).str[1:]

test_df['Cabin_Seq'] = test_df['Cabin'].astype(str).str[1:]

train_df.head()
train_df.info()
train_df['Cabin_Seq'] = pd.to_numeric(train_df['Cabin_Seq'], errors='coerce')

test_df['Cabin_Seq'] = pd.to_numeric(test_df['Cabin_Seq'], errors='coerce')

train_df.info()
train_df['Cabin_Type'] = train_df['Cabin_Type'].str.upper()

test_df['Cabin_Type'] = test_df['Cabin_Type'].str.upper()

print(train_df['Cabin_Type'].unique())

print(test_df['Cabin_Type'].unique())

train_cabin_type_info = train_df [train_df['Cabin_Type'] == 'T']

print(train_cabin_type_info)
#We have only one record with cabin type T. This cabin type is not available in the test data so I will remove this outlier row from test data. 

train_df = train_df [train_df['Cabin'] != 'T']
#Now let me drop the cabin column

train_df.drop('Cabin', axis=1, inplace=True)

test_df.drop('Cabin', axis=1, inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns

corr = train_df.corr()

plt.figure(figsize=(10, 10))



sns.heatmap(corr, cmap="YlGnBu", annot=True)





plt.show()
import seaborn as sns

plt.figure(figsize=(20, 15))

sns.pairplot(train_df)

plt.show()
train_df.head()
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

label_encoder = label_encoder.fit(train_df['Sex'])

encoded_sex = label_encoder.transform(train_df['Sex'])

train_df['Sex'] = encoded_sex

test_df['Sex'] = label_encoder.transform(test_df['Sex'])





label_encoder = label_encoder.fit(train_df['Embarked'])

encoded_embarked = label_encoder.transform(train_df['Embarked'])

train_df['Embarked'] = encoded_embarked

test_df['Embarked'] = label_encoder.transform(test_df['Embarked'])



label_encoder = label_encoder.fit(train_df['Cabin_Type'])

encoded_embarked = label_encoder.transform(train_df['Cabin_Type'])

train_df['Cabin_Type'] = encoded_embarked

test_df['Cabin_Type'] = label_encoder.transform(test_df['Cabin_Type'])



#label_encoder = label_encoder.fit(train_df['Cabin_Seq'])

#encoded_embarked = label_encoder.transform(train_df['Cabin_Seq'])

#train_df['Cabin_Seq'] = encoded_embarked

#test_df['Cabin_Seq'] = label_encoder.transform(test_df['Cabin_Seq'])



print(train_df.Cabin_Type.unique())

print(train_df.Sex.unique())

print(train_df.Embarked.unique())

train_df.head()
test_df.head()
## X & Y axis split



y_train = train_df.pop('Survived')

x_train = train_df
x_train.head()
from xgboost import XGBClassifier



xgb_model = XGBClassifier()

xgb_model.fit(x_train, y_train)



xgb_predictions = xgb_model.predict(test_df)



xgb_test_result_df = pd.DataFrame({'Survived': xgb_predictions})

xgb_results_df = pd.concat([test_passengerid , xgb_test_result_df], axis = 1)



xgb_results_df.to_csv('titanic_xgb_baseline.csv', index = False)
xgb1 = XGBClassifier(learning_rate=0.001, n_estimators=1000, objective='binary:logistic', silent=False,  nthread = 5)



xgb1.fit(x_train, y_train)

xgb_predictions = xgb1.predict(test_df)



xgb_test_result_df = pd.DataFrame({'Survived': xgb_predictions})

xgb_results_df = pd.concat([test_passengerid , xgb_test_result_df], axis = 1)



xgb_results_df.to_csv('titanic_xgb1.csv', index = False)
from sklearn.model_selection import GridSearchCV



xgb2 = XGBClassifier(n_estimators=100, objective='binary:logistic', silent=False)



params = {'learning_rate' : [.1, .01, .001],

            'min_child_weight': [1, 3, 5],

            'gamma': [0.5, 1, 2],

            'subsample': [0.5, 0.7, .9],

            'colsample_bytree': [0.6, 0.8, 1.0],

            'max_depth': [3, 4, 5]

        }



gcv = GridSearchCV(xgb2,

                  params,

                  cv = 4,

                  scoring = 'roc_auc',

                  n_jobs = 5,

                  verbose = 20)



gcv.fit(x_train, y_train)

gcv.best_estimator_
gcv_xgb_predictions = gcv.predict(test_df)



gcv_xgb_test_result_df = pd.DataFrame({'Survived': gcv_xgb_predictions})

gcv_xgb_results_df = pd.concat([test_passengerid , gcv_xgb_test_result_df], axis = 1)



xgb_results_df.to_csv('titanic_gcv_xgb_v2.csv', index = False)