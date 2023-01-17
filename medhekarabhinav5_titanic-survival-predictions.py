import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import cross_val_score

import warnings

import re

import os



for dirname, _, filenames in os.walk('/kaggle/input'): # To list files

    for filename in filenames:

        print(os.path.join(dirname, filename))

# let's Name filesa as train_data and test_data.

# train_data consists of training data and test_data consists data from which we are going to predict survival of passengers.

# files are in CDV format, we will use read_csv()

warnings.filterwarnings("ignore")



train_data = pd.read_csv('/kaggle/input/titanic/train.csv',index_col = 'PassengerId') # with adding passnger is as index

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col = 'PassengerId')# with adding passnger is as index
train_data.columns
train_data
test_data
train_data['head_count'] = train_data['SibSp'] + train_data['Parch']

train_data
test_data['head_count'] = test_data['SibSp'] + test_data['Parch']

test_data
def Cabin_class_available_data(n,t_data):

    train_cabclass = t_data['Pclass'] == n

    train_cabclass = t_data[train_cabclass]

    train_cabclass_available_data = train_cabclass['Cabin'].notnull()

    train_cabclass_available_data = train_cabclass[train_cabclass_available_data]

    print(train_cabclass_available_data)

    print(train_cabclass['Cabin'].unique())
def Cabin_Class_Change(n, t_data):

    train_cabclass = t_data['Pclass'] == n

    train_cabclass = t_data[train_cabclass]

    train_cabclass['Cabin'] = 1

    t_data['Cabin_Class_' + str(n)] = train_cabclass['Cabin']

    print(t_data)
Cabin_class_available_data(3, train_data)
# If observed Pclass 3 data is  in between E,F,G



Cabin_Class_Change(3, train_data)
# now Check availbale data with Pclass 2 related to Cabin

Cabin_class_available_data(2, train_data)
# If observed Pclass 2 data is  in between D,E,F



Cabin_Class_Change(2, train_data)
# same check available data with cabin class 1

Cabin_class_available_data(1, train_data)
# here Pclass one has A, B,C,D

Cabin_Class_Change(1, train_data)
#changing same for test_data

Cabin_class_available_data(3, test_data)

Cabin_class_available_data(2, test_data)

Cabin_class_available_data(1, test_data)

Cabin_Class_Change(3, test_data)

Cabin_Class_Change(2, test_data)

Cabin_Class_Change(1, test_data)
object_cols = [col for col in train_data.columns if train_data[col].dtype == 'object']

numerical_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]

print(numerical_cols)

print(object_cols)
column_with_missing_values = train_data.isnull().sum()

column_with_missing_values[column_with_missing_values > 0]
y = pd.DataFrame(train_data['Survived'])

X = train_data.drop(['Survived'], axis = 1)

X_test = test_data.copy()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)
print(X_train.count())

X_train
print(X_valid.count())
print(y_train.count())
print(y_valid.count())
X_test.count()
# Cleaning Categorical varibles.

# Categorical variables are which are having missing values are as follow:

# Cabin and Embarked in main DataFrame.

# Observed that from above count Embarked is missing frm X_train only.

# Cabin is missing from both X_train and X_valid



# Let's check Good and bad columns from X_train and X_valid



object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

print(object_cols)



# Good Columns those have same set of data in both train and valid

good_columns = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

print(good_columns)



bad_columns = list(set(object_cols) - set(good_columns))

print(bad_columns)
# Checking Cardinality of X_train and X_valid

print(X_train[object_cols].nunique())

X_train['Embarked'].unique()
print(X_valid[object_cols].nunique())

X_valid['Embarked'].unique()
good_columns.append('Embarked')

bad_columns.remove('Embarked')

bad_columns.append('SibSp')

bad_columns.append('Parch')

print(good_columns)

print(bad_columns)
# removing bad columns from sets

label_X_train = X_train.drop(bad_columns, axis=1)

label_X_valid = X_valid.drop(bad_columns, axis=1)

label_X_test = X_test.drop(bad_columns, axis=1)



print(label_X_train)



# Sex column has only 2 cardinality, it is easy to use label encoder for it. Male = 1 and Female = 0

myLabelEncoder = LabelEncoder()

label_X_train['Sex'] = myLabelEncoder.fit_transform(label_X_train['Sex'])

label_X_valid['Sex'] = myLabelEncoder.transform(label_X_valid['Sex'])

label_X_test['Sex'] = myLabelEncoder.transform(label_X_test['Sex'])



#Cleaning outEmbarked with One Hot encoding

label_X_train["Embarked"].fillna("S", inplace=True)

# there are no missing values in test data for embarked column

# instead it has a value missing in fare 

label_X_test['Fare'].fillna(label_X_test['Fare'].mean(), inplace=True)







# We will fill Cabin_class_1,Cabin_class_2 and Cabin_class_3 here.Simple imputer we are going to use will replace value with (mean/median) which is not good.

label_X_train["Cabin_Class_1"].fillna(0, inplace=True)

label_X_valid["Cabin_Class_1"].fillna(0, inplace=True)

label_X_test["Cabin_Class_1"].fillna(0, inplace=True)



label_X_train["Cabin_Class_2"].fillna(0, inplace=True)

label_X_valid["Cabin_Class_2"].fillna(0, inplace=True)

label_X_test["Cabin_Class_2"].fillna(0, inplace=True)



label_X_train["Cabin_Class_3"].fillna(0, inplace=True)

label_X_valid["Cabin_Class_3"].fillna(0, inplace=True)

label_X_test["Cabin_Class_3"].fillna(0, inplace=True)
label_X_train
OH_X_train = pd.get_dummies(label_X_train, columns=["Embarked"])

OH_X_valid = pd.get_dummies(label_X_valid, columns=["Embarked"])

OH_X_test = pd.get_dummies(label_X_test, columns=['Embarked'])

print(OH_X_train)

print(OH_X_valid)

print(OH_X_test)
OH_X_train.isnull().sum()
OH_X_test.isnull().sum()
OH_X_valid.isnull().sum()
OH_X_valid.index
# there are 3 columns which are having missing value, all are numerical. Let's first clear them out

# For numerical values rather than dropping column, i will use SimpleImputer
mySimpleImputer = SimpleImputer() # for now we will keep it as default(mean)

final_X_train_imputed = pd.DataFrame(mySimpleImputer.fit_transform(OH_X_train))

final_X_valid_imputed = pd.DataFrame(mySimpleImputer.transform(OH_X_valid))

final_X_test_imputed = pd.DataFrame(mySimpleImputer.transform(OH_X_test))
final_X_train_imputed.columns = OH_X_train.columns

final_X_valid_imputed.columns = OH_X_valid.columns

final_X_test_imputed.columns = OH_X_test.columns

final_X_train_imputed.index = OH_X_train.index

final_X_valid_imputed.index = OH_X_valid.index

final_X_test_imputed.index = OH_X_test.index
print(final_X_train_imputed.head())

print(final_X_valid_imputed.head())

print(final_X_test_imputed.head())
X_train
train_names = X_train['Name']

valid_names = X_valid['Name']

test_names = X_test['Name']

result_train = train_names.str.extract(pat = "(Mr|Mrs|Miss|Rev|Master|Don)")

result_valid = valid_names.str.extract(pat = "(Mr|Mrs|Miss|Rev|Master|Don)")

result_test = test_names.str.extract(pat = "(Mr|Mrs|Miss|Rev|Master|Don)")

final_X_train_imputed['titles'] = result_train[0]

final_X_valid_imputed['titles'] = result_valid[0]

final_X_test_imputed['titles'] = result_test[0]
final_X_valid_imputed.loc[496]
print(final_X_train_imputed.isnull().sum())
def title_selector_fill(X):

    for index in X.index:

        Age = X['Age'].loc[index]

        Sex = X['Sex'].loc[index]

        title = X['titles'].loc[index]

        

        if title == 'Rev' or title == 'Don':

            X['titles'].loc[index] = 'Mr'

        

        if (title == '') and (Age < 18 and Sex == 1):

            X['titles'].loc[index] = 'Master'

        elif Age >= 18 and Sex == 1:

            X['titles'].loc[index] = 'Mr'

        elif Age < 18 and Sex == 0:

            X['titles'].loc[index] = 'Miss'

        elif Age >= 18 and Sex == 0:

            X['titles'].loc[index] = 'Mrs'

            

    print(X.isnull().sum())
# let's try to fill them up



# Train Data

test_set = pd.DataFrame(final_X_train_imputed['titles'])

title_age_sex = pd.concat([test_set, final_X_train_imputed[['Age','Sex']]],axis = 1)

before_imp = final_X_train_imputed[title_age_sex['titles'].isnull()]



title_selector_fill(final_X_train_imputed)

final_X_train_imputed['titles'].unique()
# Valid Data

test_set = pd.DataFrame(final_X_valid_imputed['titles'])

title_age_sex = pd.concat([test_set, final_X_valid_imputed[['Age','Sex']]],axis = 1)

before_imp = final_X_valid_imputed[title_age_sex['titles'].isnull()]



title_selector_fill(final_X_valid_imputed)

final_X_valid_imputed['titles'].unique()
#test data

test_set = pd.DataFrame(final_X_test_imputed['titles'])

title_age_sex = pd.concat([test_set, final_X_test_imputed[['Age','Sex']]],axis = 1)

before_imp = final_X_test_imputed[title_age_sex['titles'].isnull()]



title_selector_fill(final_X_test_imputed)

final_X_test_imputed['titles'].unique()
final_X_train_imputed['titles'].unique()
final_X_valid_imputed
final_X_test_imputed
final_X_train_imputed['titles'] = myLabelEncoder.fit_transform(final_X_train_imputed['titles'])

final_X_valid_imputed['titles'] = myLabelEncoder.transform(final_X_valid_imputed['titles'])

final_X_test_imputed['titles'] = myLabelEncoder.transform(final_X_test_imputed['titles'])
final_X = pd.concat([final_X_train_imputed, final_X_valid_imputed], axis=0)
final_X
final_y = pd.concat([y_train, y_valid])
final_y
final_X_test = final_X_test_imputed.copy()
final_X_test.columns
final_X.columns
def get_cross_val(n):

    #model = DecisionTreeClassifier(random_state = 1)

    #model = RandomForestClassifier(n_estimators = n, n_jobs = -1,  random_state = 1)

    model = XGBClassifier(n_estimators = n, learning_rate = 0.125, random_state = 0)

    model.fit(final_X_train_imputed, y_train)

    val_preds = model.predict(final_X_valid_imputed) 

    scores = -1 * cross_val_score(model, final_X_train_imputed, y_train , cv = 10, scoring = 'neg_mean_absolute_error')

    return scores.mean()
n_esti = [50,80,100,120,130,140,150,170,180,200,250,300,350,400,450,500,750,1000,1500,2000,2500,3000,3500]

#for n in n_esti:

    #print(str(n) + " : " + str(get_cross_val(n)))
def get_mae(predicted_vals,actual_vals):

    mae = mean_absolute_error(predicted_vals, actual_vals)

    return mae
#model = RandomForestClassifier(n_estimators = 120, random_state = 1)

model = XGBClassifier(n_estimators = 50, learning_rate = 0.125, random_state = 1)

model_fit = model.fit(final_X_train_imputed, y_train)

print(model_fit)

val_preds = model.predict(final_X_valid_imputed) 
mae_score = get_mae(val_preds, y_valid)

print(mae_score)

accuracy_score = accuracy_score(y_valid, val_preds)

print(accuracy_score)
model_xgbc = XGBClassifier(n_estimators = 120, learning_rate = 0.12395, random_state = 1)

model_xgbc.fit(final_X, final_y)

test_data_predictions = model_xgbc.predict(final_X_test)
final_X_test.reset_index(inplace = True)

final_X_test
output = pd.DataFrame({'PassengerId' : final_X_test.PassengerId, 'Survived' : test_data_predictions})

output.to_csv('my_submission.csv', index=False)
print(test_data_predictions)
#print((output['Survived'] == 0).sum())

#print((output['Survived'] == 1).sum())