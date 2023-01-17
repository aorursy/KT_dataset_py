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
test=pd.read_csv("/kaggle/input/test.csv")

train=pd.read_csv("/kaggle/input/train.csv")

train=train.drop([ 'Ticket', 'Cabin', 'Embarked'], axis=1)

train.head()
y=train.Survived

X=train[['Pclass']]

# from sklearn.ensemble import RandomForestRegressor

# model=RandomForestRegressor()

# model.fit(X,y)

# from sklearn.metrics import mean_absolute_error

# pred=model.predict(X)

# mean_absolute_error(pred,y)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.regplot(x="Pclass",y="Survived",data=train)

plt.ylim(0)
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=0)

# model=RandomForestRegressor()

# model.fit(train_X,train_y)

# from sklearn.metrics import mean_absolute_error

# pred=model.predict(val_X)

# mean_absolute_error(pred,val_y)

#label_X_train[],label_X_valid[]

s=(train.dtypes=="object")

object_cols=list(s[s].index)

print(object_cols)

label_X_train=train.copy

label_X_valid=test.copy

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in ["Sex"]:

    lable_X_train[col]=le.fit_transform(train[col])

    lable_X_valid[col]=le.transform(test[col])

label_X_train  

    
from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('../input/train.csv') 

X_test = pd.read_csv('../input/test.csv')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['Survived'], inplace=True)

y = X.Survived

X.drop(['Survived'], axis=1, inplace=True)



# To keep things simple, we'll drop columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 

X.drop(cols_with_missing, axis=1, inplace=True)

X_test.drop(cols_with_missing, axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)




# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

#label_X_train = X_train.copy

#label_X_valid = X_valid.copy

label_encoder=LabelEncoder()

for col in set(good_label_cols):

    label_X_train[col]=label_encoder.fit_transform(X_train[col])

    label_X_valid[col]=label_encoder.transform(X_valid[col])

print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))    
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder





OH_encoder=OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train=pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid=pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))



OH_cols_train.index=X_train.index

OH_cols_valid.index=X_valid.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

OH_cols_valid.head()
#final_X_test = pd.DataFrame(final_imputer.transform(X_test))



# Get test predictions

# preds_test = model.predict(final_X_test)

# step_4.b.check()



# output = pd.DataFrame({'PassengerId': X_test.index,

#                        'Survived': preds})

train.to_csv('submission.csv', index=False)