# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn import base



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the data

df_train=pd.read_csv('../input/cat-in-the-dat/train.csv', index_col='id')

df_test=pd.read_csv('../input/cat-in-the-dat/test.csv', index_col='id')



# Remove rows with missing target, separate target from predictors

y = df_train.target

df_train.drop(['target'], axis=1, inplace=True)

df_train.shape



# Remove rows with missing target, separate target from predictors

df_test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.svm import LinearSVC

import xgboost as xgb



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    global model

    #model = LinearSVC(random_state=0, tol=1e-5)

    #RandomForestRegressor(n_estimators=100, random_state=0)

    model = xgb.XGBClassifier(

        learning_rate =0.1,

        n_estimators=1000,

        max_depth=5,

        min_child_weight=1,

        gamma=0,

        subsample=0.8,

        colsample_bytree=0.8,

        objective= 'binary:logistic')

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



def logistic(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

    lr=LogisticRegression()

    lr.fit(X_train,y_train)

    y_pre=lr.predict(X_test)

    print('Accuracy : ',accuracy_score(y_test,y_pre))
print('train data set has got {} rows and {} columns'.format(df_train.shape[0],df_train.shape[1]))

print('test data set has got {} rows and {} columns'.format(df_test.shape[0],df_test.shape[1]))
df_train.info()
# All categorical columns

object_cols = [col for col in df_train.columns if df_train[col].dtype == "object"]

print(object_cols)
df_train[object_cols].head(10)
ordinal_col=['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

nominal_col=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)
# Transforming ordinal Features

df_train.ord_1 = df_train.ord_1.astype(ord_1)

df_train.ord_2 = df_train.ord_2.astype(ord_2)

df_train.ord_3 = df_train.ord_3.astype(ord_3)

df_train.ord_4 = df_train.ord_4.astype(ord_4)



# test dataset

df_test.ord_1 = df_test.ord_1.astype(ord_1)

df_test.ord_2 = df_test.ord_2.astype(ord_2)

df_test.ord_3 = df_test.ord_3.astype(ord_3)

df_test.ord_4 = df_test.ord_4.astype(ord_4)

df_train.ord_2.head()
# Geting the codes of ordinal categoy's - train

df_train.ord_1 = df_train.ord_1.cat.codes

df_train.ord_2 = df_train.ord_2.cat.codes

df_train.ord_3 = df_train.ord_3.cat.codes

df_train.ord_4 = df_train.ord_4.cat.codes



# Geting the codes of ordinal categoy's - test

df_test.ord_1 = df_test.ord_1.cat.codes

df_test.ord_2 = df_test.ord_2.cat.codes

df_test.ord_3 = df_test.ord_3.cat.codes

df_test.ord_4 = df_test.ord_4.cat.codes
df_train[['ord_0', 'ord_1', 'ord_2', 'ord_3']].head()
from sklearn.preprocessing import LabelEncoder



# Apply label encoder 

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in ['ord_5']:

    df_train[col] = label_encoder.fit_transform(df_train[col])

    df_test[col] = label_encoder.transform(df_test[col])
high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

#df_train.drop(high_card_feats, axis=1, inplace=True)

#df_test.drop(high_card_feats, axis=1, inplace=True)

from category_encoders import  LeaveOneOutEncoder
leaveOneOut_encoder = LeaveOneOutEncoder()

for col in high_card_feats:

    df_train[col] = leaveOneOut_encoder.fit_transform(df_train[col], y)

    df_test[col] = leaveOneOut_encoder.transform(df_test[col])
df_test.head()
# dictionary to map the feature

bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}

# Maping the category values in our dict

df_train['bin_3'] = df_train['bin_3'].map(bin_dict)

df_train['bin_4'] = df_train['bin_4'].map(bin_dict)

df_test['bin_3'] = df_test['bin_3'].map(bin_dict)

df_test['bin_4'] = df_test['bin_4'].map(bin_dict)
df_train.head(10)
low_card_feats = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

df_train[low_card_feats].head(10)
from sklearn.preprocessing import OneHotEncoder
%%time 



#---------------------------------------------------

# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_df_train = pd.DataFrame(OH_encoder.fit_transform(df_train[low_card_feats]))

OH_df_test = pd.DataFrame(OH_encoder.transform(df_test[low_card_feats]))



# One-hot encoding removed index; put it back

OH_df_train.index = df_train.index

OH_df_test.index = df_test.index



# Remove categorical columns (will replace with one-hot encoding)

num_df_train = df_train.drop(low_card_feats, axis=1)

num_df_test = df_test.drop(low_card_feats, axis=1)



# Add one-hot encoded columns to numerical features

OH_df_train_fin = pd.concat([num_df_train, OH_df_train], axis=1)

OH_df_test_fin = pd.concat([num_df_test, OH_df_test], axis=1)

#---------------------------------------------------
OH_df_train_fin.head()
def cyclic_columns_encode(columns, df):

    for col in columns:

        df[col+'_sin']=np.sin((2*np.pi*df[col])/max(df[col]))

        df[col+'_cos']=np.cos((2*np.pi*df[col])/max(df[col]))

    df=df.drop(columns,axis=1)

    return df



columns=['day','month']



X_train=cyclic_columns_encode(columns, OH_df_train_fin)

X_test=cyclic_columns_encode(columns, OH_df_test_fin)

print(X_train.info())

print(X_test.info())
logistic(X_train,y)
clf=LogisticRegression()  # MODEL

clf.fit(X_train, y)

pred=clf.predict_proba(X_test)[:,1]

pd.DataFrame({"id": X_test.index, "target": pred}).to_csv("submission.csv", index=False)