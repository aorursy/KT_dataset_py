# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import useful libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

import statsmodels.api as sm

import scipy as sp

from sklearn.metrics import mean_squared_error

from sklearn.externals import joblib

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/Admission_Predict.csv")

df2 = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")  
print(df1.shape)

print(df2.shape)
# find how many rows which instances are in df1 but not

# in df2. Note: longer dataset.isin(shorter dataset)

len(df2[~df2.isin(df1)].dropna())
df_train = df1

df_test = df2.iloc[400:]

df_test.shape
# check columns in train data

df_train.columns
# change some column names to the simple style

df_train = df_train.rename(columns={'Serial No.':'SerialNo','GRE Score':'GREScore',

                         'TOEFL Score':'TOEFLScore','LOR ':'LOR',

                         'University Rating':'UniversityRating',

                         'Chance of Admit ':'ChanceOfAdmit'})
# check train dataset info

df_train.info()
# get statistics of train dataset

df_train.describe()
df_train['GREScore'].hist()
df_train['TOEFLScore'].hist()
df_train['UniversityRating'].hist()
df_train['SOP'].hist()
df_train['LOR'].hist()
df_train['Research'].hist()
df_train['CGPA'].hist()
# Firstly check relationship between TOEFLScore(GREScore has high correlation with

# TOEFLScore, so just check TOEFLScore), Research, SOP(

# LOR has high correlation with SOP) and UniversityRating seperately.

bins = [0, 92, 102, 112, 122]

df_train['TOEFL_binned'] = pd.cut(df_train['TOEFLScore'], bins)

sns.countplot(x='TOEFL_binned', hue='UniversityRating', data=df_train)

plt.show()
sns.countplot(x='SOP', hue='UniversityRating', 

              data=df_train)

plt.show()
sns.countplot(x='Research', hue='UniversityRating', data=df_train)

plt.show()
# Here to check the relationship between SOP, UniversityRating, 

# Research and ChanceofAdmit seperately.

bins = [0.2, 0.4, 0.6, 0.8, 1.0]

df_train['ChanceOfAdmit_binned'] = pd.cut(df_train['ChanceOfAdmit'], bins)
sns.countplot(x='SOP', hue='ChanceOfAdmit_binned', data=df_train)

plt.show()
sns.countplot(x='UniversityRating', hue='ChanceOfAdmit_binned', data=df_train)

plt.show()
sns.countplot(x='Research', hue='ChanceOfAdmit_binned', data=df_train)

plt.show()
# split cols into two parts num_cols and rest_cols

num_cols = ['GREScore', 'TOEFLScore', 'CGPA']

rest_cols = ['UniversityRating', 'SOP','LOR','Research']

num_cols_target = num_cols + ['ChanceOfAdmit']

rest_cols_target = rest_cols + ['ChanceOfAdmit']
sns.pairplot(df_train[num_cols_target])
sns.pairplot(df_train[rest_cols_target])
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df_train[num_cols+rest_cols+['ChanceOfAdmit']].corr(),

            ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
# Before build a linear model, we need to scale the data

def scaling(df, train=True):

    '''use StandardScaler to scale raw data'''

    cols = df.columns

    scaler = StandardScaler()

    if train:

        scaler.fit(df)

        joblib.dump(scaler, "scaler.save")          

    else:

        scaler = joblib.load("scaler.save")

    scaled_df = scaler.transform(df)

    df_X = pd.DataFrame(scaled_df, columns=cols)

    

    return df_X
# sklearn modeling

def model_sklearn(x, y):

    '''Use sklearn to build model'''

    reg = linear_model.LinearRegression()

    reg.fit(x, y)

    joblib.dump(reg, 'model.h5')

    print('Intercept: \n', reg.intercept_)

    print('Coefficients: \n', reg.coef_)
# statsmodels modeling

def model_stats(x, y):

    x = sm.add_constant(x) # adding a constant



    model = sm.OLS(y, x).fit()

    predictions = model.predict(x) 



    print_model = model.summary()

    print(print_model)

    

    return predictions
# Use sklearn to build the model  

X = scaling(df_train[num_cols]) 

Y = df_train['ChanceOfAdmit']

model_sklearn(X, Y)

# Use statsmodels to build the model

y_pred = model_stats(X, Y)

def dia_res(y_true, y_pred):

    residual = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6,2.5))

    _, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)
dia_res(df_train['ChanceOfAdmit'], y_pred)
# Do the same preprocess for test data

df_test = df_test.rename(columns={'Serial No.':'SerialNo','GRE Score':'GREScore',

                         'TOEFL Score':'TOEFLScore','LOR ':'LOR',

                         'University Rating':'UniversityRating',

                         'Chance of Admit ':'ChanceOfAdmit'})

df_test_y = df_test['ChanceOfAdmit']
# calculate MSE

def evaluate_model(x, y_true):

    model = joblib.load('model.h5')

    pred = model.predict(x)

    mse = mean_squared_error(df_test_y, pred)

    

    return mse
test_X = scaling(df_test[num_cols], train=False)

mse = evaluate_model(test_X, df_test_y)

print(mse)
# Use sklearn to build the model  

X = scaling(df_train[rest_cols]) 

Y = df_train['ChanceOfAdmit']

model_sklearn(X, Y)
# Use statsmodels to build the model

y_pred = model_stats(X, Y)
dia_res(df_train['ChanceOfAdmit'], y_pred)
test_X = scaling(df_test[rest_cols], train=False)

mse = evaluate_model(test_X, df_test_y)

print(mse)
# Use sklearn to build the model  

X = scaling(df_train[num_cols + rest_cols]) 

Y = df_train['ChanceOfAdmit']

model_sklearn(X, Y)
# Use statsmodels to build the model

y_pred = model_stats(X, Y)
dia_res(df_train['ChanceOfAdmit'], y_pred)
test_X = scaling(df_test[num_cols + rest_cols], train=False)

mse = evaluate_model(test_X, df_test_y)

print(mse)