import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv',na_values='?')

df.head()
df.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)

plt.title('Histogram for every feature')

plt.show()
# dropping features that aren't used in this regression model

n_df = df.drop(['id','date','lat','long'], axis=1)

n_df.head()
corr = n_df.corr()

mask = np.zeros_like(corr, dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

plt.figure(figsize=(15,15))

plt.title('Correlation Matrix')

sns.set(font_scale=0.9)

sns.heatmap(

    corr

    , annot=True

    , fmt='.2f'

    , cmap = 'inferno'

    , vmax = 1

    , square = True

    , mask=mask

)

plt.show()
# one-hot encoding for yr_built

year_built_df = pd.get_dummies(n_df['yr_built'])

year_built_df.head()
# one-hot encoding for zipcode column

zipcodes_df = pd.get_dummies(n_df['zipcode'])

zipcodes_df.head()
# divide into categorical and continous dfs

continuos_df = n_df.drop([

    'waterfront'

    , 'view'

    , 'yr_built'

    , 'yr_renovated'

    , 'sqft_above'

    , 'zipcode'

    , 'price'

    , 'sqft_lot15'

    , 'sqft_living15'

    , 'sqft_above'

],axis=1)

continuos_df.head() 
categorical_df = n_df[['waterfront','view']]

categorical_df.head()
# standard scaler

ss = preprocessing.StandardScaler()

ss_array = ss.fit_transform(continuos_df)

ss_array[:5]
# MinMaxScaler

mms = preprocessing.MinMaxScaler()

mms_array = mms.fit_transform(continuos_df)

mms_array[:5]
# Normalizer

norm = preprocessing.Normalizer()

norm_array = norm.fit_transform(continuos_df)

norm_array[:5]
# MaxAbsScaler

mas = preprocessing.MaxAbsScaler()

mas_array = mas.fit_transform(continuos_df)

mas_array[:5]
y = df[['price']]

y.head()
# training data with no scaling

X = pd.concat(

    [

        continuos_df

        , categorical_df

        , zipcodes_df

        , year_built_df

    ]

    , axis = 1 

)

X.head()
# Training data scaled with standard scaler

X_ss = pd.concat(

    [

        pd.DataFrame(ss_array,columns=continuos_df.columns)

        , categorical_df

        , zipcodes_df

        , year_built_df

    ]

    , axis = 1 

)

X_ss.head()
# training data scaled with MinMaxScaler

X_mms = pd.concat(

    [

        pd.DataFrame(mms_array,columns=continuos_df.columns)

        , categorical_df

        , zipcodes_df

        , year_built_df

    ]

    , axis = 1 

)

X_mms.head()
# Training data scaled with a Normalizer

X_norm = pd.concat(

    [

        pd.DataFrame(norm_array,columns=continuos_df.columns)

        , categorical_df

        , zipcodes_df

        , year_built_df

    ]

    , axis = 1 

)

X_norm.head()
# Traning data scaled with a MinAbsScaler

X_mas = pd.concat(

    [

        pd.DataFrame(mas_array,columns=continuos_df.columns)

        , categorical_df

        , zipcodes_df

        , year_built_df

    ]

    , axis = 1 

)

X_mas.head()
# Split with 80:20 ratio for ss training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=69)

print('ss train X and y ->\n','-'*69)

print(X_train[:5])

print(y_train[:5])
# Split with 80:20 ratio for ss training data

X_ss_train, X_ss_test, y_ss_train, y_ss_test = train_test_split(X_ss, y, test_size = 0.2, random_state=69)

print('ss train X and y ->\n','-'*69)

print(X_ss_train[:5])

print(y_ss_train[:5])
# Split with 80:20 ratio for mms training data

X_mms_train, X_mms_test, y_mms_train, y_mms_test = train_test_split(X_mms, y, test_size = 0.2, random_state=69)

print('mms train X and y ->\n','-'*69)

print(X_mms_train[:5])

print(y_mms_train[:5])
# Split with 80:20 ratio for norm training data

X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y, test_size = 0.2, random_state=69)

print('norm train X and y ->\n','-'*69)

print(X_norm_train[:5])

print(y_norm_train[:5])
# Split with 80:20 ratio for mas training data

X_mas_train, X_mas_test, y_mas_train, y_mas_test = train_test_split(X_mas, y, test_size = 0.2, random_state=69)

print('mas train X and y ->\n','-'*69)

print(X_mas_train[:5])

print(y_mas_train[:5])
regression_types = []

train_r2_scores = []

test_r2_scores = []

train_mses = []

train_rmss = []

test_mses = []

test_rmss = []

def add_model_metrics(lr_model,x_tr,y_tr,x_te,y_te,lr_type):

    print('Linear regression type -> ',lr_type)

    r2_train = lr_model.score(x_tr,y_tr)

    r2_test = lr_model.score(x_te,y_te)

    print('R2 score for training set -> ',r2_train)

    print('R2 score for testing set -> ',r2_test)

    pred_train = lr_model.predict(x_tr)

    train_mse = mean_squared_error(y_tr,pred_train)

    train_rms = math.sqrt(mean_squared_error(y_tr,pred_train))

    print('MSE for training set -> ',train_mse)

    print('RMS for training set -> ',train_rms)

    pred_test = lr_model.predict(x_te)

    test_mse = mean_squared_error(y_te,pred_test)

    test_rms = math.sqrt(mean_squared_error(y_te,pred_test))

    print('MSE for training set -> ',test_mse)

    print('RMS for training set -> ',test_rms)

    regression_types.append(lr_type)

    train_r2_scores.append(r2_train)

    test_r2_scores.append(r2_test)

    train_mses.append(train_mse)

    train_rmss.append(train_rms)

    test_mses.append(test_mse)

    test_rmss.append(test_rms)
# Linear regression using no scaling

import math

lr = LinearRegression(fit_intercept=False)

lr.fit(X_train,y_train)

add_model_metrics(lr,X_train,y_train,X_test,y_test,'No Scaling')
# Linear Sink regression using ss scaling

lr_ss = LinearRegression(fit_intercept=False)

lr_ss.fit(X_ss_train,y_ss_train)

add_model_metrics(lr_ss,X_ss_train,y_ss_train,X_ss_test,y_ss_test,'Standard Scaling')
# Linear Sink regression using mms scaling

lr_mms = LinearRegression(fit_intercept=False)

lr_mms.fit(X_mms_train,y_mms_train)

add_model_metrics(lr_mms,X_mms_train,y_mms_train,X_mms_test,y_mms_test,'MinMax Scaling')
# Linear Sink regression using norm scaling

lr_norm = LinearRegression(fit_intercept=False)

lr_norm.fit(X_norm_train,y_norm_train)

add_model_metrics(lr_norm,X_norm_train,y_norm_train,X_norm_test,y_norm_test,'Normalizer')
# Linear Sink regression using mas scaling

lr_mas = LinearRegression(fit_intercept=False)

lr_mas.fit(X_mas_train,y_mas_train)

add_model_metrics(lr_mas,X_mas_train,y_mas_train,X_mas_test,y_mas_test,'MaxAbsScaler')
results_df = pd.DataFrame({

    'Type':regression_types

    , 'Train R2 Scores' : train_r2_scores

    , 'Test R2 Scores' : test_r2_scores

    , 'Train MSE' : train_mses

    , 'Train RMS' : train_rmss

    , 'Test MSE' : test_mses

    , 'Test RMS' : test_rmss

}

)

results_df.head()
plt.figure(figsize=(30,60))

plt.title('Linear Regression with different types of scaling')

plt.subplot(5,2,1)

plt.title(' No Scaling training data')

plt.scatter(y_train,lr.predict(X_train),label='train labels',color='green')

plt.subplot(5,2,2)

plt.scatter(y_test,lr.predict(X_test),label='test labels',color='blue')

plt.title(' No Scaling testing data')

plt.subplot(5,2,3)

plt.title(' StandardScaler training data')

plt.scatter(y_ss_train,lr.predict(X_ss_train),label='train labels',color='green')

plt.subplot(5,2,4)

plt.title(' StandardScaler testing data')

plt.scatter(y_ss_test,lr_ss.predict(X_ss_test),label='test labels',color='blue')

plt.legend()

plt.subplot(5,2,5)

plt.title(' MinMaxScaler training data')

plt.scatter(y_mms_train,lr_mms.predict(X_mms_train),label='train labels',color='green')

plt.subplot(5,2,6)

plt.title(' MinMaxScaler testing data')

plt.scatter(y_mms_test,lr_mms.predict(X_mms_test),label='test labels',color='blue')

plt.legend()

plt.subplot(5,2,7)

plt.title(' Normalizer training data')

plt.scatter(y_norm_train,lr_norm.predict(X_norm_train),label='train labels',color='green')

plt.subplot(5,2,8)

plt.title(' Normalizer testing data')

plt.scatter(y_norm_test,lr_norm.predict(X_norm_test),label='test labels',color='blue')

plt.legend()

plt.subplot(5,2,9)

plt.title(' MaxAbsScaler training data')

plt.scatter(y_mas_train,lr_mas.predict(X_mas_train),label='train labels',color='green')

plt.subplot(5,2,10)

plt.title(' MaxAbsScaler testing data')

plt.scatter(y_mas_test,lr_mas.predict(X_mas_test),label='test labels',color='blue')

plt.legend()

plt.show()
# standard scaled linear regression training coefficients

coef_df_ss = pd.Series(lr_ss.coef_[0],X_ss.columns).sort_values()

for i,val in zip(list(coef_df_ss.index),coef_df_ss):

    print(i,'=',val)
# no scaling linear regression training coefficients

coef_df = pd.Series(lr.coef_[0],X.columns).sort_values()

for i,val in zip(list(coef_df.index),coef_df):

    print(i,'=',val)
# MinMax scaled linear regression training coefficients

coef_df_mms = pd.Series(lr_mms.coef_[0],X_mms.columns).sort_values()

for i,val in zip(list(coef_df_mms.index),coef_df_mms):

    print(i,'=',val)
# Normalized linear regression training coefficients

coef_df_norm = pd.Series(lr_norm.coef_[0],X_norm.columns).sort_values()

for i,val in zip(list(coef_df_norm.index),coef_df_norm):

    print(i,'=',val)
# MaxAbs scaled linear regression training coefficients

coef_df_mas = pd.Series(lr_mas.coef_[0],X_mas.columns).sort_values()

for i,val in zip(list(coef_df_mas.index),coef_df_mas):

    print(i,'=',val)