# to handle datasets

import pandas as pd

import numpy as np



# for plotting

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import StandardScaler



# for variable transformation

import scipy.stats as stats



# to build the models

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb



# to evaluate the models

from sklearn.metrics import mean_squared_error



pd.pandas.set_option('display.max_columns', None)
# load dataset

data = pd.read_csv("../input/train.csv")

print(data.shape)

data.head()
# Load the dataset for submission (the one on which our model will be evaluated by Kaggle)

# it contains exactly the same variables, but not the target



submission = pd.read_csv("../input/test.csv")

submission.head()
# find categorical variables

categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables'.format(len(categorical)))
# find numerical variables

numerical = [var for var in data.columns if data[var].dtype!='O']

print('There are {} numerical variables'.format(len(numerical)))
# let's visualise the values of the discrete variables

discrete = []

for var in numerical:

    if len(data[var].unique())<20:

        print(var, ' values: ', data[var].unique())

        discrete.append(var)

        

print('There are {} discrete variables'.format(len(discrete)))
# let's visualise the percentage of missing values

for var in data.columns:

    if data[var].isnull().sum()>0:

        print(var, data[var].isnull().mean())
# let's inspect the type of those variables with a lot of missing information

for var in data.columns:

    if data[var].isnull().mean()>0.80:

        print(var, data[var].unique())
# first we make a list of continuous variables (from the numerical ones)

continuous = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice']]

continuous
# let's make boxplots to visualise outliers in the continuous variables 

# and histograms to get an idea of the distribution



for var in continuous:

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    fig = sns.boxplot(y=data[var])

    fig.set_title('')

    fig.set_ylabel(var)

    

    plt.subplot(1, 2, 2)

    fig = sns.distplot(data[var].dropna())

    fig.set_ylabel('Number of houses')

    fig.set_xlabel(var)



    plt.show()
# let's look at the distribution of the target variable



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = sns.boxplot(y=data['SalePrice'])

fig.set_title('')

fig.set_ylabel(var)



plt.subplot(1, 2, 2)

fig = sns.distplot(data['SalePrice'].dropna())#.hist(bins=20)

fig.set_ylabel('Number of passengers')

fig.set_xlabel(var)



plt.show()
# outlies in discrete variables

for var in discrete:

    print(data[var].value_counts() / np.float(len(data)))

    print()
for var in categorical:

    print(var, ' contains ', len(data[var].unique()), ' labels')
# Let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size=0.2,

                                                    random_state=0)

X_train.shape, X_test.shape
# print variables with missing data

for col in continuous:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add variable indicating missingness + median imputation

for df in [X_train, X_test, submission]:

    for var in ['LotFrontage', 'GarageYrBlt']:

        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

        df[var].fillna(X_train[var].median(), inplace=True) 



for df in [X_train, X_test, submission]:

    df.MasVnrArea.fillna(X_train.MasVnrArea.median(), inplace=True)
# print variables with missing data

for col in discrete:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# print variables with missing data

for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add label indicating 'Missing' to categorical variables



for df in [X_train, X_test, submission]:

    for var in categorical:

        df[var].fillna('Missing', inplace=True)
# check absence of null values

for var in X_train.columns:

    if X_train[var].isnull().sum()>0:

        print(var, X_train[var].isnull().sum())
# check absence of null values

for var in X_train.columns:

    if X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
# check absence of null values

submission_vars = []

for var in X_train.columns:

    if var!='SalePrice' and submission[var].isnull().sum()>0:

        print(var, submission[var].isnull().sum())

        submission_vars.append(var)
#  I will replace NAN by the median 

for var in submission_vars:

    submission[var].fillna(X_train[var].median(), inplace=True)
def boxcox_transformation(var):

    X_train[var], param = stats.boxcox(X_train[var]+1) 

    X_test[var], param = stats.boxcox(X_test[var]+1) 

    submission[var], param = stats.boxcox(submission[var]+1) 
for var in continuous:

    boxcox_transformation(var)

    

X_train[continuous].head()
# let's  check if the transformation created infinite values

for var in continuous:

    if np.isinf(X_train[var]).sum()>1:

        print(var)
for var in continuous:

    if np.isinf(X_test[var]).sum()>1:

        print(var)
for var in continuous:

    if np.isinf(submission[var]).sum()>1:

        print(var)
# check absence of null values(there should be none)

for var in X_train.columns:

    if X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
# let's make boxplots to visualise outliers in the continuous variables

# and histograms to get an idea of the distribution

# hopefully the transformation yielded variables more "Gaussian"looking





for var in continuous:

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    fig = sns.boxplot(y=X_train[var])

    fig.set_title('')

    fig.set_ylabel(var)

    

    plt.subplot(1, 2, 2)

    fig = sns.distplot(X_train[var].dropna())#.hist(bins=20)

    fig.set_ylabel('Number of passengers')

    fig.set_xlabel(var)



    plt.show()
var = 'SalePrice'

y_train = np.log(y_train) 

y_test = np.log(y_test) 



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = sns.boxplot(y=y_train)

fig.set_title('')

fig.set_ylabel(var)



plt.subplot(1, 2, 2)

fig = sns.distplot(y_train)#.hist(bins=20)

fig.set_ylabel('Number of passengers')

fig.set_xlabel(var)



plt.show()
def rare_imputation(variable):

    # find frequent labels / discrete numbers

    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))

    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]

    

    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')

    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')

    submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')

    

# find unfrequent labels in categorical variables

for var in categorical:

    rare_imputation(var)

    

for var in ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:

    submission[var] = submission[var].astype('int')
def encode_categorical_variables(var, target):

        # make label to house price dictionary

        ordered_labels = X_train.groupby([var])[target].mean().sort_values().index

        ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 

        

        # encode variables

        X_train[var] = X_train[var].map(ordinal_label)

        X_test[var] = X_test[var].map(ordinal_label)

        submission[var] = submission[var].map(ordinal_label)



# encode labels in categorical vars

for var in categorical:

    encode_categorical_variables(var, 'SalePrice')

for var in X_train.columns:

    if var!='SalePrice' and submission[var].isnull().sum()>0:

        print(var, submission[var].isnull().sum())
# let's inspect the dataset

X_train.head()
training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]
# fit scaler

scaler = StandardScaler() # create an instance

scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set for later use
xgb_model = xgb.XGBRegressor()



eval_set = [(X_test[training_vars], y_test)]

xgb_model.fit(X_train[training_vars], y_train, eval_set=eval_set, verbose=False)



pred = xgb_model.predict(X_train[training_vars])

print('xgb train mse: {}'.format(mean_squared_error(y_train, pred)))

pred = xgb_model.predict(X_test[training_vars])

print('xgb test mse: {}'.format(mean_squared_error(y_test, pred)))
SVR_model = SVR()

SVR_model.fit(scaler.transform(X_train[training_vars]), y_train)



pred = SVR_model.predict(scaler.transform(X_train[training_vars]))

print('SVR train mse: {}'.format(mean_squared_error(y_train, pred)))

pred = SVR_model.predict(scaler.transform(X_test[training_vars]))

print('SVR test mse: {}'.format(mean_squared_error(y_test, pred)))
lin_model = Lasso(random_state=2909)

lin_model.fit(scaler.transform(X_train[training_vars]), y_train)



pred = lin_model.predict(scaler.transform(X_train[training_vars]))

print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))

pred = lin_model.predict(scaler.transform(X_test[training_vars]))

print('linear test mse: {}'.format(mean_squared_error(y_test, pred)))
pred_ls = []

pred_ls.append(pd.Series(xgb_model.predict(submission[training_vars])))



pred = SVR_model.predict(scaler.transform(submission[training_vars]))

pred_ls.append(pd.Series(pred))



pred = lin_model.predict(scaler.transform(submission[training_vars]))

pred_ls.append(pd.Series(pred))



final_pred = np.exp(pd.concat(pred_ls, axis=1).mean(axis=1))
temp = pd.concat([submission.Id, final_pred], axis=1)

temp.columns = ['Id', 'SalePrice']

temp.head()