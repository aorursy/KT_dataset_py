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

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn import metrics
def print_missing(df):

    m_count = df.isnull().sum()

    m_pct = m_count / len(df) *100



    temp = pd.DataFrame(dict(count=m_count, percent=m_pct, dtype=df.dtypes)) #pd.concat([m_count,m_pct], axis=1)

    print(temp.sort_values(by='count', ascending=False).head(20))

    return temp.sort_values(by='count', ascending=False)

    

def encode_df(df):

    le = LabelEncoder()



    categorical_feature_mask = df.dtypes==object

    categorical_cols = df.columns[categorical_feature_mask].tolist()



    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

    print(df[categorical_cols].head(10))

    

def regression_results(y_true, y_pred):



    # Regression metrics

    explained_variance=metrics.explained_variance_score(y_true, y_pred)

    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 

    mse=metrics.mean_squared_error(y_true, y_pred) 

    try:

        mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)

    except:

        print('cannot do mean squared log')

        mean_squared_log_error=-100

    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)

    r2=metrics.r2_score(y_true, y_pred)



    print('explained_variance: ', round(explained_variance,4))    

    print('mean_squared_log_error: ', round(mean_squared_log_error,4))

    print('r2: ', round(r2,4))

    print('MAE: ', round(mean_absolute_error,4))

    print('MSE: ', round(mse,4))

    print('RMSE: ', round(np.sqrt(mse),4))

    

def feature_select_score(df):

    lr = LinearRegression()

    scores = cross_val_score(lr, df.drop(['LogPrice'],axis=1),df.LogPrice, cv=10,scoring='neg_mean_squared_error')

    scores = -scores

    mean = scores.mean()

    rmse = np.sqrt(mean)

    print(scores)

    print(mean, rmse, np.exp(rmse))

    return rmse
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print(train.shape, test.shape, sample_submission.shape)
missing = print_missing(train)
train['LogPrice'] = np.log(train.SalePrice)

cor = train.corr()

cor_table = pd.DataFrame({'feature': cor.index, 'cor':cor.SalePrice}).sort_values(by='cor', ascending=False)

print(cor_table)



cor = train.corr()

cor_table['logcor'] = cor.LogPrice

print(cor_table)
cor_missing = cor_table.copy()

cor_missing['mcount'] = missing['count']

print(cor_missing)
fig, ax = plt.subplots()

# sns.barplot(x='YrSold', y='LogPrice', data=train, ax=ax)

# sns.barplot(x='YrSold', y='SalePrice', data=train, ax=ax)

sns.violinplot(x='YrSold', y='SalePrice', data=train, ax=ax);
full_feature_list = cor_table[2:]

processed_train, processed_test = pd.DataFrame(), pd.DataFrame()

processed_train['LogPrice'] = train.LogPrice

o_all = [train, test]

p_all = [processed_train, processed_test]

print(full_feature_list[:6])
sns.boxplot(x='OverallQual',y='LogPrice',data=train);
fig, axs = plt.subplots(ncols=3, nrows=2, sharey=True,figsize=(10,6))

# sns.scatterplot(x=train.OverallQual, y=train.LogPrice, ax=axs[0][0])

# sns.scatterplot(x=train.GrLivArea, y=train.LogPrice, ax=axs[0][1])

# sns.scatterplot(x=train.GarageCars, y=train.LogPrice, ax=axs[0][2])

# sns.scatterplot(x=train.GarageArea, y=train.LogPrice, ax=axs[1][0])

# sns.scatterplot(x=train.TotalBsmtSF, y=train.LogPrice, ax=axs[1][1])

# sns.scatterplot(x=train['1stFlrSF'], y=train.LogPrice, ax=axs[1][2]);



for f, n in zip(full_feature_list.index[:6],range(6)):

    sns.scatterplot(x=train[f], y=train.LogPrice, ax=axs[int(n/3)][n%3]);

rmse_log = []
for df1, df2 in zip(o_all, p_all):

    df2['OverallQual'] = df1['OverallQual'].astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['Garage'] = (df1.GarageCars*df1.GarageArea).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['LivingArea'] = (df1.GrLivArea+df1['1stFlrSF']).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['TotalBsmtSF'] = df1.TotalBsmtSF.astype('float64')

rmse_log.append(feature_select_score(processed_train))
sns.lineplot(y=rmse_log, x=range(len(rmse_log)))

plt.ylim(0,0.25);
from sklearn.preprocessing import MinMaxScaler

s = MinMaxScaler()

scaled = s.fit_transform(processed_train.drop('LogPrice', axis=1))

scaled = pd.DataFrame(scaled)

scaled['LogPrice'] = processed_train.LogPrice

feature_select_score(scaled)
fig, axs = plt.subplots(ncols=3, nrows=2, sharey=True,figsize=(10,6))



for f, n in zip(full_feature_list.index[6:12],range(6)):

    sns.scatterplot(x=train[f], y=train.LogPrice, ax=axs[int(n/3)][n%3]);
for df1, df2 in zip(o_all, p_all):

    df2['FullBath'] = (df1.FullBath).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['TotRmsAbvGrd'] = (df1.TotRmsAbvGrd).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['YearBuilt'] = (df1.YearBuilt).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['YearRemodAdd'] = (df1.YearRemodAdd).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['GarageYrBlt'] = (df1.GarageYrBlt.fillna(1790)).astype('float64')

rmse_log.append(feature_select_score(processed_train))
for df1, df2 in zip(o_all, p_all):

    df2['MasVnrArea'] = (df1.MasVnrArea.fillna(df1.MasVnrArea.median())).astype('float64')

rmse_log.append(feature_select_score(processed_train))
s = MinMaxScaler()

scaled = s.fit_transform(processed_train.drop('LogPrice', axis=1))

scaled = pd.DataFrame(scaled)

scaled['LogPrice'] = processed_train.LogPrice

feature_select_score(scaled)
sns.lineplot(y=rmse_log, x=range(len(rmse_log)))

plt.ylim(0,0.25);
lr = LinearRegression()

lr.fit(processed_train.drop('LogPrice', axis=1),processed_train.LogPrice)

y_pred = lr.predict(processed_test.fillna(processed_test.mean()))

output = pd.DataFrame({'Id':test.Id, 'SalePrice':np.exp(y_pred)})

output.head()
output.to_csv('lr_submission02.csv', index=None)