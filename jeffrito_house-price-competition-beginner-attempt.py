import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualiation tools

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# sci-kit learn tools

from scipy.stats import skew



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# given data imports

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# copies of DS for manipulation (we will use this for remainder of project)

train_df = train.copy()

test_df = test.copy()



# copies for EDA purposes

EDA_train = train.copy()

EDA_test = test.copy()



# combine data sets to avoid dimension misalignment

all_data = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],

                      test_df.loc[:,'MSSubClass':'SaleCondition']))



print(train_df.shape, test_df.shape, all_data.shape)
# drop target (dependent variable) from training dataframe

actual_y = train_df['SalePrice']

#train_df = train_df.drop('SalePrice', axis=1)



train_df.shape
# from Abhinand "Predicting HousingPrices: Simple Approach" Kernel

def show_all(df):

    #This fuction lets us view the full dataframe

    with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):

        display(df)
show_all(train_df.head())
f, ax = plt.subplots(figsize=(12, 6))

sns.distplot(actual_y)

print("Skew is: ", actual_y.skew())
log_actual_y = np.log(actual_y)

f, ax = plt.subplots(figsize=(12, 6))

sns.distplot(log_actual_y)



print("Skew is: ", log_actual_y.skew())
quant = train.select_dtypes(include=[np.number])

quant.dtypes
corr = quant.corr()

print(corr['SalePrice'].sort_values(ascending=False)[:5])

print(corr['SalePrice'].sort_values(ascending=False)[-5:])
corr_map = train_df.corr()

fig, ax = plt.subplots(figsize=(20,16))

sns.heatmap(corr_map, vmax=.8, square=True, annot=True, fmt='.1f')

plt.show();
# Top 10 high correlation to SalePrice matrix

n = 10

cols = corr_map.nlargest(n, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

fig, ax = plt.subplots(figsize=(10,8))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_df.OverallQual.unique()
# pivot table to further investigate relationship between 'OverallQual' and 'SalePrice'

quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

quality_pivot
f, ax = plt.subplots(figsize=(8, 4))

sns.lineplot(x='OverallQual', y = train_df.SalePrice, color='green',data=train_df)
plt.figure(figsize=(8, 6), dpi=80)

plt.scatter(x = train_df['GrLivArea'], y = log_actual_y)

plt.ylabel('LogSalePrice')

plt.xlabel('GrLivArea')

plt.show()
# remove outliers and update EDA_train

EDA_train = EDA_train[EDA_train['GrLivArea'] < 4000]



plt.figure(figsize=(8, 6), dpi=80)

plt.scatter(x = EDA_train['GrLivArea'], y = np.log(EDA_train.SalePrice))

plt.xlim(-200,6000) # keeps same scale as first scatter plot

plt.ylabel('LogSalePrice')

plt.xlabel('GrLivArea')

plt.show()
# lets do the same for all data

#all_data = all_data[all_data['GrLivArea'] < 4000]
plt.figure(figsize=(8, 6), dpi=80)

plt.scatter(x = train_df['GarageArea'], y = np.log(train_df.SalePrice))

plt.ylabel('LogSalePrice')

plt.xlabel('GarageArea')

plt.show()
# remove outliers and update train_df

EDA_train = EDA_train[EDA_train['GarageArea'] < 1200]



plt.figure(figsize=(8, 6), dpi=80)

plt.scatter(x = EDA_train['GarageArea'], y = np.log(EDA_train.SalePrice))

plt.xlim(-50,1475)

plt.ylabel('LogSalePrice')

plt.xlabel('GarageArea')

plt.show()
# instead of removing outlier rows, lets try to impute them with a value

# dropping rows with outliers is misaligning my data and preventing submission

#all_data = all_data[all_data['GarageArea'] < 1200]
# Number of missing values in each column of training data

missing_vals = (train_df.isnull().sum())

print(missing_vals[missing_vals > 0])
# GrLivArea

f, ax = plt.subplots(figsize=(8, 4))

sns.distplot(train_df['GrLivArea'])

print("Skew is: ", train_df['GrLivArea'].skew())
train_df['GrLivArea'] = np.log(train_df['GrLivArea'])



f, ax = plt.subplots(figsize=(8, 4))

sns.distplot(train_df['GrLivArea'])

print("Skew is: ", train_df['GrLivArea'].skew())
# TotalBsmtSF

f, ax = plt.subplots(figsize=(8, 4))

sns.distplot(train_df['TotalBsmtSF'])

print("Skew is: ", train_df['TotalBsmtSF'].skew())
all_data.shape
# we will begin by applying log transformation to skewed numeric features

num_data = all_data.dtypes[all_data.dtypes != "object"].index



skew_data = all_data[num_data].apply(lambda x: skew(x.dropna()))

skew_data = skew_data[skew_data > 0.75]

skew_data = skew_data.index



all_data[skew_data] = np.log1p(all_data[skew_data])
all_data.shape
# drop all features with missing values, noted above : keep electrical

all_data = all_data.drop((missing_vals[missing_vals > 1]).index,1)

#all_data = all_data.drop(all_data.loc[all_data['Electrical'].isnull()].index)



# fix few number of missing vals in test set

all_data = all_data.fillna(all_data.mean())
all_data.shape
all_data = pd.get_dummies(all_data)
# drop variables noted in EDA section

drop_me = ['GarageArea', '1stFlrSF', 'TotRmsAbvGrd']

all_data = all_data.drop(drop_me, axis=1)
# quick look under the hood

show_all(all_data.head())

print(all_data.shape)
# split concatonated data into train and test dataframes



y = np.log1p(train_df["SalePrice"])

train_df = train_df.drop('SalePrice', axis=1)

X_train = all_data[:train_df.shape[0]]

X_test = all_data[train_df.shape[0]:]





X_train.shape
# Root-Mean-Squared-Error (RMSE) evaluation metric

from sklearn.model_selection import cross_val_score



# from "Regularized Linear Models" w/ cross validation

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
# Linear Regression !

from sklearn import linear_model



linear_model = linear_model.LinearRegression()

lr_model = linear_model.fit(X_train, y)



rmse_cv(lr_model).mean()
# LassoCV !

from sklearn.linear_model import LassoCV

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



lasso_model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 0.005, 0.0001, 0.5, 0.2]).fit(X_train, y)

rmse_cv(lasso_model).mean()
from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(random_state=42, max_depth = 6, n_jobs = 5)

rf_model.fit(X_train, y)



rmse_cv(rf_model).mean()
linear_pred = np.expm1(lr_model.predict(X_test))

lasso_pred = np.expm1(lasso_model.predict(X_test))
lasso_pred.shape
#submit!

output = pd.DataFrame({"id":test.Id, "SalePrice":lasso_pred})

output.to_csv("lasso_solution.csv", index = False)