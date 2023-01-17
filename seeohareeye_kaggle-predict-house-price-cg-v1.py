# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

import scipy.stats as stats
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# test data is almost the same size as the training data

print('train data: ' + str(train_data.shape))

print('test data: ' + str(test_data.shape))
# display sample rows of training data

train_data.head()
# training data has 1 dependent variable (SalePrice) and 80 independent variables

print('number of columns: ' + str(len(train_data.columns)))

print(train_data.columns)
# Visualize distribution of SalePrice

# it's non-normal, skewed right. should consider log transform

sns.distplot(train_data['SalePrice'])
# Visualize Correlation for SalesPrice vs. Quant Variables

# to get a sense of which variables have the strongest linear relationship with SalesPrice





quant_train_data = train_data[['SalePrice', 'GarageArea', '1stFlrSF', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'GarageYrBlt',

                               'Fireplaces', 'LotFrontage', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea', 

                               'BedroomAbvGr', 'ScreenPorch', 'PoolArea']]





figure(num=None, figsize=(20, 1), dpi=80, facecolor='w', edgecolor='k')

corr = quant_train_data.corr()

chart = sns.heatmap(corr.iloc[:1, 1:], annot=True, cmap="Blues")

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
# Visualize Correlation across all Quant Variables

# to get a sense of whether variables with strong linear relationships with SalesPrice also have strong linear relationships with each other

# (they do)



figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

corr = quant_train_data.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap="Blues")

plt.show()
# Deep dive on X with high correlation with SalePrice via scatterplot

# there's a positive relationship with makes sense since the 2 variables have a positive correlation

ax = sns.regplot(x="GarageArea", y="SalePrice", data=train_data)
# Deep dive on X with low correlation with SalePrice via scatterplot

# most houses have PoolArea=0, and there's large spread in SalePrice for those that do 

# which makes this variable a good candidate for transformation

ax = sns.regplot(x="PoolArea", y="SalePrice", data=train_data)
# which categorical varibles are ordinal

cat_vars = ['OverallCond', 'Alley', 'Street', 'LandContour', 'MSSubClass', 'MSZoning', 'LotShape','Utilities', 'LotConfig', 

            'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'BldgType']



cat_vars_ordinal = ['OverallCond', 'OverallQual']
# get a sense of how well categorical variables explain variation in SalesPrice with ANOVA results

print('ANOVA p-value results predicting SalesPrice by the following categorical variables:')

stat, pvalue = stats.f_oneway(

            train_data['SalePrice'][train_data['Street'] == 'Pave'],

            train_data['SalePrice'][train_data['Street'] == 'Grvl'])

print('Street: ' + str(round(pvalue, 2)))



stat, pvalue = stats.f_oneway(

            train_data['SalePrice'][train_data['LotConfig'] == 'Inside'],

            train_data['SalePrice'][train_data['LotConfig'] == 'Corner'],

            train_data['SalePrice'][train_data['LotConfig'] == 'CulDSac'],

    train_data['SalePrice'][train_data['LotConfig'] == 'CulDSac']

)

print('LotConfig: ' + str(round(pvalue, 2)))









############## WIP: automate ANOVA results from array of variables

## segment categorical variable by level

# grps = pd.unique(train_data.Street.values)

# grps_data = {grp:train_data['SalePrice'][train_data.Street == grp] for grp in grps}





# x=[]

# for i in range(len(grps)):

#     x.append("grps_data['" + str(grps[i]) + "']")

# print(x)



# stat, pvalue = stats.f_oneway(x)
# Visualize Spread for SalePrice vs. Categorical Variable: Overall Condition

# positive relationship between SalePrice and overall condition, and inbalance between levels



fig = figure(num=None, figsize=(10, 4), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x="OverallCond", y="SalePrice", data=train_data)

sns.swarmplot(x="OverallCond", y="SalePrice", data=train_data, color=".25")
# Visualize Spread for SalePrice vs. Categorical Variable: Overall Quality

# positive relationship between SalePrice and overall quality, and inbalance between levels



fig = figure(num=None, figsize=(10, 4), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x="OverallQual", y="SalePrice", data=train_data)

sns.swarmplot(x="OverallQual", y="SalePrice", data=train_data, color=".25")
# Visualize Spread for SalePrice vs. Categorical Variable: Neighborhood

# appears there is a relationship between SalePrice and neighborhood, however there is inbalance and some neighborhoods have wide spread



fig = figure(num=None, figsize=(10, 4), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x="Neighborhood", y="SalePrice", data=train_data)

chart = sns.swarmplot(x="Neighborhood", y="SalePrice", data=train_data, color=".25")

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
# Create 'DateSold' to review price trend over time



# create date sold variable

first_of_the_month = [1] * len(train_data)

train_data['DaySold'] = first_of_the_month

df2 = train_data[["YrSold", "MoSold", "DaySold"]].copy()

df2.columns = ["year", "month", "day"]

train_data['DateSold'] = pd.to_datetime(df2)



# plot price over date sold

train_data.plot.scatter(x='DateSold',

                      y='SalePrice',

                      c='YrSold',

                      colormap='viridis')



# Create 'Was_Remod' - with caution, since it's not great practice to build boolean variables from continuous variables (lose info)

train_data['Was_Remod'] = np.where(train_data.YearRemodAdd == train_data.YearBuilt, 0, 1)
# combine training and test data, so that pre-processing only has to be done once AND all dummy variables are included during model training
# identify variables by type

y_var = ['SalePrice']

quant_vars = ['Fireplaces', 'LotFrontage', 'LotArea', 'HalfBath', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', 'OpenPorchSF', 'PoolArea', 

              'ScreenPorch','1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'YearRemodAdd', 'YearBuilt']

cat_vars = ['OverallCond', 'Alley', 'Street', 'LandContour', 'MSSubClass', 'MSZoning', 'LotShape','Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 

            'HouseStyle', 'OverallQual', 'BldgType']

cat_vars_ordinal = ['OverallCond', 'OverallQual']



# combine each type

column_names = y_var + quant_vars + cat_vars

# column_names_for_test = quant_vars + cat_vars



# Add empty y column to test (so they can be appended later)

test_data['SalePrice'] = None



# filter data on columns

train_data = train_data[column_names]

test_data = test_data[column_names]

# test_data = test_data[column_names_for_test]



# Identify which data is from train.csv file so they can be seperated after append

train_data['from_train.csv'] = 'Yes'

test_data['from_train.csv'] = 'No'



column_names = train_data.columns + ['from_train.csv']
# Which train data columns have null values? 

column_names = train_data.columns

for i in range(len(column_names)):

    print('%s: ' % (column_names[i])+ str(pd.isnull(train_data[column_names[i]]).sum()))
# Exclude columns with missing data - can revisit imputing missing values later

train_data = train_data.drop(['LotFrontage', 'GarageYrBlt', 'Alley'], axis=1)

test_data = test_data.drop(['LotFrontage', 'GarageYrBlt', 'Alley'], axis=1)



column_names = train_data.columns



print('train data: ' + str(train_data.shape))

print('test data: ' + str(test_data.shape))
# Re order columns

train_data = train_data[column_names]

test_data = test_data[column_names]
# combine data

all_data = train_data.append(test_data) 



print('train data shape: ' + str(train_data.shape))

print('test data shape: ' + str(test_data.shape))

print('train + test shape: ' + str(all_data.shape))
# create dummy variables from non-ordinal categorical Xs

#all_data = pd.get_dummies(all_data, columns=['OverallCond'])

all_data = pd.get_dummies(all_data, columns=['Street'])

all_data = pd.get_dummies(all_data, columns=['LandContour'])

all_data = pd.get_dummies(all_data, columns=['MSSubClass'])

all_data = pd.get_dummies(all_data, columns=['MSZoning'])

all_data = pd.get_dummies(all_data, columns=['LotShape'])

all_data = pd.get_dummies(all_data, columns=['Utilities'])

all_data = pd.get_dummies(all_data, columns=['LotConfig'])

all_data = pd.get_dummies(all_data, columns=['LandSlope'])

all_data = pd.get_dummies(all_data, columns=['Neighborhood'])

all_data = pd.get_dummies(all_data, columns=['BldgType'])

all_data = pd.get_dummies(all_data, columns=['HouseStyle'])

#all_data = pd.get_dummies(all_data, columns=['OverallQual'])
print('# of columns after dummy explosion: ' + str(len(all_data.columns)))
# now that pre-processing is done, separate train and test data

train_data = all_data[all_data['from_train.csv'] == 'Yes']

test_data = all_data[all_data['from_train.csv'] == 'No']



# Exclude column because no longer needed

train_data = train_data.drop(['from_train.csv'], axis=1)

test_data = test_data.drop(['from_train.csv'], axis=1)



train_data.shape
# Split training data into train and validation, in order to get a sense of model performance before submitting 

X_train, X_test, y_train, y_test = train_test_split(train_data, train_data.SalePrice, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
columns = ['FullBath', 'Fireplaces']



# Fit Model

model = LinearRegression(fit_intercept=True)

clf = model.fit(X_train[columns], y_train)



# Model Accuracy: Root Mean Squared Log Error

y_true = y_test

y_pred = model.predict(X_test[columns])

print('Root Mean Squared Log Error on test set: {:.5f}'.format(mean_squared_log_error(y_true, y_pred)))
X_train[train_data.columns]

X_train = X_train.drop(['SalePrice'], axis=1)

X_test = X_test.drop(['SalePrice'], axis=1)

X_train.columns
# Fit Model

model = LinearRegression(fit_intercept=True)

clf = model.fit(X_train, y_train)



# Model Accuracy: Root Mean Squared Log Error

y_true = y_test

y_pred = model.predict(X_test)

print('Root Mean Squared Log Error on test set: {:.5f}'.format(mean_squared_log_error(y_true, y_pred)))
# Model Results  [broken, need to debug]

# mod = sm.OLS(y_train, X_train)

# res = mod.fit()

# print(res.summary())
all_columns = train_data.columns

X = X_train





y=['SalePrice']

X=[i for i in all_columns if i not in y]



from sklearn.feature_selection import RFE



model = LinearRegression(fit_intercept=True)

rfe = RFE(model, 20)

rfe = rfe.fit(X_train, y_train.values.ravel())

print(rfe.support_)

print(rfe.ranking_)
column_rfe = list(zip(train_data.columns, rfe.support_)) 

all_columns = pd.DataFrame(column_rfe, columns = ['column', 'is_sig'])

sig_columns = all_columns[all_columns.is_sig == True]



sig_columns = list(sig_columns['column'])
# Remove variables identified by RFE

model = LinearRegression(fit_intercept=True)

column_names = sig_columns

clf = model.fit(X_train[column_names], y_train)



# Model Accuracy: Root Mean Squared Log Error

y_true = y_test

y_pred = model.predict(X_test[column_names])

print('Root Mean Squared Log Error on test set: {:.5f}'.format(mean_squared_log_error(y_true, y_pred)))
# Model Output - Need to debug

# mod = sm.OLS(y_train, X_train[column_names])

# res = mod.fit()

# print(res.summary())
# Fit Model

columns = ['FullBath', 'Fireplaces']

model = LinearRegression(fit_intercept=True)

clf = model.fit(X_train[columns], y_train)



# Model Accuracy: Root Mean Squared Log Error

y_true = y_test

y_pred = model.predict(X_test[columns])

print('Root Mean Squared Log Error on test set: {:.5f}'.format(mean_squared_log_error(y_true, y_pred)))
# Apply model to test data

test_data = test_data[['FullBath', 'Fireplaces']]

y_pred_final = model.predict(test_data)

y_pred_final = pd.DataFrame(y_pred_final)
# Export predictions for kaggle competition upload

y_pred_final.to_csv('predicted_house_prices_v1.csv')

y_pred_final
# print('# of columns after dummy explosion: ' + str(len(test_data.columns)))

# print(test_data.columns)
# print('shape of train data: ' + str(X_train[column_names].shape)) 



# # filter test data

# test_data = test_data[sig_columns]

# test_data = test_data.drop(['SalePrice'], axis=1)



# print('shape of test data: ' + str(test_data.shape)) 

# print('test data has same # of columns as train data, except for the predictor (SalePrice)')
# filter test data (only need columns considered significant from chosen model)



# sig_cat_columns = ['OverallCond', 'MSSubClass', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual']

# sig_quant_columns = ['HalfBath', 'YearBuilt', 'YearRemodAdd']

# sig_parent_columns = sig_cat_columns + sig_quant_columns

# test_data = test_data[sig_parent_columns]
# Do any of the columns have null values? 



# column_names = test_data.columns

# for i in range(len(column_names)):

#     print('%s: ' % (column_names[i])+ str(pd.isnull(test_data[column_names[i]]).sum()))
# create test data predictions from model built on training data

# y_pred_final = model.predict(test_data)

# y_pred_final = pd.DataFrame(y_pred_final)