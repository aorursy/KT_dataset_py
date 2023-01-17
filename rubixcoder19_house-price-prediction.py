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
pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',100)
housing = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

housing.head()
housing.info()
housing.shape
null_values = pd.DataFrame({'null_val':round(100*housing.isnull().sum()/len(housing),2)})

null_values
nul_val_cols = list((null_values[null_values['null_val']> 0.0]).index)

nul_val_cols
housing = housing.drop(columns=nul_val_cols)

housing.info()
housing_clean_cols = list(housing.columns)

housing_clean_cols
null_values = pd.DataFrame({'null_val':round(100*housing.isnull().sum()/len(housing),2)})

null_values
import seaborn as sns

import matplotlib.pyplot as plt
# plt.figure(figsize=(30,25))

# sns.heatmap(round(housing.corr(),2),annot=True)

# plt.show()
plt.figure(figsize=(30,25))

sns.heatmap(round(housing.corr()>0.20,2),annot=True)

plt.show()
h_corr = round(housing.corr(),2)

h_corr = h_corr['SalePrice']
col_list =[]

for i in range(len(h_corr)):

    if h_corr.values[i] >=0.2:

        col_list.append(h_corr.index[i])

col_list
housing = housing[col_list]

housing.info()
housing.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99])
# sns.pairplot(housing)

# plt.show()
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score,f1_score,r2_score

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

scalar = MinMaxScaler()
cols = list(housing.columns)

cols
# housing[cols] = scalar.fit_transform(housing[cols])

housing.describe()
plt.figure(figsize=(30,25))

sns.heatmap(round(housing.corr(),2),annot=True)

plt.show()
y_train = housing.pop('SalePrice')

X_train = housing
X_train.describe()

y_train.describe()
sns.distplot(y_train)

plt.show()
X_train.isnull().sum()
# Calculate the VIFs for the model

def vif_calc(X_train_rfe):

    vif = pd.DataFrame()

    X = X_train_rfe

    X = X.drop(['const'], axis=1)

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif



# Get the variable having max p-value

def get_p_max(model_num):

    model_summary = pd.DataFrame()

    model_summary['p_val'] = model_num.pvalues

    model_summary['coef'] = model_num.params

    max_p_val = model_summary.loc[model_summary['p_val']== model_summary['p_val'].max()]

    print("R-squared: ",model_num.rsquared)

    print("Adj R-squared: ",model_num.rsquared_adj)

    print("P(F-statistics): ",model_num.f_pvalue)

    return max_p_val
# Model 1

# with statsmodels

X_train = sm.add_constant(X_train) # adding a constant

 

m1 = sm.OLS(y_train, X_train).fit()

predictions = m1.predict(X_train) 

m_1 = get_p_max(m1)

print(m_1,"\n\n",vif_calc(X_train))
# Model 2

# with statsmodels

X_train_fe = X_train.drop(columns=m_1.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m2 = sm.OLS(y_train, X_train_fe).fit()

predictions = m2.predict(X_train_fe) 

m_2 = get_p_max(m2)

print(m_2,"\n\n",vif_calc(X_train_fe))
# Model 3

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_2.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m3 = sm.OLS(y_train, X_train_fe).fit()

predictions = m3.predict(X_train_fe) 

m_3 = get_p_max(m3)

print(m_3,"\n\n",vif_calc(X_train_fe))

# Model 4

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_3.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m4 = sm.OLS(y_train, X_train_fe).fit()

predictions = m4.predict(X_train_fe) 

m_4 = get_p_max(m4)

print(m_4,"\n\n",vif_calc(X_train_fe))

# Model 5

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_4.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m5 = sm.OLS(y_train, X_train_fe).fit()

predictions = m5.predict(X_train_fe) 

m_5 = get_p_max(m5)

print(m_5,"\n\n",vif_calc(X_train_fe))
# Model 6

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_5.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m6 = sm.OLS(y_train, X_train_fe).fit()

predictions = m6.predict(X_train_fe) 

m_6 = get_p_max(m6)

print(m_6,"\n\n",vif_calc(X_train_fe))
# Model 7

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_6.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m7 = sm.OLS(y_train, X_train_fe).fit()

predictions = m7.predict(X_train_fe) 

m_7 = get_p_max(m7)

print(m_7,"\n\n",vif_calc(X_train_fe))
# Model 8

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_7.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m8 = sm.OLS(y_train, X_train_fe).fit()

predictions = m8.predict(X_train_fe) 

m_8 = get_p_max(m8)

print(m_8,"\n\n",vif_calc(X_train_fe))
# Model 9

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=['OverallQual'])

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m9 = sm.OLS(y_train, X_train_fe).fit()

predictions = m9.predict(X_train_fe) 

m_9 = get_p_max(m9)

print(m_9,"\n\n",vif_calc(X_train_fe))
# Model 10

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=m_9.index.values)

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m10 = sm.OLS(y_train, X_train_fe).fit()

predictions = m10.predict(X_train_fe) 

m_10 = get_p_max(m10)

print(m_10,"\n\n",vif_calc(X_train_fe))
# Model 11

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=['TotalBsmtSF'])

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m11 = sm.OLS(y_train, X_train_fe).fit()

predictions = m11.predict(X_train_fe) 

m_11 = get_p_max(m11)

print(m_11,"\n\n",vif_calc(X_train_fe))
# Model 12

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=['YearBuilt'])

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m12 = sm.OLS(y_train, X_train_fe).fit()

predictions = m12.predict(X_train_fe) 

m_12 = get_p_max(m12)

print(m_12,"\n\n",vif_calc(X_train_fe))
# Model 13

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=['GarageCars'])

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m13 = sm.OLS(y_train, X_train_fe).fit()

predictions = m13.predict(X_train_fe) 

m_13 = get_p_max(m13)

print(m_13,"\n\n",vif_calc(X_train_fe))
# Model 14

# with statsmodels 

X_train_fe = X_train_fe.drop(columns=['1stFlrSF'])

X_train_fe = sm.add_constant(X_train_fe) # adding a constant

 

m14 = sm.OLS(y_train, X_train_fe).fit()

predictions = m14.predict(X_train_fe) 

m_14 = get_p_max(m14)

print(m_14,"\n\n",vif_calc(X_train_fe))
y_pred = m14.predict(X_train_fe)
sns.scatterplot(y_pred,y_train)
#Actual vs Predicted

fig = plt.figure(figsize = (15,6))

c = [i for i in range(len(X_train))]

plt.plot(c,y_train, color="blue")

plt.plot(c,y_pred, color="red")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('price', fontsize=16)                               # Y-label

plt.grid(1)

plt.show()
cols = list(housing.select_dtypes(include=['int64','float64']))

cols
X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

X_t = X_test.copy()
y_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
X_test['SalePrice'] = y_test['SalePrice']

X_test[col_list].info()
X_test[col_list].info()
# X_test[col_list] = scalar.transform(X_test[col_list])
y_sample = X_test['SalePrice']
X_test = X_test[col_list].drop(columns = 'SalePrice')

X_test= sm.add_constant(X_test)
X_test.info()
X_test = X_test[list(X_train_fe.columns)]

X_test.info()
# Predicting the Sale price for test dataset

y_p = m14.predict(X_test)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_sample,y_p)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
#Actual vs Predicted

fig = plt.figure(figsize = (15,6))

c = [i for i in range(len(X_test))]

plt.plot(c,y_sample, color="blue")

plt.plot(c,y_p, color="red")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('price', fontsize=16)                               # Y-label

plt.grid(1)

plt.show()
# Error terms

fig = plt.figure(figsize = (15,6))

c = [i for i in range(len(X_test))]

plt.scatter(c,y_sample-y_p, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('Views_show-Predicted_views', fontsize=16)                # Y-label

plt.grid(1)

plt.show()
submission = pd.DataFrame({'Id': X_t['Id'],'SalePrice': y_p})
compression_opts = dict(method='zip',archive_name='out.csv')  

submission.to_csv('out.zip', index=False,compression=compression_opts)  # submission.to_csv('D:\iwork\file_name.csv')