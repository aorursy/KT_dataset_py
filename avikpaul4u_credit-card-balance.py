import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

print(os.listdir("../input"))

Credit_df = pd.read_csv("../input/Credit.csv",index_col=0)
print(Credit_df.shape)
Credit_df.head()
Credit_df.info()
Credit_df.describe()
Missing_val =Credit_df.isnull().sum()

print(Missing_val)
sns.distplot(Credit_df.Balance, color='teal')
f, axes = plt.subplots(2, 2, figsize=(15, 6))

f.subplots_adjust(hspace=.3, wspace=.25)

Credit_df.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')

Credit_df.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')

Credit_df.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')

Credit_df.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')
sns.heatmap(Credit_df.corr(), cmap='BuGn')
Credit_df_Numeric=pd.DataFrame(Credit_df[['Income','Limit','Rating','Cards','Age','Education','Balance']])

X = Credit_df_Numeric.assign(const=0)

pd.Series([variance_inflation_factor(X.values, i) 

               for i in range(X.shape[1])], 

              index=X.columns)
mod_reg = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity',

               data = Credit_df).fit()

mod_reg.summary()
len(Credit_df[Credit_df.Balance==0])
Credit_df_active= Credit_df[Credit_df.Balance >0]
sns.distplot(Credit_df_active.Balance, color='teal')
f, axes = plt.subplots(2, 2, figsize=(15, 6))

f.subplots_adjust(hspace=.3, wspace=.25)

Credit_df_active.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')

Credit_df_active.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')

Credit_df_active.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')

Credit_df_active.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')
mod_active = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity',

               data = Credit_df_active).fit()

mod_active.summary()