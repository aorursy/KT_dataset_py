# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
income_df=pd.read_csv("/kaggle/input/incomeexpenditure-dataset/Inc_Exp_Data.csv")
income_df.head()
income_df.info()
income_df.shape
income_df.describe().T
income_df.isna().any()
income_df["Mthly_HH_Expense"].mean()
income_df["Mthly_HH_Expense"].median()
mth_exp_tmp = pd.crosstab(index=income_df["Mthly_HH_Expense"], columns="count")

mth_exp_tmp.reset_index(inplace=True)

mth_exp_tmp[mth_exp_tmp['count'] == income_df.Mthly_HH_Expense.value_counts().max()]
income_df["Highest_Qualified_Member"].value_counts().plot(kind="bar")
income_df.plot(x="Mthly_HH_Income", y="Mthly_HH_Expense")

IQR=income_df["Mthly_HH_Expense"].quantile(0.75)-income_df["Mthly_HH_Expense"].quantile(0.25)

IQR
pd.DataFrame(income_df.iloc[:,0:5].std().to_frame()).T
pd.DataFrame(income_df.iloc[:,0:4].var().to_frame()).T
income_df["Highest_Qualified_Member"].value_counts().to_frame().T
income_df["Highest_Qualified_Member"].value_counts().to_frame().T
income_df["No_of_Earning_Members"].value_counts().plot(kind="bar")
#Here we need to calculate the coeff of variation 



Coeff_of_var_StockA=10/15

print(Coeff_of_var_StockA)

Coeff_of_var_StockB=5/10

print(Coeff_of_var_StockB)