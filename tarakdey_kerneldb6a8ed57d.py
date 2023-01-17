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
cell_df=pd.read_csv("/kaggle/input/practice-dataset-predict-customer-churn-telecom/Dataset_Cellphone.csv")

print(cell_df.head())
import pandas_profiling

data_profile=cell_df.profile_report(title='Pandas Profiling Report')

#data_profile.to_file(output_file="Dataset Cellphone profiling.html")

print(cell_df.info())
print(cell_df.columns)
import matplotlib.pyplot as plt

import seaborn as sns

plt.hist(cell_df.AccountWeeks,bins=10,alpha=0.65,rwidth=0.95)

plt.title('Graph of AccountWeeks')

print(plt.show())
AccountWeeks_Range=pd.cut(cell_df.AccountWeeks,[0,50,100,150,200,250])

sns.countplot(x=AccountWeeks_Range,hue='Churn', data=cell_df).legend(labels = ["churn customer", "No churn Customer"])

plt.title('AccountWeeks Vs Churn')

print(plt.show())
sns.countplot(x='ContractRenewal',hue='Churn', data=cell_df).legend(labels = ["churn customer", "No churn Customer"])

plt.title('ContractRenewal Vs Churn')

print(plt.show())
DayCalls_Range=pd.cut(cell_df.DayCalls,[0,30,60,90,120,150,180])

fig,ax=plt.subplots(2,2)

sns.countplot(x='DataPlan',hue='Churn', data=cell_df,ax=ax[0,0]).legend(labels = ["churn customer", "No churn Customer"])

sns.countplot(x='CustServCalls',hue='Churn', data=cell_df,ax=ax[0,1]).legend(labels = ["churn customer", "No churn Customer"])

sns.countplot(x=DayCalls_Range,hue='Churn', data=cell_df,ax=ax[1,0]).legend(labels = ["churn customer", "No churn Customer"])

sns.countplot(x='ContractRenewal',hue='Churn', data=cell_df,ax=ax[1,1]).legend(labels = ["churn customer", "No churn Customer"])

print(plt.show())