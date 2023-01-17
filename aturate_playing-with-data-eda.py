# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loding the Customer Data
cust_raw = pd.read_csv('../input/Mall_Customers.csv')
cust_raw.head()
# Encoding Gender so as to make it numerical Male-0, Female-1
Gender_num = { 'Gender' : {'Male': 0 , 'Female':1}}
cust_raw.replace(Gender_num, inplace=True)
cust_raw.head()
# Getting info about the data
cust_raw.info()
cust_raw.describe()
# Getting count of records age wise
bins = pd.IntervalIndex.from_tuples([(17, 25), (25, 50), (50, 70)])
cust_gp_age = cust_raw.groupby([cust_raw.Gender,pd.cut(cust_raw.Age, bins)])['CustomerID']\
                    .count().reset_index().rename(columns={'CustomerID':'Count'})
sns.set(style="whitegrid")
sns.barplot(x='Age', y='Count',hue='Gender',data=cust_gp_age)
print(cust_gp_age.sort_values(by=['Count'],ascending = False))
cust_corr = cust_raw.corr()
sns.heatmap(cust_corr,cmap="YlGnBu")
cust_corr
# ScatterPlot b/w Age and Spending score
plt.subplots(figsize=(10,8))
sns.scatterplot(x="Age", y="Spending Score (1-100)",data = cust_raw,hue="Gender",\
                size="Annual Income (k$)")
sns.lmplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue = 'Gender' ,data=cust_raw);