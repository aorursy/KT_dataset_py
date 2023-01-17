# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/startup_funding.csv")
data.head()
pd.isnull(data['AmountInUSD']).sum()
# Finding the datatype
data.dtypes
#Finding the NaN values with count
pd.isnull(data).sum()
pd.isnull(data).sum()/data.shape[0]*100
#Diagrammatically finding the missing values
missingno.matrix(data)
data.describe(include="all")# remove the amount usd
data.shape
#optimized code
data = pd.read_csv("../input/startup_funding.csv")
def temp(v):
    try:
        return pd.to_datetime(v.replace('.','/').replace('//','/'))
    except:
        print(v)
data['Date']=pd.to_datetime(data['Date'].apply(lambda v: temp(v)))
data['month_year']=data['Date'].dt.strftime('%Y-%m')
data['amount']=data['AmountInUSD'].str.replace(',','').astype(float)
print(data[['Date','month_year','amount']].head())

data['amount']=data['AmountInUSD'].str.replace(',','').astype(float)
data["month_year"].value_counts().plot.bar(figsize=(12,5),color='green')
data.groupby(["month_year"]).size().plot.bar(figsize=(12,5),color='green')
x=data['IndustryVertical'].value_counts()/data.shape[0]*100
x.head(10).plot.bar(figsize=(15,5),color="blue")
y=data['CityLocation'].value_counts()/data.shape[0]*100
y.head(10).plot.bar(figsize=(15,5),color="blue")
a=data['InvestorsName'].value_counts()/data.shape[0]*100
a.head(10).plot.bar(figsize=(15,5),color="blue")
b=data['SubVertical'].value_counts()/data.shape[0]*100
b.head(10).plot.bar(figsize=(15,5),color="green")
data.groupby(['month_year'])['amount'].mean().sort_values(ascending=False ).plot.bar(figsize=(15,5),color="pink")
data.groupby(['SubVertical'])['amount'].max().sort_values(ascending=False ).head(10).plot.bar(figsize=(15,5),color="pink")
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind 
import seaborn as sns
from scipy.stats import chi2_contingency

# finding outliers
p=data.boxplot(column='amount',by='InvestmentType',figsize=(12,5),rot=10)
p.grid(False)
data.groupby("InvestmentType")['amount'].mean()
obs=data.groupby(['InvestmentType','IndustryVertical']).size()
obs.name = 'Freq'# renaming the values
obs= obs.reset_index()# this index is used renaming to retain in dataframe
obs=obs.pivot_table(index ='IndustryVertical',columns = 'InvestmentType',values='Freq')
sns.heatmap(obs,cmap="Blues")
stat,p,dof,exp=chi2_contingency(obs.fillna(0).values)##exp is expected values
p  # they are dependent(Accepting H1 hypothesis)(p< 0.05).