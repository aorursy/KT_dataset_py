import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import scipy.stats as stats
df=pd.read_excel("/kaggle/input/superstore/US Superstore data.xls")
df.head()
df.tail()
df.info()
df.describe()
df.columns
pd.set_option('display.max_columns', None)

df.head()
df.isnull().sum().sum()
df["Segment"].value_counts().plot(kind="bar")
data=df[['Sales','Quantity','Discount','Profit']]

sns.heatmap(data.corr(),annot=True)
df['Category'].value_counts()
df['Category'].value_counts().plot(kind="bar")
pd.crosstab(df['Segment'],df['Category']).plot(kind="bar",stacked=True)
sns.distplot(df["Sales"])
df.columns


sns.scatterplot("Sales",'Profit',data=df)
axes,fig=plt.subplots(0,1,figsize=(18,5))

sns.scatterplot("Discount",'Profit',data=df)
axes,fig=plt.subplots(0,1,figsize=(18,5))

sns.scatterplot('Sales','Discount',data=df)
df['Sub-Category'].value_counts().plot(kind="bar")
pd.crosstab(df["Region"],df["Category"],df["Profit"],aggfunc='sum').plot(kind="bar",stacked=True)
pd.crosstab(index=df["Category"],columns=df["Segment"],values=df["Profit"],aggfunc="sum").plot(kind="bar",stacked=True)
pd.crosstab(index=df["Category"],columns=df["Ship Mode"],values=df["Profit"],aggfunc="sum").plot(kind="bar",stacked=True)
sns.lmplot(x="Profit",y="Sales",data=df,fit_reg=False,col="Category")

plt.show()
sns.lmplot(x="Profit",y="Sales",data=df,fit_reg=False,col="Ship Mode")
axes,fig=plt.subplots(0,1,figsize=(18,5))

sns.barplot("Sub-Category","Profit",data=df)
sns.scatterplot("Quantity","Profit",data=df)