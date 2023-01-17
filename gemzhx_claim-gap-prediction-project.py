# Harry Zheng

# Claim Gap Prediction Project

# EDA
%matplotlib inline

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd 

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/DS_ClaimPeriodProject_withDate.csv')

df.head()
# # of no-null values

df.count()
df.info()
# fill null values with 0

df.fillna(0,inplace=True)
# change the datatype

df['insurance1'] = df['insurance1'].astype('int64')

df['insurance2'] = df['insurance2'].astype('int64')

df['insurance3'] = df['insurance3'].astype('int64')

df['ClaimStartDate'] = df['ClaimStartDate'].astype('datetime64[ns]').dt.date
# get rid of white space happened in TransDiag column

df['TransDiag'] = df['TransDiag'].str.strip()
df.count()
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['TransDiag'] = le.fit_transform(df['TransDiag'])

df.head()
df['HoldOrNot'] = le.fit_transform(df['HoldOrNot'])
df['DeniedOrNot'] = le.fit_transform(df['DeniedOrNot'])
# for numeric data, show the statistics

cols = [5,6,7,8,9,10,11,12,21]

df[df.columns[cols]].describe()
sns.boxplot(data = df['Age'])
# remove the outliers

rare_rows = df.index[df['PaymentAmt']>10000].tolist()

df.drop(rare_rows, inplace=True)  
#check out outliers!!!!!!

fig = plt.figure(figsize = (15,6))

cols_01 = [10,11,12]

pos_df01 = df[df.columns[cols_01]]

sns.boxplot(data = pos_df01)
rare_rows = df.index[df['DatePeriod']>300].tolist()

df.drop(rare_rows, inplace=True)  
# box plot for DatePeriod

fig = plt.figure(figsize = (18,6))

cols_02 = [21]

pos_df02 = df[df.columns[cols_02]]

sns.boxplot(x = 'DatePeriod', data = pos_df02)
# DeniedOrNot

sns.countplot(x="DeniedOrNot", data=df)
#  percentage for Hold or not

sns.countplot(x="HoldOrNot", data=df)
# percentage for Gender

sns.countplot(x="Gender", data=df)
df.count()
df.reset_index(drop=True, inplace = True)

df
# generate a new csv file

#df.to_csv("New_DS_ClaimPeriodProject_withDate.csv", sep=',', index=False)
#headmap

fig = plt.figure(figsize = (10,8))

pos_df05 = df

corr = pos_df05.corr()

corr = corr[abs(corr)>0.2]

sns.heatmap(data = corr,cmap='coolwarm')