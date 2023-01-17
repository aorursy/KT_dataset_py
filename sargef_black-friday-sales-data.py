import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

df = pd.read_csv('../input/black-friday/BlackFriday.csv')
df.info()
df.head(5)
df.isnull().sum()
modifieddf=df.fillna(0)
df.isnull().sum()
df.head(5)
df['Age'].value_counts().head(5)
df['Insights'] = df['Age'].apply(lambda title: title.split(':')[0])

sns.countplot(x='Insights',data=df,palette='viridis')
ByAge = df.groupby(by=['Gender','Age']).count()['Insights'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(ByAge,cmap='viridis')
df['Gender'].value_counts().head()
sns.countplot(df['Gender'],data=df,palette='viridis')
ByGender = df.groupby(by=['Age', 'Gender']).count()['Insights'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(ByGender,cmap='viridis')
df['Occupation'].value_counts().head(10)
sns.countplot(df['Occupation'],data=df,palette='viridis')
ByOccupation = df.groupby(by=['Occupation','Age']).count()['Insights'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(ByOccupation,cmap='viridis')
ByOccupation2 = df.groupby(by=['Occupation','Gender']).count()['Insights'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(ByOccupation2,cmap='viridis')
df['Stay_In_Current_City_Years'].value_counts().head(10)
sns.countplot(df['Stay_In_Current_City_Years'],data=df,palette='viridis')
ByCityStay = df.groupby(by=['Age','Stay_In_Current_City_Years']).count()['Insights'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(ByCityStay,cmap='viridis')
df['Marital_Status'].value_counts().head(10)
sns.countplot(df['Marital_Status'],data=df,palette='viridis')