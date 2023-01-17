## Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
## Read in the data file

df = pd.read_csv('../input/mental-health-in-tech-survey/survey.csv')
df.shape
df.head()
## Set the style of our plots
sns.set_style('darkgrid')
sns.set_palette('pastel')
sns.countplot(x=df['treatment'])
sns.countplot(x=df['treatment'],hue=df['family_history'])
## Visualize the missing data
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
## Display the column-wide percentages of missing data.
null = df.isnull().sum()
(null[null!=0]/len(df)).sort_values(ascending=False)
df = df.drop(['state','comments','Timestamp'],axis=1)
df['self_employed'].value_counts()
df['self_employed'] = df['self_employed'].fillna('No')
## People with no condition
df[df['treatment']=='No']['work_interfere'].value_counts(dropna=False)
## People with a condition
df[df['treatment']=='Yes']['work_interfere'].value_counts(dropna=False)
condition = df[df['treatment']=='Yes']
nc = df[df['treatment']=='No']

nc['work_interfere'] = nc['work_interfere'].fillna('Never')
condition['work_interfere'] = condition['work_interfere'].fillna('Sometimes')

df = pd.concat([condition,nc],axis=0)
## All purple means no missing data
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
for col in df.columns:
    print(col,df[col].nunique(),sep='\t')
df.drop(['Gender','Country'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder

def encode_df(dataframe):
    le = LabelEncoder()
    for col in dataframe.columns:
        dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe
df = encode_df(df)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)

scale = scaler.transform(df)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scale)
x_pca = pca.transform(scale)
plt.figure(figsize=(8,6)) 
plt.scatter(x_pca[:,0],x_pca[:,1],c='red',s=14,alpha=.7)
plt.colorbar()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('treatment',axis=1))

scale = scaler.transform(df.drop('treatment',axis=1))
pca = PCA(n_components=2)
pca.fit(scale)
x_pca = pca.transform(scale)
plt.figure(figsize=(8,6)) 
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['treatment'],cmap='RdYlBu',s=14,alpha=.7)
plt.colorbar()
plt.figure(figsize=(8,6)) 
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['supervisor'],cmap='RdYlBu',s=14,alpha=.7)
plt.colorbar()
plt.figure(figsize=(8,6)) 
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['coworkers'],cmap='RdYlBu',s=14,alpha=.7)
plt.colorbar()
plt.figure(figsize=(8,6)) 
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['seek_help'],cmap='RdYlBu',s=14,alpha=.7)
plt.colorbar()
