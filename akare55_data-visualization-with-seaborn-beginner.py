# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading data from file.
data=pd.read_csv('../input/data.csv',encoding="windows-1252")
#print(data)
col=data.columns
print(col)
# Data have 33 columns and 569 entries
data.info()

data.head(10)
# y includes our labels and x includes our features
y=data.diagnosis # M or B
list=['Unnamed: 32','id','diagnosis']
x=data.drop(list,axis=1)
x.head()
ax=sns.countplot(y,label="Count")   
B,M=y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant: ',M)
x.describe()

data_dia=y # data diagnosis
data=x     # dropped data  
data_n_2=(data-data.mean())/(data.std())
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value') # melt can run without var_name and value_name
print(data)

#loc gets rows (or columns) with particular labels from the index. 
#iloc gets rows (or columns) at particular positions in the index (so it only takes integers).
# first ten features
data_dia=y # data diagnosis
data=x     # dropped data  
data_n_2=(data-data.mean())/(data.std())

# standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')

# violin plot
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

# box plot
plt.figure(figsize=(10,10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
# swarm plot
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
# correlation map
f,ax = plt.subplots(figsize=(10, 10))
a=x.iloc[:,:10]
sns.heatmap(a.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
# Second ten features
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data, id_vars="diagnosis", var_name="features", value_name="value")

plt.figure(figsize=(10,10))
sns.violinplot(x="features",y='value',hue='diagnosis',data=data,split=True,inner="quart")
plt.xticks(rotation=90)
plt.show()

# boxplot
f,ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
plt.show()
# swarm plot
plt.figure(figsize=(15,15))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
# correlation map
f,ax = plt.subplots(figsize=(10, 10))
a=x.iloc[:,10:20]
sns.heatmap(a.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
# Rest of features
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data, id_vars="diagnosis", var_name="features", value_name="value")

plt.figure(figsize=(10,10))
sns.violinplot(x="features",y='value',hue='diagnosis',data=data,split=True,inner="quart")
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10,10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
# swarm plot
plt.figure(figsize=(15,15))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
# correlation map
f,ax = plt.subplots(figsize=(10, 10))
a=x.iloc[:,20:31]
sns.heatmap(a.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

