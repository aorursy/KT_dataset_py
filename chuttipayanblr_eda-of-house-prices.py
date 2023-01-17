# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
traindata =pd.read_csv('../input/train.csv')
traindata.info()
traindata.shape
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
df =traindata.isnull().sum().reset_index(name='MissingValues')
df=df.reset_index()
plt.figure(figsize=(20,20))
sn.barplot(df['index'],df['MissingValues'],data=df)
plt.xticks(rotation=90)

plt.figure(figsize=(20,20))
sn.barplot(x=traindata.MSSubClass.value_counts().index, y=traindata.MSSubClass.value_counts())
pd.crosstab(traindata.MSSubClass,traindata.MSZoning)
plt.figure(figsize=(20,20))
sn.countplot(x="MSSubClass", hue="MSZoning", data=traindata);
traindata.LotFrontage.describe()
sn.distplot(traindata.LotFrontage.dropna())
sn.distplot(traindata.LotArea.dropna())
traindata.drop(labels=['Id'],axis=1,inplace=True)
traindata.info()
traindata["MSSubClass"]=traindata["MSSubClass"].astype('category')
traindata["OverallQual"]=traindata["OverallQual"].astype('category')
traindata["OverallCond"]=traindata["OverallCond"].astype('category')
traindata["GarageCars"]=traindata["GarageCars"].astype('category')
traindata["BsmtFullBath"]=traindata["BsmtFullBath"].astype('category')
traindata["KitchenAbvGr"]=traindata["KitchenAbvGr"].astype('category')
traindata["Fireplaces"]=traindata["Fireplaces"].astype('category')
traindata["YearBuilt"]=traindata["YearBuilt"].astype('category')
traindata["YearRemodAdd"]=traindata["YearRemodAdd"].astype('category')
traindata["MoSold"]=traindata["MoSold"].astype('category')
traindata["YrSold"]=traindata["YrSold"].astype('category')


def plotBarPlots():
    df= traindata.select_dtypes(include=['object','category'])
    for name in df.columns:
        print("Distribution for the feature ",name)
        print(traindata[name].value_counts())
        plt.figure()
        sn.barplot(x=traindata[name].value_counts().index, y=traindata[name].value_counts())
        plt.xticks(rotation=90)
traindata.info()
plotBarPlots()
df= traindata.select_dtypes(include=['int','float64'])
df.columns
for i in range(len(df.columns)):
    for col in df.columns[i:]:
        
        fig = plt.figure()
            
        if col != df.columns[i]:
            plt.scatter(df.iloc[:, i], df[col])
            plt.xlabel(df.columns[i])
            plt.ylabel(col)
            plt.title(df.columns[i] + ' vs ' + col)

df.corr()
