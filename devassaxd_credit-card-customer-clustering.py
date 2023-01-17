# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("/kaggle/input/ccdata/CC GENERAL.csv")
data.head()
data.info()
data=data.dropna()
data=data.reset_index().drop("index",axis=1)
data.drop("CUST_ID", axis=1, inplace=True)
data.info()
data.describe()
for i in list(data.columns)[:-1]:
    plt.figure(figsize=(10,5))
    sns.distplot(data[i], bins=100)
    plt.show()
len(data[data["TENURE"]==12])/len(data)
#grouping variables with the same order of magnitude
v1=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
        'PAYMENTS', 'MINIMUM_PAYMENTS']
v2=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']
v3=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']
data[v1].describe()
data[v2].describe()
data.describe()["BALANCE"]["std"]
def cutoff_function(col,data,metrics):
    std_value=metrics[col]["std"]
    mean_value=metrics[col]["mean"]
    return data[data[col]<=(mean_value+3*std_value)]
metrics=data.describe()
for i in v1:
    data=cutoff_function(i,data,metrics)
for i in v3:
    data=cutoff_function(i,data,metrics)
data=data.reset_index().drop("index",axis=1)
data.info()
for i in list(data.columns)[:-1]:
    plt.figure(figsize=(10,5))
    sns.distplot(data[i], bins=100)
    plt.show()
data.describe()
def range_function(col,data,metrics):
    col_s=list(data[col])
    std=metrics[col]["std"]
    col_s=pd.Series(col_s/std)
    return pd.Series(col_s.apply(apply_funct))

def apply_funct(val):
    if val<=0.5:
        return 1
    elif val<=1:
        return 2
    elif val<=1.5:
        return 3
    elif val<=2:
        return 4
    elif val<=2.5:
        return 5
    elif val<=3:
        return 6
    elif val<=3.5:
        return 7
    elif val<=4:
        return 8
    elif val<=4.5:
        return 9
    elif val<=5:
        return 10
    else:
        return 11
metrics=data.describe()
for i in v1:
    col_name=i+"_RANGE"
    v4.append(col_name)
    data[col_name]=range_function(i,data,metrics)
for i in v3:
    col_name=i+"_RANGE"
    v5.append(col_name)
    data[col_name]=range_function(i,data,metrics)
def v2_range(val):
    if val<=0.1:
        return 1
    elif val<=0.2:
        return 2
    elif val<=0.3:
        return 3
    elif val<=0.4:
        return 4
    elif val<=0.5:
        return 5
    elif val<=0.6:
        return 6
    elif val<=0.7:
        return 7
    elif val<=0.8:
        return 8
    elif val<=0.9:
        return 9
    else:
        return 10
for i in v2:
    col_name=i+"_RANGE"
    data[col_name]=data[i].apply(v2_range)
data.head()
data.drop(v1+v2+v3+["TENURE"], axis=1, inplace=True)
data.head()
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,20):
    km=KMeans(n_clusters=i)
    km.fit(data)
    inertia.append(km.inertia_)
plt.plot(range(1,20), inertia, "--o")
plt.title("Elbow method")
plt.xlabel("K")
plt.ylabel("Inertia")
km=KMeans(n_clusters=4)
km.fit(data)
clusters=km.fit_predict(data)
data["Clusters"]=clusters
for column in data:
    g=sns.FacetGrid(data,col="Clusters")
    g.map(sns.distplot,column,kde=False)
km=KMeans(n_clusters=5)
km.fit(data)
clusters=km.fit_predict(data)
data["Clusters"]=clusters
for column in data:
    g=sns.FacetGrid(data,col="Clusters")
    g.map(sns.distplot,column,kde=False)
