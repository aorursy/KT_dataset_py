import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
%pylab inline
%matplotlib inline
from scipy.stats import norm

plt.figure(figsize=(40,40))
num = 0

num_cols = [i for i in data.columns if data[i].dtype != 'object']

for i in num_cols:
    num += 1
    plt.subplot(6,7, num)
    sns.distplot(data[i], fit=norm)
data.isnull().sum()
data['salary'] = data['salary'].fillna(0)
data = data.drop('sl_no', axis=1)

worker_data = data[data['salary'] != 0]
non_worker_data = data[data['salary'] == 0]
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
label.fit(list( data['status'].values ))
data['status'] = label.transform(list( data['status'].values ))  
cat_cols = [i for i in data.columns if data[i].dtype == 'object']

plt.figure(figsize=(30,30))
num = 0
sns.set(font_scale=1)

for i in cat_cols:
    num+=1
    plt.subplot(3,3, num)
    sns.countplot(data[i], hue=data['status'])
print("Success relationship")
for i in cat_cols:
    a = list(data[i].unique())
    for j in a:
        num_suc = len(data[(data[i] == j) & (data['status'] == 1)])
        num_tot = len(data[(data[i] == j)])
        print(i, j, "is", str(num_suc/num_tot))
num_cols = [i for i in data.columns if data[i].dtype != 'object']

plt.figure(figsize=(30,30))
num = 0

for i in num_cols:
    num+=1
    plt.subplot(3,3, num)
    sns.boxplot(x=data['status'], y=data[i])
    plt.ylabel(i)
    plt.xlabel('status')
worker_data = data[data['salary'] != 0]
non_worker_data = data[data['salary'] == 0]
from scipy.stats import norm

sns.distplot(worker_data['salary'], fit=norm)
mu, sigma = norm.fit(worker_data['salary'])
sns.boxplot(worker_data['salary'])
np.quantile(worker_data['salary'], 0.90)
sns.distplot(worker_data['salary'], fit=norm)
worker_data['high_sal'] = 0
worker_data['high_sal'].loc[worker_data['salary'] >= 320000] = 1

# How to turn off comments? :(

worker_data
cat_cols = [i for i in data.columns if data[i].dtype == 'object']

plt.figure(figsize=(30,30))
num = 0
sns.set(font_scale=1)

for i in cat_cols:
    num+=1
    plt.subplot(3,3, num)
    sns.countplot(worker_data[i], hue=worker_data['high_sal'])
num_cols = [i for i in worker_data.columns if worker_data[i].dtype != 'object']

plt.figure(figsize=(30,30))
num = 0

for i in num_cols:
    num+=1
    plt.subplot(3,3, num)
    plt.scatter(x=worker_data['salary'], y=worker_data[i])
    plt.ylabel(i)
    plt.xlabel('salary')
# to be continued..