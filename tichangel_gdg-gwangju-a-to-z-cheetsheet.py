# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
#train데이터와 test데이터를 읽어옵니다.

df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head()
#목적변수의 중위값, 최소값, 최대값 등을 살펴봅니다.

df_train['price'].describe()
f, ax = plt.subplots(figsize=(8,6))

sns.distplot(df_train['price'])
print("왜도: %f" % df_train['price'].skew())

print("첨도: %f" % df_train['price'].kurt())
fig = plt.figure(figsize = (15, 10))

fig.add_subplot(1,2,1)

res = stats.probplot(df_train['price'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['price']), plot=plt)
df_train['price'] = np.log1p(df_train['price'])
print("왜도: %f" % df_train['price'].skew())

print("첨도: %f" % df_train['price'].kurt())

df_train['price'].describe()
#histogram

plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])