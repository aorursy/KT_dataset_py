import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
colors = ['#001c57','#50248f','#a6a6a6','#38d1ff']
sns.palplot(sns.color_palette(colors))
train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')
test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv.zip')
sub = pd.read_csv('../input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')

print("Train shape : ", train.shape)
print("Test shape : ", test.shape)
train.head()
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.distplot(train.y.values, bins=50, color=colors[1])
plt.title('Распределение целевой переменной - y\n',fontsize=15)
plt.xlabel('Значение в секундах'); plt.ylabel('Кол-во (частота)');

plt.subplot(122)
sns.boxplot(train.y.values, color=colors[3])
plt.title('Распределение целевой переменной - y\n',fontsize=15)
plt.xlabel('Значение в секундах'); 
train.y.describe()
train.dtypes.value_counts()
train.dtypes[train.dtypes=='float']
dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
train.dtypes[train.dtypes=='object']
obj = train.dtypes[train.dtypes=='object'].index
for i in obj:
    print(i, train[i].unique())
train.isna().sum()[train.isna().sum()>0]
fig,ax = plt.subplots(len(obj), figsize=(20,70))

for i, col in enumerate(obj):
    sns.boxplot(x=col, y='y', data=train, ax=ax[i])
num = train.dtypes[train.dtypes=='int'].index[1:]
nan_num = []
for i in num:
    if (train[i].var()==0):
        print(i, train[i].var())
        nan_num.append(i)
train = train.drop(columns=nan_num, axis=1)
for i in obj:
    le = LabelEncoder()
    le.fit(list(train[i].values) + list(train[i].values))
    train[i] = le.transform(list(train[i].values))
train[obj].head()
corr = train[train.columns[1:10]].corr()

fig,ax = plt.subplots(figsize=(7,6))
sns.heatmap(corr, vmax=.7, square=True,annot=True);
threshold = 1

corr_all = train.drop(columns=obj, axis=1).corr()
corr_all.loc[:,:] =  np.tril(corr_all, k=-1) 
already_in = set()
result = []
for col in corr_all:
    perfect_corr = corr_all[col][corr_all[col] == threshold ].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)
result
train.T.drop_duplicates().T