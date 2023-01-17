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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import numpy as np

#테스트 데이터의 'target' 변수를 결측값으로 설정 (두 개의 데이터를 합쳐야 하기 때문에)

test['target'] = np.nan

train['train'] = 1

test['train'] = 0



#데이터 합치기

df  = pd.concat([train, test], axis = 0)



del train['train']

del test['train']

del test['target']
df.head()
train.head()
test.head()
import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt
sns.countplot(train['target'])

print(train['target'].value_counts())
variable = []

for i in np.arange(0,200).astype(str):

    variable.append('var_'+i)
for col in variable:

    sns.distplot(df[df['train']==1][col])

    sns.distplot(df[df['train']==0][col])

    plt.legend(['Train','Test'])

    plt.show()
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(10,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(10,10,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]

features = train.columns.values[2:102]

plot_feature_distribution(t0, t1, '0', '1', features)
t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]

features = train.columns.values[102:202]

plot_feature_distribution(t0, t1, '0', '1', features)
for col in variable:

    sns.distplot(df[df['target']==1][col])

    sns.distplot(df[df['target']==0][col])

    plt.legend(['1','0'])

    plt.show()
corr = df.corr()

cmap = sns.color_palette("Blues")

f, ax = plt.subplots(figsize=(30,30))

sns.heatmap(corr, cmap=cmap)
for col in train.columns:

    msg = 'column:{:>10}\t Percent of NaN value:{:.2f}%'.format(col, 100*(train[col].isnull().sum()/train[col].shape[0]))

    print(msg)
for col in test.columns:

    msg = 'column:{:>10}\t Percent of NaN value:{:.2f}%'.format(col, 100*(test[col].isnull().sum()/test[col].shape[0]))

    print(msg)