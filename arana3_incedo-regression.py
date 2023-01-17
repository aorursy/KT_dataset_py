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
# Import data load 



train=pd.read_csv('../input/train_file.csv')

test=pd.read_csv('../input/test_file.csv')

sample=pd.read_excel('../input/sample_submission08f968d.xlsx')
#Check train data

train.head()
#Check train columns and test columns

train.columns,test.columns
# drop columns

train.drop(['Patient_ID','LocationDesc','Greater_Risk_Question','Description','GeoLocation'],axis=1,inplace=True)

test.drop(['Patient_ID','LocationDesc','Greater_Risk_Question','Description','GeoLocation'],axis=1,inplace=True)
# check nan value in train and test

train.isnull().any().sum(),test.isnull().any().sum()
# Import seaborn and maplotlib

import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(10,6))

plt.subplot(1,2,1)

sns.distplot(train['Sample_Size'])





plt.subplot(1,2,2)

sns.boxplot(train['Sample_Size'])
#train=train.loc[(train['Sample_Size']<19000)]
train['Sample_Size']=np.log(train['Sample_Size'])

test['Sample_Size']=np.log(test['Sample_Size'])



plt.figure(figsize=(10,6))

plt.subplot(1,2,1)

sns.distplot((train['Sample_Size']))





plt.subplot(1,2,2)

sns.boxplot((train['Sample_Size']))
# probability greater than 1

train=train[train['Greater_Risk_Probability']>1]

# Check with year how is Greater_Risk_Probability 

plt.figure(figsize=(18,7))

sns.boxplot(data=train,x='YEAR',y='Greater_Risk_Probability')

# It is decreasing
# Check how Subtopic is related to Greater_Risk_Probability

sns.boxplot(data=train,x='Subtopic',y='Greater_Risk_Probability')

# Check gender 

sns.boxplot(data=train,x='Sex',y='Greater_Risk_Probability')
# Check with Race

plt.figure(figsize=(16,7))

plt.xticks( rotation='45')

sns.boxplot(data=train,x='Race',y='Greater_Risk_Probability')

# Check with Grade

plt.figure(figsize=(16,7))

sns.boxplot(data=train,x='Grade',y='Greater_Risk_Probability')

# With QuestionCode

plt.figure(figsize=(19,5))

plt.xticks(rotation='45')

sns.boxplot(data=train,x='QuestionCode',y='Greater_Risk_Probability')
# With StratID1

sns.boxplot(data=train,x='StratID1',y='Greater_Risk_Probability')
# Check with StratID2

sns.boxplot(data=train,x='StratID2',y='Greater_Risk_Probability')

sns.boxplot(data=train,x='StratID3',y='Greater_Risk_Probability')
sns.boxplot(data=train,x='StratificationType',y='Greater_Risk_Probability')
train['Sample_Size']=np.round(train['Sample_Size'])

test['Sample_Size']=np.round(test['Sample_Size'])
train.head()
test.head()
#train.columns

from sklearn.preprocessing import LabelEncoder

Feature =['YEAR','Sex', 'Race', 'QuestionCode','StratificationType']

for i in Feature:

    LR=LabelEncoder()

    train[i] = LR.fit_transform(train[i])
for i in Feature:

    LR=LabelEncoder()

    test[i] = LR.fit_transform(test[i])
test.head()