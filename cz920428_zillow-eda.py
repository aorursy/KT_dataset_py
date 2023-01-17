# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read data into notebook

#parcelid, logerror, transactiondate

train = pd.read_csv('../input/train_2016_v2.csv')

#this file has many features of the property and parcelid 

all_2016 = pd.read_csv('../input/properties_2016.csv') 
train.head()
all_2016.head()

all_2016.shape
#Analyze train first

train['transactiondate'] = pd.to_datetime(train['transactiondate'])

train['TransactionMonth'] = train['transactiondate'].dt.strftime('%b')

count = train['TransactionMonth'].value_counts()

import seaborn as sns

sns.barplot(count.index, count.values, order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',

                                              'Sep','Oct','Nov','Dec'])

#More transactions during summer. Dataset contains all transactions prior to Oct 15 2016 and 

#some transactions after Oct 15 2016. So there is less transactions in Oct, Nov and Dec. 

#Need to consider Seasonality when making predication. Jan, Feb lower transaction volumns
#Analyze the logerror 

import matplotlib.pyplot as plt

plt.hist(train['logerror'],np.linspace(-0.5, 0.5, 100), alpha=0.7)
#mean absolute log error with time 

#During winter time, the mean of absolute logerror is larger than summer time. 

mean = abs(train['logerror']).groupby(train['TransactionMonth']).mean()

sns.stripplot(x=pd.Series(mean.index), y=pd.Series(mean.values), order = ['Jan','Feb','Mar',

                                                                          'Apr','May','Jun',

                                                                          'Jul','Aug','Sep',

                                                                          'Oct','Nov','Dec'],

             size=10)
null_value = all_2016.isnull().sum().reset_index()

null_value.columns = ['feature','CountNA']

null_value = null_value.sort_values('CountNA', ascending=False)

fig, ax = plt.subplots()

fig.set_size_inches(11, 15)

graph = sns.barplot(x='CountNA',y='feature',data=null_value, ax=ax)

graph.set(xlabel='Count of NA values', ylabel='Features')
