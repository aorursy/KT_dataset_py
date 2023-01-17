# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import visualization

import matplotlib.pyplot as plt

import seaborn as sns



#import preprocessing

from sklearn import preprocessing
dfa = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')

df = dfa.drop(columns=['Unnamed: 0'], axis=1)

df.head()
df.info()
df.shape
df.describe()
df.isnull().sum().sort_values(ascending=False)
#detect percentage of missing value which above 60%

null = df.isnull().sum().sort_values(ascending=False)

print('Percentage of missing value is', round((null[0]/1000)* 100)) #size

print('Percentage of missing value is', round((null[1]/1000)* 100)) #vin

#condition

#percentage missing value of size and vin is more than 60%

#so that size and vin drop from the table
category = ['Sex','Job','Housing','Saving accounts',

            'Checking account','Purpose']

numerical  = df.drop(category, axis=1)

categorical = df[category]

numerical.head()
categorical.head()
#fill value in categorical

for cat in categorical:

    mode = categorical[cat].mode().values[0]

    categorical[cat]=df[cat].fillna(mode)
#detect categorical missing value

categorical.isnull().sum().sort_values(ascending=False) 
#concat table categorical and numerical

dfinal = pd.concat([categorical,numerical],axis=1)

dfinal.head()
fig=plt.figure(figsize=(13,12))

axes=330

#put data numerical

for num in numerical:

    axes += 1

    fig.add_subplot(axes)

    #set title of num

    sns.boxplot(data = numerical, x=num) 

plt.show()
#filter by full-type=gas

data_sex = dfinal[dfinal['Sex']=='female']

top=data_sex.sort_values('Age',ascending=False).head(5)



plt.figure(figsize=(12,6))



x=range(5)

plt.bar(x,top['Age']/6**9)

plt.xticks(x,top['Purpose'])

plt.xlabel('Purpose')

plt.ylabel('Age')

plt.title('10 Most Fuel Type In Cars')

plt.show()

top
fig=plt.figure(figsize=(20,12))

axes=230

#put data categorical

for cat in categorical:

    axes += 1

    fig.add_subplot(axes)

    #set title of cat

    sns.countplot(data = categorical, x=cat) 

plt.show()
fig=plt.figure(figsize=(13,12))

fig.add_subplot(2,2,1)

sns.distplot(numerical['Credit amount'])

fig.add_subplot(2,2,2)

sns.distplot(numerical['Age'])

fig.add_subplot(2,2,3)

sns.distplot(numerical['Duration'])
fig=plt.figure(figsize=(20,12))

axes=230

#put data categorical

for cat in numerical:

    axes += 1

    fig.add_subplot(axes)

    #set title of cat

    sns.violinplot(data = numerical, x=cat) 

plt.show()
#create correlation with hitmap



#create correlation

corr = dfinal.corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (15,12))

fig.set_size_inches(20,5)

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)
sns.pairplot(dfinal[["Age", "Job", "Credit amount", "Duration", "Purpose"]], hue='Purpose', diag_kind='hist')

plt.show()