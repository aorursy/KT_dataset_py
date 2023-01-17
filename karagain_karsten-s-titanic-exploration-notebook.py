# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
plt.tight_layout()
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.shape
df.count()
df.describe()
df.Cabin.value_counts().head()
# This code shows what expressions were used to categorize the Sexes
# In addition, it shows in the training sample, there are more men than women. 
df.Sex.value_counts()
fig = plt.figure(figsize=(20,10))

plt.subplot(231)
plt.title('Gender of Passengers')
df.Sex.value_counts(normalize=True).plot(kind = 'bar')

plt.subplot(232)
plt.title('Number of Survivors')
df.Survived.value_counts(normalize=True).plot(kind = 'bar')

plt.subplot(234)
plt.title('Female Survivors')
plt.bar(['Survived', 'Deceased'],df.Survived[df.Sex == 'female'].value_counts(normalize=True), color = ['c', 'm'])

plt.subplot(235)
plt.title('Male Survivors')
plt.bar(['Deceased', 'Survived'],df.Survived[df.Sex == 'male'].value_counts(normalize=True), color = ['r', 'b'])

plt.subplot(236)
plt.title('Deceased of both Genders')
plt.bar(['Male', 'Female'], df.Sex[df.Survived == 0].value_counts(normalize=True), color = ['r', 'm'])

fig = plt.figure(figsize=(20,10))

plt.subplot2grid((2,3), (0,0))
plt.title('Age WRT Survival')
plt.scatter(df.Survived, df.Age, alpha=0.1)

plt.subplot2grid((2,3), (0,1))
plt.title('Age WRT Survival')
plt.scatter(df.Survived[df.Age<=18], df.Age[df.Age <=18])
print(
    df.Age[df.Survived==1].describe(),
    '\n',
    df.Age[df.Survived==0].describe()
)
print('Age Distribution for Female',
      '\n',
    df.Age[(df.Survived==1) & (df.Sex=='female')].describe(),
    '\n',
    df.Age[(df.Survived==0) & (df.Sex=='female')].describe()
)
print('Age Distribution for Male',
      '\n',
    df.Age[(df.Survived==1) & (df.Sex=='male')].describe(),
    '\n',
    df.Age[(df.Survived==0) & (df.Sex=='male')].describe()
)
df.Pclass.value_counts()
# This result for how much each person paid is too long, instead I should create a graph.  
df.Fare.value_counts().head()
# I want to do some more complex visualizations. Seaborn will help with this
import seaborn as sns

fig = plt.figure(figsize=(20, 10) )

plt.subplot2grid((3,3), (0, 0), colspan=3)
df.Fare.hist(bins=70)
plt.xlabel('Price')
plt.ylabel('Purchased')
plt.title('Price of Fares Purchased')


print('\n \n')

plt.subplot2grid((3,3), (1, 0), colspan=2, rowspan=2)
# plt.scatter(df.Pclass, df.Fare)
sns.stripplot(df.Pclass, df.Fare, jitter=True, edgecolor='none', alpha=0.5)
# http://dataviztalk.blogspot.com/2016/02/how-to-add-jitter-to-plot-using-pythons.html
plt.title('Price of Fares split by Class')
sns.boxplot(df.Pclass, df.Fare)
fig = plt.figure(figsize=(20,10))
for x in [1, 2, 3]:
    plt.subplot(230+ x)
    names = ['Upper Class', 'Middle Class', 'Lower Class']
    plt.title('Survivability of the ' + names[x-1])
    df.Survived[df.Pclass==x].value_counts(normalize=True).plot(kind='bar')
plt.subplot()
plt.hist(df.Fare[df.Survived==1], color = 'orange', bins=15, alpha=0.5, label='Survived')
plt.hist(df.Fare[df.Survived==0], color = 'blue', bins=15, alpha=0.5, label='Deceased')
plt.title('Fares and Survivability')
plt.legend()
plt.ylabel('Count')
plt.xlabel('Fare Price')
print(df.Fare.corr(df.Survived))
print(df.Pclass.corr(df.Survived))