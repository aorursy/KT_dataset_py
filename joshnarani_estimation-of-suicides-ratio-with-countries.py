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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import os

data = pd.read_csv('../input/master.csv')

data.info()
data.head()
data.plot.scatter('year','suicides_no')
data.columns.values
data[['sex','suicides_no']].groupby(['sex']).mean().sort_values(by='suicides_no', ascending=True).plot.bar()
age_su = data[['age','suicides_no']].groupby(['age']).sum()

age_su.plot(kind='bar', figsize=(40,10), fontsize=25,color='pink').set_title('age vs suicides_no',fontsize=40)
cont_su = data[['country','suicides_no']].groupby(['country']).sum()

cont_su.plot(kind='bar', figsize=(40,10), fontsize=25,color='orange').set_title('country vs suicides_no',fontsize=40)
cont_su = cont_su.reset_index().sort_values(by='suicides_no', ascending=False)

top5 = cont_su[:5]

sns.barplot(x='country', y='suicides_no', data=top5).set_title('countries with most suicides')

plt.xticks(rotation=90)
cont_su = cont_su.reset_index().sort_values(by='suicides_no', ascending=False)

bot = cont_su[-10:]

sns.barplot(x='country', y='suicides_no', data=bot).set_title('countries with less suicides')

plt.xticks(rotation=90)
data[['year','suicides_no']].groupby(['year']).sum().plot()
top5data = data.loc[data['country'].isin(top5.country)]

country_suicides_sex = top5data[['country','suicides_no','sex']].groupby(['country','sex']).sum().reset_index().sort_values(by='suicides_no', ascending=False)

plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.barplot(x='country', y='suicides_no', hue='sex', data=country_suicides_sex).set_title('countries suicides rate w.r.t sex')
list_year = data.year

count_year = Counter(list_year)

most = count_year.most_common(15)



x, y = zip(*most)

x , y = list(x), list(y)



plt.figure(figsize = (15, 8))

sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(len(x)), order = x)

plt.xlabel('Years')

plt.ylabel('Frequency')

plt.title('the years with high suicides',color='green')

plt.show()
female_data = data.loc[data['sex']=='female']

female_suicides = female_data[['country','suicides_no','sex']].groupby(['country','sex']).sum().reset_index().sort_values(by='suicides_no', ascending=False)

plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.barplot(x='country', y='suicides_no', data=female_suicides).set_title('females suicide rate w.r.t country',fontsize=40)
data.head()
cont_su = data[['generation','suicides_no']].groupby(['generation']).sum()

cont_su.plot(kind='bar', figsize=(40,10), fontsize=25,color='green').set_title('generation vs suicides_no',fontsize=40)
sns.heatmap(data.corr())
data=pd.get_dummies(data)
from sklearn.model_selection import train_test_split
y=data['suicides_no']
x_train, x_test, y_train, y_test = train_test_split(data,y, test_size=0.2,random_state=0)
x_train=data.drop('suicides_no',axis=1)
x_test=data.drop('suicides_no',axis=1)
y_train=data['suicides_no']
y_test=data['suicides_no']
x_train.shape,y_train.shape
x_test.shape,y_test.shape
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)
x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True)
from sklearn.linear_model import LinearRegression
lg=LinearRegression()
lg.fit(x_train,y_train)
pred=lg.predict(x_test)
lg.score(x_train,y_train)
pred