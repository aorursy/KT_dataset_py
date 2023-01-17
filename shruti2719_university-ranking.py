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
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
data= pd.read_csv('../input/cwurData.csv')

data.info()

data.describe()

data.year.unique()

ins_count = data[data['year'] == 2015].groupby('country').size().sort_values(ascending = False)

plt.figure(figsize = (15,15))

ax = sns.barplot(x = ins_count.values, y = ins_count.index)

ax.set(xlabel = 'Country', ylabel = 'Number of Institution')

for i in ax.patches:

    ax.text(i.get_width()+3.0, i.get_y()+0.6,i.get_width().astype(int), color='black', ha="center")

plt.xticks(rotation = 70)

plt.show()
top_count = data[data['year'] == 2015].head(100).groupby('country').size().sort_values(ascending = False)

plt.figure(figsize = (15,10))

ax = sns.barplot(x = top_count.values, y = top_count.index)

ax.set(xlabel = 'Country', ylabel = 'Number of top 100 Institution')

for i in ax.patches:

    ax.text(i.get_width()+0.5, i.get_y()+0.6,i.get_width().astype(int))

plt.xticks(rotation = 70)

plt.show()
per_count = top_count/ins_count

per_count.dropna(inplace = True)

per_count.sort_values(ascending = False, inplace = True)

plt.figure(figsize = (15,10))

ax = sns.barplot(x = per_count.values, y = per_count.index)

ax.set(xlabel = 'Country', ylabel = 'Percentage of top 100 Institution')

for i in ax.patches:

    ax.text(i.get_width(), i.get_y()+0.6,str(round(i.get_width()*100,1))+'%')

plt.xticks(rotation = 70)

plt.show()
# Identify Universities whose rank has been non-decreasing.
institution = list(data.institution.unique())

non_decreasing = pd.DataFrame(data=institution,columns=['institution'])

non_decreasing.head()
def non_decreasing_rank(institution):

    world_rank = list(data[data.institution == institution]['world_rank'])

    count = data.groupby('institution').size()[institution]

    for i in range(1,count): #1,2,3

        if world_rank[i-1] < world_rank[i]:

            return False

    return True

non_decreasing[non_decreasing['institution'].apply(non_decreasing_rank) == True]
rank2015 = data[data.year == 2015]

rank2015.drop(['country','national_rank','year','broad_impact'],axis = 1, inplace = True)

rank2015.head()
y = rank2015.quality_of_education.max() + 1
factor = list(rank2015.columns.values)[2:9]

factor
for i in range(len(factor)):

    z = rank2015[factor[i]].apply(lambda x:y-x)

    plt.figure(i)

    sns.regplot(x=z, y='score', data = rank2015)
cor = pd.DataFrame()

for i in range(len(factor)):

    cor[factor[i]] = rank2015[factor[i]].apply(lambda x:y-x)

cor['score'] = rank2015.score

cor.corr() 
score = data.score

train = data[factor] 

lab_enc = preprocessing.LabelEncoder()

score_encoded = lab_enc.fit_transform(score)

x_train, y_train, x_test, y_test = train_test_split(train,score_encoded,train_size = 0.9, random_state = 0)
tree = DecisionTreeClassifier()

tree.fit(x_train,x_test)

y_pred = tree.predict(y_train)

y1 = lab_enc.inverse_transform(y_test)

y2 = lab_enc.inverse_transform(y_pred)

np.corrcoef(y1,y2)

sns.regplot(y1,y2)
fit = 0

for i in range(len(y1)):

    if (y1[i] - 0.5) <= y2[i] <= (y1[i] + 0.5):

        fit = fit + 1

        

print(fit/len(y1))

forest = RandomForestClassifier()

forest.fit(x_train,x_test)

y_pred = forest.predict(y_train)

y1 = lab_enc.inverse_transform(y_test)

y2 = lab_enc.inverse_transform(y_pred)

sns.regplot(y1,y2)

np.corrcoef(y1,y2)
fit = 0

for i in range(len(y1)):

    if (y1[i] - 0.5) <= y2[i] <= (y1[i] + 0.5):

        fit = fit + 1

        

print(fit/len(y1))