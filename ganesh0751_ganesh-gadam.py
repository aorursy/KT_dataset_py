import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('../input/cwurData.csv')
data.head()
data.tail()
data.info()
data.describe()
a= data[data['year'] == 2015].groupby('country').size().sort_values(ascending = False)

plt.figure(figsize = (15,15))

ax = sns.barplot(x = a.values, y = a.index)

ax.set(xlabel = 'Country', ylabel = 'Number of Institution')

for i in ax.patches:

    ax.text(i.get_width()+3.0, i.get_y()+0.6,i.get_width().astype(int), color='black', ha="center")

plt.xticks(rotation = 70)

plt.show()
#top 100 company in 2015

b = data[data['year'] == 2015].head(100).groupby('country').size().sort_values(ascending = False)

plt.figure(figsize = (15,14))

ax = sns.barplot(x = b.values, y = b.index)

ax.set(xlabel = 'Country', ylabel = 'Number of top 100 Institution')

for i in ax.patches:

    ax.text(i.get_width()+0.8, i.get_y()+0.9,i.get_width().astype(int))

plt.xticks(rotation = 70)

plt.show()
c = b/a

c.dropna(inplace = True)

c.sort_values(ascending = False, inplace = True)

plt.figure(figsize = (15,16))

ax = sns.barplot(x = c.values, y = c.index)

ax.set(xlabel = 'Country', ylabel = 'Percentage of top 100 Institution')

for i in ax.patches:

    ax.text(i.get_width(), i.get_y()+0.6,str(round(i.get_width()*100,1))+'%')

plt.xticks(rotation = 70)

plt.show()
institution = list(data.institution.unique())

non_decreasing = pd.DataFrame(data=institution,columns=['institution'])

non_decreasing.head()
def non_decreasing_data(institution):

    world_rank = list(data[data.institution == institution]['world_rank'])

    count = data.groupby('institution').size()[institution]

    for i in range(1,count): #1,2,3

        if world_rank[i-1] < world_rank[i]:

            return False

    return True

non_decreasing[non_decreasing['institution'].apply(non_decreasing_data) == True]
g = data[data.year == 2015]

g.drop(['country','national_rank','year','broad_impact'],axis = 1, inplace = True)

g.head()
y = g.quality_of_education.max() + 1
o = list(g.columns.values)[2:9]

o
for i in range(len(o)):

    z = g[o[i]].apply(lambda x:y-x)

    plt.figure(i)

    sns.regplot(x=z, y='score', data = g)
cor = pd.DataFrame()

for i in range(len(o)):

    cor[o[i]] = g[o[i]].apply(lambda x:y-x)

cor['score'] = g.score

cor.corr() 
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
score = data.score

train = data[o] 

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
neigh = KNeighborsClassifier()

neigh.fit(x_train, x_test)

y_pred = neigh.predict(y_train)

y1 = lab_enc.inverse_transform(y_test)

y2 = lab_enc.inverse_transform(y_pred)
sns.regplot(y1,y2)

np.corrcoef(y1,y2)
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