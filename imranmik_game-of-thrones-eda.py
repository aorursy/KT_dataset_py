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
import matplotlib.pyplot as plt

import seaborn as sns
battles = pd.read_csv('../input/game-of-thrones/battles.csv')

prediction = pd.read_csv('../input/game-of-thrones/character-predictions.csv')

deaths = pd.read_csv('../input/game-of-thrones/character-deaths.csv')
battles.head()
prediction.head()
deaths.head()
battles.describe()
battles= battles.drop(['attacker_1','attacker_2','attacker_3','attacker_4','defender_1','defender_2','defender_3','defender_4','note'],axis=1)
battles.fillna(method='ffill',inplace=True)
battles.isnull().sum()
battles.head()
battles_per_year = battles.groupby('year',as_index=False).sum()

plt.bar(battles_per_year['year'],battles_per_year['battle_number'])

plt.title('No of battles per year')

plt.xlabel('year')

plt.ylabel('battles')
plt.figure(figsize=(15,5))

sns.countplot(battles['attacker_size'])
plt.figure(figsize=(15,5))

sns.countplot(battles['defender_size'])
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

sns.countplot(battles['attacker_king'])

plt.subplot(1,2,2)

sns.countplot(battles['battle_type'])
pd.crosstab(battles['attacker_king'],battles['attacker_outcome']).plot(kind='bar',figsize=(15,5))

plt.xticks(rotation='horizontal')
plt.figure(figsize=(10,5))

sns.countplot(battles['attacker_king'],hue= battles['battle_type'])
sns.countplot(x= battles['battle_type'],hue=battles['attacker_outcome'])

plt.show()
commanders= battles['attacker_commander'].str.cat(sep=', ').split(', ')

commanders= pd.Series(commanders).value_counts()

graph=commanders.plot.bar()

graph.set_xlim(right=10)
sns.countplot(battles['region'])

plt.xticks(rotation=45)

plt.figure(figsize=(13,4))

plt.subplot(1,2,1)

sns.countplot(deaths['Gender'])



plt.subplot(1,2,2)

sns.countplot(deaths['Allegiances'])

plt.title('Deaths in houses')

plt.xticks(rotation=90)

plt.show()
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
battles.head(5)
battles['attacker_outcome']=battles['attacker_outcome'].map({'win':1,'loss':0})
lb= LabelEncoder()

battles['attacker_king']=lb.fit_transform(battles['attacker_king'])

battles['defender_king']=lb.fit_transform(battles['defender_king'])

battles['battle_type']=lb.fit_transform(battles['battle_type'])

battles['location']=lb.fit_transform(battles['location'])

battles['region']=lb.fit_transform(battles['region'])

battles['attacker_commander']=lb.fit_transform(battles['attacker_commander'])

battles['defender_commander']=lb.fit_transform(battles['defender_commander'])

X= battles.drop(['name','attacker_outcome'],axis=1)

y= battles['attacker_outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
clf= RandomForestClassifier(n_estimators=2500,max_features='sqrt',max_depth=6)

clf.fit(X_train,y_train)

pred= clf.predict(X_test)

a=accuracy_score(y_test,pred)

print("The score is :{}".format(round(a*100,2)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)

scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')

scores.mean()