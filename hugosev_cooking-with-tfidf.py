# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
train['cuisine'].value_counts()[:5]
train['nb'] = train['ingredients'].apply(len)
f, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim([0, 40])
sns.distplot(train['nb'],kde=False,ax=ax,bins=40)
dataset = pd.concat([train,test],sort=True)

ingredients = pd.Series([item for sublist in list(dataset['ingredients']) for item in sublist])\
                                                                .value_counts() #pd.Series
print(ingredients.head(6))

unique_ing = np.array(ingredients.index) #array
print('\n',len(unique_ing),'ingrédients différents en tout')
l=[]
rang =list(range(3,80))
for i in rang:
    l.append(sum(ingredients.values==i))
f, ax = plt.subplots(figsize=(10, 3))
ax.set_xlabel('number of ingredients per recipe')
ax.set_ylabel('number of recipes')
plt.plot(rang,l)
plt.show()
sns.factorplot(x="cuisine",y="nb",data=train[['cuisine','nb']],kind='bar',size=8,aspect=2.5)
vect = TfidfVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
vect.fit(dataset['ingredients'].apply(','.join))

Xtf_train = vect.transform(train['ingredients'].apply(','.join)).toarray()
Xtf_train = pd.DataFrame(Xtf_train,columns=vect.get_feature_names())

Xtf_test = vect.transform(test['ingredients'].apply(','.join)).toarray()
Xtf_test = pd.DataFrame(Xtf_test,columns=vect.get_feature_names())

print(Xtf_train.shape,Xtf_test.shape)

lb = LabelEncoder()
target = train['cuisine']
y_train = lb.fit_transform(target)
print(y_train.shape)
Xtf_train.head()
svc = LinearSVC()
print(cross_val_score(svc,Xtf_train,y_train,cv=3))
svc.fit(Xtf_train,y_train)
y_predict = svc.predict(Xtf_test)
y_string = lb.inverse_transform(y_predict)
my_submission = pd.DataFrame({'id':test['id'],'cuisine':y_string})
my_submission.to_csv('submission.csv', index=False)
