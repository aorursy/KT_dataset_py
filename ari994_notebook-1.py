# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree



train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')
#

#train_data.head()

#import matplotlib.pyplot as plt

#train_data['color']=train_data['color'].apply(lambda x : x.replace('clear','black'))

#train_data['color']=train_data['color'].apply(lambda x : x.replace('blood','red'))

#plt.scatter(train_data['bone_length'], train_data['rotting_flesh'], s=train_data['hair_length']*100\

#            , c=train_data['color'], alpha=0.5)

#plt.show()
colors=set(train_data['color'])

colors_dict={}

i=0

for c in colors:

    colors_dict[c]=i

    i+=1

types=set(train_data['type'])

types_dict={}

i=0

for c in types:

    types_dict[c]=i

    i+=1

print (colors_dict)

print (types_dict)



train_data['color']=train_data['color'].apply(lambda x : colors_dict[x])

test_data['color']=test_data['color'].apply(lambda x : colors_dict[x])

train_data['type']=train_data['type'].apply(lambda x : types_dict[x])
del train_data['color']

del test_data['color']

train_data.count()
from sklearn.model_selection import train_test_split

from sklearn import cross_validation

import matplotlib.pyplot as plt



plt.scatter(train_data['has_soul'], np.log(train_data['hair_length']/train_data['rotting_flesh']), s=train_data['rotting_flesh']*100\

            , c=train_data['type'], alpha=0.5)





#train_data['f_x1']=np.log(train_data['hair_length']/train_data['rotting_flesh'])

#test_data['f_x1']=np.log(test_data['hair_length']/test_data['rotting_flesh'])
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier

rev_type=['Ghoul','Goblin','Ghost']

Y=train_data['type']

X=train_data.ix[:,0:-1]

#X=train_data[['f_x']]

X.head()



clf =AdaBoostClassifier()

res=cross_validation.cross_val_score(clf, X, Y, scoring='f1_macro',cv=10)

print ('Mean    :   ',res.mean())

print ('Max     :   ',res.max())

print ('Min     :   ',res.min())
clf.fit(X,Y)

preds=clf.predict(test_data)

se = pd.Series(preds).apply(lambda x:rev_type[x])

results=test_data[['id']]

results['type']=se.values

results.head()
print (results.groupby('type').count())

results.to_csv('submisson.csv', index=False)