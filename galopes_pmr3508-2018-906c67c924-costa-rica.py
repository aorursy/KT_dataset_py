import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
pd.set_option('display.max_columns', 500)
import os
print(os.listdir('../input/'))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.shape
n_more_train = train.fillna(value=0)
#ntrain = pd.concat([ntrain, pd.get_dummies(ntrain['edjefe'], prefix='edjefe')], axis=1)
ntrain = n_more_train.drop(['edjefe', 'edjefa', 'dependency', 'idhogar'], axis=1)
ntrain.set_index('Id', inplace=True)
ntrain.head()
knn = KNeighborsClassifier(n_neighbors=13)
xtrain = ntrain.drop('Target', axis=1)
ytrain = ntrain.Target
score = cross_val_score(knn, xtrain, ytrain, cv=10)
score.mean()
knn.fit(xtrain, ytrain)
newtest = test.drop(['edjefe', 'edjefa', 'dependency', 'idhogar'], axis=1)
newtest.set_index('Id', inplace=True)
ntest = newtest.fillna(value=0)
predictions = knn.predict(ntest)
results = np.vstack((test['Id'], predictions)).T
x = ['Id','Target']
result = pd.DataFrame(columns=x, data=results)
result.set_index('Id', inplace=True)
predicted = pd.DataFrame(data=predictions)
predicted[0].value_counts().sort_index().plot(kind='bar')
plt.show()
result.to_csv('myresults.csv')