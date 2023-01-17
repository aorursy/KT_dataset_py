# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/dont-overfit-ii/train.csv")

test = pd.read_csv("../input/dont-overfit-ii/test.csv")

submission = pd.read_csv("../input/dont-overfit-ii/sample_submission.csv")
train
train['target'].value_counts().sort_values().plot(kind = 'barh')
test
train.dtypes[train.dtypes == object]
train.isnull().sum()[train.isnull().sum() != 0]
test.dtypes[test.dtypes == object]
test.isnull().sum()[test.isnull().sum() != 0]
train.drop(['id','target'],axis=1).shape
test.drop(['id'],axis=1).shape
from sklearn.cluster import AffinityPropagation

model = AffinityPropagation(damping=0.9)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# assign a cluster to each example

yhat = model.predict(test.drop(['id'],axis=1))

yhat
df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
from sklearn.cluster import AffinityPropagation

model = AffinityPropagation(damping=0.5)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# assign a cluster to each example

yhat = model.predict(test.drop(['id'],axis=1))

df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1).iloc[:len(train)*8//10])

# fit model and predict clusters

y_val = model.fit_predict(train.drop(['id','target'],axis=1).iloc[len(train)*8//10:])



abs( sum(y_val) - sum(train['target'].iloc[len(train)*8//10:]) )/sum(train['target'].iloc[len(train)*8//10:])
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# fit model and predict clusters

yhat = model.fit_predict(test.drop(['id'],axis=1))

df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
submission['target'] = yhat

submission.to_csv('submission_AgglomerativeClustering.csv',index=False)
from sklearn.cluster import Birch

model = Birch(threshold=0.01, n_clusters=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1).iloc[:len(train)*8//10])

# fit model and predict clusters

y_val = model.predict(train.drop(['id','target'],axis=1).iloc[len(train)*8//10:])



abs(sum(y_val) - sum(train['target'].iloc[len(train)*8//10:]) )/sum(train['target'].iloc[len(train)*8//10:])
from sklearn.cluster import Birch

model = Birch(threshold=0.01, n_clusters=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# assign a cluster to each example

yhat = model.predict(test.drop(['id'],axis=1))

df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
submission['target'] = yhat

submission.to_csv('submission_Birch.csv',index=False)
from sklearn.cluster import DBSCAN

# define the model

model = DBSCAN(eps=0.30, min_samples=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1).iloc[:len(train)*8//10])

# fit model and predict clusters

y_val = model.fit_predict(train.drop(['id','target'],axis=1).iloc[len(train)*8//10:])



abs(sum(y_val) - sum(train['target'].iloc[len(train)*8//10:]) )/sum(train['target'].iloc[len(train)*8//10:])
from sklearn.cluster import DBSCAN

# define the model

model = DBSCAN(eps=0.30, min_samples=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# fit model and predict clusters

yhat = model.fit_predict(test.drop(['id'],axis=1))

df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
submission['target'] = yhat

submission.to_csv('submission_DBSCAN.csv',index=False)
from sklearn.cluster import KMeans

# define the model

model = KMeans(n_clusters=2)



# fit the model

model.fit(train.drop(['id','target'],axis=1).iloc[:len(train)*8//10])

# fit model and predict clusters

y_val = model.predict(train.drop(['id','target'],axis=1).iloc[len(train)*8//10:])



abs(sum(y_val) - sum(train['target'].iloc[len(train)*8//10:]) )/sum(train['target'].iloc[len(train)*8//10:])
from sklearn.cluster import KMeans

# define the model

model = KMeans(n_clusters=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# fit model and predict clusters

yhat = model.predict(test.drop(['id'],axis=1))

df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
submission['target'] = yhat

submission.to_csv('submission_KMeans.csv',index=False)
from sklearn.cluster import MiniBatchKMeans

# define the model

model = MiniBatchKMeans(n_clusters=2)



# fit the model

model.fit(train.drop(['id','target'],axis=1).iloc[:len(train)*8//10])

# fit model and predict clusters

y_val = model.predict(train.drop(['id','target'],axis=1).iloc[len(train)*8//10:])



abs(sum(y_val) - sum(train['target'].iloc[len(train)*8//10:]) )/sum(train['target'].iloc[len(train)*8//10:])
from sklearn.cluster import MiniBatchKMeans

# define the model

model = MiniBatchKMeans(n_clusters=2)

# fit the model

model.fit(train.drop(['id','target'],axis=1))

# fit model and predict clusters

yhat = model.predict(test.drop(['id'],axis=1))

df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
submission['target'] = yhat

submission.to_csv('submission_MiniBatchKMeans.csv',index=False)
submission