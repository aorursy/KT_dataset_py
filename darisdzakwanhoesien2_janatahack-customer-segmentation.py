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
submission = pd.read_csv("../input/analytics-vidhya-janatahack-customer-segmentation/sample_submission_wyi0h0z.csv")

train = pd.read_csv("../input/analytics-vidhya-janatahack-customer-segmentation/Train_aBjfeNk.csv")

test = pd.read_csv("../input/analytics-vidhya-janatahack-customer-segmentation/Test_LqhgPWU.csv")
train
y_train = train['Segmentation']

y_train_set = train[['Segmentation']]

y_train.value_counts().plot(kind='barh')
y_train_set['Segmentation'] = y_hist

y_train_set
from sklearn.preprocessing import LabelEncoder

y_hist = LabelEncoder().fit_transform(y_train)

df = pd.DataFrame(data=y_hist, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
train = train.drop(['Segmentation'],axis=1)
test
dataset = pd.concat([train,test]).reset_index(drop=True).drop(['ID'],axis=1)

dataset
object_column = dataset.dtypes[dataset.dtypes == object].index

numerical_column = dataset.dtypes[dataset.dtypes != object].index
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,3,figsize=(12,8))

for i in range(3):

    dataset[object_column[i]].value_counts().plot(kind='barh', ax=ax[0,i])

    ax[0,i].title.set_text(object_column[i])

for i in range(3):

    dataset[object_column[3+i]].value_counts().plot(kind='barh', ax=ax[1,i])

    ax[1,i].title.set_text(object_column[3+i])
fig, ax = plt.subplots(2,2,figsize=(8,8))



dataset[numerical_column[0]].plot(kind='hist', ax=ax[0,0])

ax[0,0].title.set_text(numerical_column[0])



dataset[numerical_column[1]].plot(kind='hist', ax=ax[0,1])

ax[0,1].title.set_text(numerical_column[1])



dataset[numerical_column[2]].plot(kind='hist', ax=ax[1,0])

ax[1,0].title.set_text(numerical_column[2])
empty_object = list(object_column)[1:4] + [list(object_column)[-1]]

empty_object
dataset.isnull().sum()
dataset.isnull().sum()[dataset.isnull().sum() != 0].loc[empty_object]
dataset[numerical_column].mean()
map_empty = {}



dataset.isnull().sum()[dataset.isnull().sum() != 0].loc[empty_object].index

for key in dataset.isnull().sum()[dataset.isnull().sum() != 0].loc[empty_object].index:

    map_empty[key] = 'etc'



for key,value in dataset[numerical_column].mean().items():

    map_empty[key] = round(value,2)

    

map_empty
dataset = dataset.fillna(value=map_empty)

dataset
from sklearn.preprocessing import LabelEncoder

for column in object_column:

    dataset[column] = LabelEncoder().fit_transform(dataset[column])

dataset
dataset.isnull().sum()
dataset.isnull().sum()[dataset.isnull().sum() != 0]
dataset
dataset.iloc[:len(train)*8//10]
dataset.iloc[len(train)*8//10:len(train)]
y_train_set.iloc[:len(train)*8//10]
y_train_set.iloc[len(train)*8//10:]
x_train = dataset.iloc[:len(train)*8//10]

x_val = dataset.iloc[len(train)*8//10:len(train)]

x_test = dataset.iloc[len(train):]



y_train = y_train_set.iloc[:len(train)*8//10]

y_val = y_train_set.iloc[len(train)*8//10:]



import time

from sklearn.cluster import AgglomerativeClustering

ts = time.time()



model = AgglomerativeClustering(n_clusters=4)

yhat1 = model.fit_predict(dataset.iloc[:len(train)])

yhat1
from sklearn.cluster import MiniBatchKMeans

# define the model

model = MiniBatchKMeans(n_clusters=4)

# fit the model

model.fit(dataset.iloc[:len(train)])

# assign a cluster to each example

yhat2 = model.predict(dataset.iloc[:len(train)])

yhat2
from sklearn.cluster import KMeans

# define the model

model = KMeans(n_clusters=4)

# fit the model

model.fit(dataset.iloc[:len(train)])

# assign a cluster to each example

yhat3 = model.predict(dataset.iloc[:len(train)])

yhat3
from sklearn.cluster import Birch

# define the model

model = Birch(threshold=0.01, n_clusters=4)

# fit the model

model.fit(dataset.iloc[:len(train)])

# assign a cluster to each example

yhat4 = model.predict(dataset.iloc[:len(train)])

yhat4
def plot_dist(yhat):   

    df = pd.DataFrame(data=yhat, columns=["column1"])

    df['column1'].value_counts().sort_values().plot(kind = 'barh')



from sklearn.cluster import AffinityPropagation

# define the model

model = AffinityPropagation(damping=0.5)

# fit the model

model.fit(dataset.iloc[:len(train)])

# assign a cluster to each example

yhat5 = model.predict(dataset.iloc[:len(train)])

plot_dist(yhat5)
from sklearn.cluster import OPTICS

# define the model

model = OPTICS(eps=0.8, min_samples=4)

# fit model and predict clusters

yhat6 = model.fit_predict(dataset.iloc[:len(train)])

eval_accuracy(y_hist, yhat6)
from sklearn.cluster import SpectralClustering

# define the model

model = SpectralClustering(n_clusters=4)

# fit model and predict clusters

yhat7 = model.fit_predict(dataset.iloc[:len(train)])

eval_accuracy(y_hist, yhat7)
from sklearn.mixture import GaussianMixture

# define the model

model = GaussianMixture(n_components=4)

# fit the model

model.fit(dataset.iloc[:len(train)])

# assign a cluster to each example

yhat8 = model.predict(dataset.iloc[:len(train)])

eval_accuracy(y_hist, yhat8)
def eval_accuracy(y_hist, yhat):

    agg_cluster = pd.DataFrame()

    agg_cluster['actual'] = y_hist

    agg_cluster['prediction'] = yhat

    agg_cluster['bool'] = (agg_cluster['actual'] == agg_cluster['prediction'])

    print(sum(agg_cluster['bool']/len(agg_cluster)))

    

    df = pd.DataFrame(data=yhat, columns=["column1"])

    df['column1'].value_counts().sort_values().plot(kind = 'barh')



    return agg_cluster

eval_accuracy(y_hist, yhat1)
eval_accuracy(y_hist, yhat2)
eval_accuracy(y_hist, yhat3)
eval_accuracy(y_hist, yhat4)
df = pd.DataFrame(data=yhat, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
df = pd.DataFrame(data=y_hist, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
Y_test_class = model.fit_predict(x_test)



df = pd.DataFrame(data=Y_test_class, columns=["column1"])

df['column1'].value_counts().sort_values().plot(kind = 'barh')
df
df['column1'] = df['column1'].apply(lambda x:dict_dummy[x])

df
submission['Segmentation'] = df['column1']

submission.to_csv('submission.csv',index=False)

submission