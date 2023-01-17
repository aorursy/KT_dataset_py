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
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling



from sklearn import cluster, datasets

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice

import time

np.random.seed(0)
df=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df.head()
missc=df.isnull().sum()

missc[missc>0]
y=df['Satisfied']
df.drop(['custId','Satisfied'],axis=1,inplace=True)
X=df.copy()
X.head()
X = pd.get_dummies(data=X,columns=['gender','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
X.head()
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'],errors='coerce')
X['TotalCharges']
X.fillna(value=X.mean(),inplace=True)

missc=X.isnull().sum()

missc[missc>0]
X.head()
from sklearn.metrics import roc_auc_score

temp=(X,y)

datasets=[(temp, {'n_neighbors': 2})]



default_base= {'n_clusters': 2}



algo_list=[]

for i_dataset,(dataset, algo_params) in enumerate(datasets):

    params = default_base.copy()

    params.update(algo_params)

    

    X,y=dataset

    

    X = StandardScaler().fit_transform(X)

    

    ward = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='ward')

    complete = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='complete')

    average = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='average')

    single = cluster.AgglomerativeClustering(

        n_clusters=params['n_clusters'], linkage='single')

    

    clustering_algorithms = (

        ('Single Linkage', single),

        ('Average Linkage', average),

        ('Complete Linkage', complete),

        ('Ward Linkage', ward),

    )

    

    for name, algorithm in clustering_algorithms:

        t0 = time.time()

        

        with warnings.catch_warnings():

            warnings.filterwarnings(

                "ignore",

                message="the number of connected components of the " +

                "connectivity matrix is [0-9]{1,2}" +

                " > 1. Completing it to avoid stopping the tree early.",

                category=UserWarning)

            algorithm.fit(X)

            

        t1 = time.time()

        if hasattr(algorithm, 'labels_'):

            y_pred=algorithm.labels_.astype(np.int)

        else:

            y_pred=algorithm.predict(X)

        algo_list.append(algorithm)

        rac=roc_auc_score(y_pred,y)

        print(algorithm,":",rac)
df2=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df2.head()
missc=df.isnull().sum()

missc[missc>0]
df2.drop(['custId'],axis=1,inplace=True)
X1=df2.copy()
X1 = pd.get_dummies(data=X1,columns=['gender','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
X1.head()
X1['TotalCharges'] = pd.to_numeric(X1['TotalCharges'],errors='coerce')
X1['TotalCharges']
X1.fillna(value=X1.mean(),inplace=True)

missc=X1.isnull().sum()

missc[missc>0]
X1.head()
algo_list
X1 = StandardScaler().fit_transform(X1)
y_pred1=algo_list[0].fit_predict(X1)
df3=pd.DataFrame()

df3['Satisfied']=y_pred1
df3['Satisfied'].value_counts()
y.value_counts()
df4=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df5=pd.DataFrame(index=df4['custId'])

df5['Satisfied']=y_pred1
df5.to_csv('S32.csv')
#for 2 best solution-
df6=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df6.head()
df6.drop(['custId','Satisfied'],axis=1,inplace=True)
XX=df6.copy()
XX.head()
XX = pd.get_dummies(data=XX,columns=['gender','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
XX.head()
XX['TotalCharges'] = pd.to_numeric(XX['TotalCharges'],errors='coerce')
XX.fillna(value=XX.mean(),inplace=True)

missc=XX.isnull().sum()

missc[missc>0]
XX.head()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=42)

kmeans.fit(XX)

y_kmeans11 = kmeans.predict(XX)

rac=roc_auc_score(y_kmeans11,y)

print(rac)
df7=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df7.head()
df7.drop(['custId'],axis=1,inplace=True)
X11=df7.copy()
X11 = pd.get_dummies(data=X11,columns=['gender','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
X11.head()
X11['TotalCharges'] = pd.to_numeric(X11['TotalCharges'],errors='coerce')
X11.fillna(value=X11.mean(),inplace=True)

missc=X11.isnull().sum()

missc[missc>0]
X11.head()
yy = kmeans.predict(X11)
df8=pd.DataFrame()

df8['Satisfied']=yy
df8['Satisfied'].value_counts()
y.value_counts()
df9=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df10=pd.DataFrame(index=df9['custId'])

df10['Satisfied']=yy
df10.to_csv('S31.csv')