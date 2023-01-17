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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

from imblearn.over_sampling import KMeansSMOTE

warnings.filterwarnings('ignore')
df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df.head()
df=df[df['TotalCharges'] !=" "]

df['TotalCharges']=pd.to_numeric(df['TotalCharges'])
print(df.gender.unique())

#df.SeniorCitizen.unique()

print(df.Married.unique())

print(df.Children.unique())

print(df.TVConnection.unique())

print(df.Channel1.unique())

print(df.Channel2.unique())

print(df.Channel3.unique())

print(df.Channel4.unique())

print(df.Channel5.unique())

print(df.Channel6.unique())

print(df.Internet.unique())

print(df.HighSpeed.unique())

print(df.AddedServices.unique())

print(df.PaymentMethod.unique())

print(df.Subscription.unique())
Y_train=df[df.columns[20]]

Y_train.shape
df.head()
df.shape
dfDummies = pd.get_dummies(df['PaymentMethod'],prefix='Paying_method')

df = pd.concat([df, dfDummies], axis=1)

df=df.drop('PaymentMethod',axis=1)

df.head()
dfDummies = pd.get_dummies(df['Subscription'],prefix='sub')

df = pd.concat([df, dfDummies], axis=1)

df=df.drop('Subscription',axis=1)

df.head()
dfDummies = pd.get_dummies(df['TVConnection'],prefix='t')

df = pd.concat([df, dfDummies], axis=1)

df=df.drop('TVConnection',axis=1)

df.head()
df=df.replace(to_replace="Male",value=0)

df=df.replace(to_replace="Female",value=1)

df=df.replace(to_replace="Yes",value=1)

df=df.replace(to_replace="No",value=0)

df=df.replace(to_replace="No tv connection",value=0)

df=df.replace(to_replace="No internet",value=0)
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error



corr = df.corr()

print(corr['Satisfied'])

sns.heatmap(corr)
cor_relation_check = abs(corr["Satisfied"])

#Selecting highly correlated features

Non_important = cor_relation_check[cor_relation_check<0.15]

Non_important.index
#df=df.drop(['custId','gender','Channel1','Channel2','Internet','HighSpeed','Satisfied'], axis=1)
df=df.drop(['custId', 'gender', 'SeniorCitizen', 'Channel1', 'Channel2', 'Channel3',

       'Channel4', 'Internet', 'HighSpeed', 'Paying_method_Bank transfer',

       'Paying_method_Cash', 'Paying_method_Credit card', 't_DTH','Satisfied'],axis=1)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

X_std=df

#X_std=pd.get_dummies(X_std)

X_std = MinMaxScaler().fit_transform(X_std)

from imblearn.under_sampling import AllKNN

allknn = AllKNN()

X_std, Y_train = allknn.fit_resample(X_std,Y_train)
X_std.shape
from sklearn.cluster import KMeans,SpectralClustering,Birch

#cluster=KMeans(n_clusters=2,init='k-means++',n_init=200,max_iter=700).fit(PCA_components)

cluster = Birch(n_clusters=2).fit(X_std)
df260=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df260.head()
Labels=df260[df260.columns[0]]

print(Labels)
df260.loc[df260['TotalCharges'] ==" ",'TotalCharges']=df260['MonthlyCharges']

df260['TotalCharges']=pd.to_numeric(df260['TotalCharges'])
df260.shape
dfDummies = pd.get_dummies(df260['PaymentMethod'],prefix='Paying_method')

df260 = pd.concat([df260, dfDummies], axis=1)

df260=df260.drop('PaymentMethod',axis=1)

df260.head()
dfDummies = pd.get_dummies(df260['Subscription'],prefix='sub')

df260 = pd.concat([df260, dfDummies], axis=1)

df260=df260.drop('Subscription',axis=1)

df260.head()

dfDummies = pd.get_dummies(df260['TVConnection'],prefix='t')

df260 = pd.concat([df260, dfDummies], axis=1)

df260=df260.drop('TVConnection',axis=1)

df260.head()
df260=df260.replace(to_replace="Male",value=0)

df260=df260.replace(to_replace="Female",value=1)

df260=df260.replace(to_replace="Yes",value=1)

df260=df260.replace(to_replace="No",value=0)

df260=df260.replace(to_replace="No tv connection",value=0)

df260=df260.replace(to_replace="No internet",value=0)
#df260=df260.drop(['custId','gender','Channel1','Channel2','Internet','HighSpeed'], axis=1)
df260=df260.drop(['custId', 'gender', 'SeniorCitizen', 'Channel1', 'Channel2', 'Channel3',

       'Channel4', 'Internet', 'HighSpeed', 'Paying_method_Bank transfer',

       'Paying_method_Cash', 'Paying_method_Credit card', 't_DTH'],axis=1)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

X_test=df260

#X_std=pd.get_dummies(X_std)

X_test = MinMaxScaler().fit_transform(X_test)
cluster = Birch(n_clusters=2).fit(X_test)

answering = cluster.predict(X_test)

df3=pd.DataFrame({"custId":Labels,"Satisfied":answering})

df3.to_csv('a5.csv',index=False)
np.set_printoptions(threshold=np.inf)

print(answering)
l1=[]

for i in range(len(answering)):

    if answering[i]==0:

        l1.append(1)

    else:

        l1.append(0)
df9=pd.DataFrame({"custId":Labels,"Satisfied":l1})

df9.to_csv('x20.csv',index=False)