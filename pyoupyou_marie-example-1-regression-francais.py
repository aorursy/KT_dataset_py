import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import numpy as np

import graphviz

import math

from IPython.display import Image

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn import metrics



df = pd.read_csv('../input/data.csv')



#A quoi ressemble les colonnes de notre fichier de data?

df.sample(10)
#Generons un tableau de correlations pour la valeur count.

corr_matrix=df.corr()

print(corr_matrix['count'].sort_values(ascending=False))
#Extraire les heures: Datetime: 01-01-2011 23:00:00 -> 23

df['datetime'] = pd.to_datetime(df['datetime'])

#Extraire d'autres donnees de temps.

df['hour']=df['datetime'].dt.hour

df['month']=df['datetime'].dt.month

df['year']=df['datetime'].dt.year



#Que sait-on sur la valeur count?

print(df['count'].describe())
#Generation d'un plot de la variable "count"

df['count'].hist(bins=4, figsize=(5,4))

plt.show()



nb_velos=0

for x in range(10):

    nb_velo=Counter(df['count'])[x]

    nb_velos+=nb_velo

print('Dans',nb_velos/11000*100,'% des cas il y a moins de 10 velos emprunt´es ')

var_meteo = ["humidity", "temp","weather"]

fig, axes = plt.subplots(nrows=1, ncols=3)

i=0

for attribute in var_meteo:

    data = pd.concat([df['count'],df[attribute]], axis=1)

    data.plot.scatter( y="count",x=attribute, figsize=(15,5),ax=axes[i]);

    i+=1
var_chrono = ["hour","month","season","workingday", "holiday","Unnamed: 0"]

fig, axes = plt.subplots(nrows=2, ncols=3)

row=0

i=0

j=0



for attribute in var_chrono:

    data = pd.concat([df['count'],df[attribute]], axis=1)

    data.plot.scatter( y="count",x=attribute, figsize=(15,8),ax=axes[int(row/3),j], )

    i+=1

    j=[0,1,2,0,1,2,0][i]

    row+=1



#df.plot(x='month',y='count')

#plt.axhline(df['count'].mean(),color='r')

#df['count'].plot(kind='bar')
#Gardons 70% du corpus pour le training set

nb_split=int(len(df)*0.7)

df_train=df[:nb_split]

df_val_test=df[nb_split:]

x_train=df_train.drop('count',axis=1)

y_train=df_train['count']



#Les 30% restants serviront comme validation et test set. Le validation set est deux fois plus grand que le test set.

x_df_val_test=df_val_test.drop('count',axis=1)

y_df_val_test=df_val_test['count']

x_val,x_test,y_val,y_test=train_test_split(x_df_val_test,y_df_val_test,

train_size=0.66,

test_size=0.34,

random_state=123)
def evaluate(y_true,y_pred):

    """prend en entree les instances vraies et les instances predites

    retourne le score 

    >>>evaluate([3,4],[2,5])

    1.0

    """

    

    perf=metrics.mean_squared_error(y_true, y_pred)

    perf=math.sqrt(perf)

    print(perf)

    return perf

evaluate([3,4],[2,5])
#Utilisons le dummy classifier pour notre baseline

dummy = DummyRegressor(strategy='mean', constant = None) 

dummy.fit(x_train, y_train) 

y_pred=dummy.predict(x_val)

y_true=(y_val)

evaluate(y_true,y_pred)
#Preparation du modele

def try_feat(variables):

    clf = RandomForestRegressor(n_estimators=50, max_depth=15,

                           random_state=42)

    clf.fit(x_train[variables], y_train)

    y_pred=clf.predict(x_val[variables])

    y_true=(y_val)

    print("En utilisant les variables",variables, "notre modele performe:")

    print("score:",evaluate(y_true,y_pred))



#XP 1

variables=['hour',"workingday"]

try_feat(variables)



#XP 2

variables=['hour',"workingday","month","holiday","weather"]

try_feat(variables)



#XP 3

variables=['hour',"workingday","month","holiday","year"]

try_feat(variables)



#XP 4

variables=['hour',"workingday","month","holiday", "weather","year"]

try_feat(variables)
y_true=y_test

y_pred=dummy.predict(x_test)

print("Sur le test set notre baseline est de:")

evaluate(y_pred,y_true)

clf = RandomForestRegressor(n_estimators=50, max_depth=15,

                           random_state=42)

clf.fit(x_train[variables], y_train)

clf.predict(x_test[variables])

y_pred=clf.predict(x_test[variables])

print("Sur le test set notre systeme performe:")

evaluate(y_pred,y_true)