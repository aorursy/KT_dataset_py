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
df= pd.read_csv("/kaggle/input/cs-challenge/training_set.csv")
df_test=pd.read_csv("/kaggle/input/cs-challenge/test_set.csv")

df
df.isna().sum().sum() #nombre de NaN
(100*df.isna().sum()/432170).sort_values().tail(20)  #afficher les 20 premieres colonne avec les plus de nan et les proportion de nan
df=df.sort_values(['MAC_CODE','Date_time']) #mettre les valeurs ds l'ordre chronologique (hyp: continuité electrique dans le temps) pour chaque éolienne
df=df.fillna(method="ffill") #prend la valeur précédante valide

print(df.isna().sum().sum()) #on vérifie que ca remplie bien

df_test=df_test.sort_values(['MAC_CODE','Date_time']) #mettre les valeurs ds l'ordre chronologique (hyp: continuité electrique dans le temps) pour chaque éolienne
df_test=df_test.fillna(method="ffill") #prend la valeur précédante valide

print(df_test.isna().sum().sum()) #on vérifie que ca remplie bien
df = df.loc[:, df.nunique() != 1] # garde que les colonnes non constantes
len(df.columns)
aa=df.corr()

for i in aa:
    for j in range(len(aa[i])):
        aa[i][j]=abs(aa[i][j])

pas_regressible=['MAC_CODE','TARGET','Date_time','ID']

df_bis=df.drop(columns=pas_regressible)
cor=df_bis.corr()
for i in cor:
    for j in range(len(cor[i])):
        cor[i][j]=abs(cor[i][j])
aa.style.background_gradient(cmap='coolwarm')
import matplotlib.pyplot as plt
import seaborn as sns

#Using Pearson Correlation
plt.figure(figsize=(12,10))
sns.heatmap(cor)
plt.show()
#on fait la liste des trucs fortement correlés (>90%)
l=[]
for i in aa:
    for j in range(len(aa[i])):
        if aa[i][j]>0.9 and i!=aa.columns[j]: l+=[(i,aa.columns[j])]
len(l)/2

# on élimine les min et les max
l=[i for i in l if not ("min" in i[0] or "min" in i[1] or "max" in i[0] or "max" in i[1])]
l=[i for i in l if i[0]<i[1]]

l
#on fait la liste des trucs pas mal correlés (>70%)
l2=[]
for i in aa:
    for j in range(len(aa[i])):
        if aa[i][j]>0.7 and i!=aa.columns[j]: l2+=[(i,aa.columns[j])]
len(l2)/2

# on élimine les min et les max
l2=[i for i in l2 if not ("min" in i[0] or "min" in i[1] or "max" in i[0] or "max" in i[1])]
l2=[i for i in l2 if i[0]<i[1]]

len(l2)
nb_de_clusters=30

#k-means sur les données centrées et réduites
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters=nb_de_clusters)
kmeans.fit(cor)
#index triés des groupes
idk = np.argsort(kmeans.labels_)
#affichage des observations et leurs groupes
pd.DataFrame(df_bis.columns[idk],kmeans.labels_[idk]) #affichage des observations et leurs groupes
garder=[[0,0]]*nb_de_clusters
for i in idk:
    a=aa[df_bis.columns[i]][1]
    if garder[kmeans.labels_[i]][0]<a: garder[kmeans.labels_[i]]=[a,df_bis.columns[i]]

garder=pd.DataFrame(garder)
garder.sort_values(0)

cb_a_garder=30
garder=garder.tail(cb_a_garder)

garder=list(garder[1])

garder
a=["TARGET"]+[i for i in df.columns if not("min" in i or "max" in i or "std" in i or i in pas_regressible)]
df_bis=df[a]
for col in a:
    df_bis[col+'2'] = df[col]*df[col]

bb=df_bis.corr()

bb=abs(bb)

plt.figure(figsize=(12,10))
sns.heatmap(bb)
plt.show()
#Correlation with output variable
cor_target = aa["TARGET"]
cor_target2 = bb["TARGET"]
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features2=cor_target2[cor_target2>0.5]

for i in relevant_features.index[1:]:
    if not i in garder: garder+=[i]

len(garder)

garder2=[i for i in relevant_features2.index if '2'==i[-1]]
garder2=[i for i in garder2 if not 'TARGET' in i]
garder2
X = df[garder]

for i in garder2:
    a=i[:-1]
    X[i]=df[a]*df[a]

Y = df["TARGET"]

X
# on normalise les données
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

#on sépare jeu d'entrainement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=404)

len(X_train) + len(X_test) == len(X)
#entrainement
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)

#test
Y_pred = reg.predict(X_test)

hat = pd.DataFrame([Y_pred, Y_test]).T
hat.columns = ['Prediction' ,'Verité']

# analyse des prédictions
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test, Y_pred))

hat['erreur'] = abs(hat.iloc[:,0] - hat.iloc[:,1])
hat
#préparation des données
ids = df_test['ID']

for i in garder2:
    a=i[:-1]
    df_test[i]=df_test[a]*df_test[a]
df_test = df_test[garder+garder2]
    
X_real_world = StandardScaler().fit_transform(df_test)

#prédiction des résultats
prediction = reg.predict(X_real_world)

#Enregistrement des résultats prédits
results = pd.DataFrame()
results['ID'] = ids
results['TARGET'] = prediction
results.to_csv('linear_results.csv', index=False)

#affichage des résultats
results
