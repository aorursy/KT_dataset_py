# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin_min

 

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# dfm = pd.read_csv('movies_metadata.csv')

db_batting = pd.read_csv('/kaggle/input/the-history-of-baseball/batting.csv') #Base de datos de bateos

db_batting.rename(columns = {'hr' : 'homeruns','g' : 'games','ab' : 'at_bats','r' : 'runs','h' : 'hits', #Cambiar el nombre de columnas a nombres mas descriptivos 

                             'bb' : 'base_on_balls','sb':'stolen_bases','so' : 'strikeouts'}, inplace = True)

db_players = pd.read_csv('/kaggle/input/the-history-of-baseball/player.csv') #Base de datos de jugadores



#Eliminar columnas que no eran útiles

db_batting.drop(["ibb","hbp","sh","sf","g_idp","cs","rbi","league_id"], axis=1, inplace=True) 

db_players.drop(["birth_state","birth_city","death_year","death_month","death_day","death_country","death_state","death_city",

                "debut","final_game","retro_id","bbref_id","birth_year","birth_month","birth_day","throws","name_first","name_last"], axis=1, inplace=True)



#Mezclar dataset de jugadores con sus estadísticas de bateo

db_batters = pd.merge(left=db_players, right=db_batting, on='player_id',how='inner')

# db_batting.head(2)

# db_players.head(2)







#dataFrame_csv = db_batters.to_csv (r'C:\Users\razie\Documents\Tec\Invierno2020\Métodos cuantitativos y simulacion\dataFrame.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



db_batters.dropna(inplace=True)



cleanDataFrame_csv = db_batters.to_csv (r'C:\Users\razie\Documents\Tec\Invierno2020\Métodos cuantitativos y simulacion\cleanDataFrame.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



#Obtener media de variables

# media_weight = db_batters['weight'].mean()

# media_height = db_batters['height'].mean()

# media_strikeouts = db_batters['strikeouts'].mean()

# media_bases = db_batters['stolen_bases'].mean()

# media_games = db_batters['games'].mean()

# print('\n Media de cada variable: \n Weight: \n', media_weight, '\n Height: \n', media_height, '\n Strikeouts: \n', media_strikeouts, 

#       '\n Bases Robadas: \n', media_bases, '\n Cantidad de juegos: \n', media_games )



#Obtener varianza de variables

# variance_weight = db_batters.loc[:,"weight"].var()

# variance_height = db_batters.loc[:,"height"].var()

# variance_strikeouts = db_batters.loc[:,"strikeouts"].var()

# variance_bases = db_batters.loc[:,"stolen_bases"].var()

# variance_games = db_batters.loc[:,"games"].var()

# print('\n Varianza de cada variable: \n Weight: \n', variance_weight, '\n Height: \n', variance_height, '\n Strikeouts: \n', variance_strikeouts, 

#       '\n Bases Robadas: \n', variance_bases, '\n Cantidad de juegos: \n', variance_games )





#Obtener desviacion estandar de variables

# print('\n Desviacion estandar de cada variable: \n Weight: \n', variance_weight**.5, '\n Height: \n', variance_height**.5, '\n Strikeouts: \n', variance_strikeouts**.5, 

#       '\n Bases Robadas: \n', variance_bases**.5, '\n Cantidad de juegos: \n', variance_games**.5 )





# Histogramas

# hist_weight = db_batters['homeruns'].hist(bins=20);

# hist_weight.set_title('Homeruns')

# hist_weight.set_xlabel('Homeruns value')

# hist_weight.set_ylabel('Frequency')



#print(db_batters.groupby('homeruns').size())



#Se define la entrada

X = np.array(db_batters[["height","strikeouts","homeruns"]])

y = np.array(db_batters['weight'])

#Graficar en 3D con 9 colores representando las categorías

# fig = plt.figure()

# ax = Axes3D(fig)

# colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']

# asignar=[]

# # for row in y:

# #     asignar.append(colores[row])

# ax.scatter(X[:, 0], X[:, 1], X[:, 2],s=60)



# Nc = range(1, 20)

# kmeans = [KMeans(n_clusters=i) for i in Nc]

# kmeans

# score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

# score

# plt.plot(Nc,score)

# plt.xlabel('Number of Clusters')

# plt.ylabel('Score')

# plt.title('Elbow Curve')

# plt.show()



kmeans = KMeans(n_clusters=5).fit(X)

centroids = kmeans.cluster_centers_

print(centroids)



# Predicting the clusters

labels = kmeans.predict(X)

# Getting the cluster centers

C = kmeans.cluster_centers_

colores=['red','green','blue','cyan','yellow']

asignar=[]

for row in labels:

    asignar.append(colores[row])

 

fig = plt.figure()

# ax = Axes3D(fig)

# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)

# ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)



#Gráficas en 2 dimensiones con las proyecciones a partir de nuestra gráfica 3D para que nos ayude a visualizar los grupos y su clasificación

f1 = db_batters['strikeouts'].values

f2 = db_batters['homeruns'].values

plt.scatter(f1, f2, c=asignar, s=70)

plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)

plt.xlabel("Strikeouts")

plt.ylabel("Homeruns")

plt.show()







# nuevo = db_batters[["weight","strikeouts","stolen_bases","games"]]

# nuevo = db_batters[["weight", "height"]]

#sb.pairplot(db_batters, hue='weight', size=7,vars=["height","strikeouts","homeruns"])

#db_batters.head(50)

#db_batters.isnull().sum()

#db_batters.shape

#print (db_batters)

# pd.merge(left=db_batting, right=players, on="player_id", how="inner")

# Any results you write to the current directory are saved as output.