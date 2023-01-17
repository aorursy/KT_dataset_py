# Pandas pour manipuler les tableaux de données

import pandas as pd

import numpy as np

pd.set_option('display.max_columns', 500)



# scikit learn pour les outils de machine learning

import sklearn

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



# librairies pour la visualisation de données

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl



# et quelques options visuelles

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

sns.set(style="whitegrid", color_codes=True)

sns.set(rc={'figure.figsize':(15,8)})
data=pd.read_csv('../input/titanicdata2/dataForML.csv',index_col=0)

data.head(2)
data=pd.concat([data, pd.get_dummies(data["Pclass"], prefix='_Pclass_')], axis=1)

data=pd.concat([data, pd.get_dummies(data["Title"], prefix='_Title_')], axis=1)

data=pd.concat([data, pd.get_dummies(data["Embarked"], prefix='_Embarked_')], axis=1)

data=pd.concat([data, pd.get_dummies(data["Ticket"], prefix='_Ticket_')], axis=1)

data=pd.concat([data, pd.get_dummies(data["LastNameNum"], prefix='_LastNameNum_')], axis=1)

data = data.drop(['AgeBin','FareBin','Title','Embarked','LastNameNum','Ticket'],axis=1)



y = data.loc['train','Survived'].astype('int')

X = data.loc['train'].drop(['Survived','PassengerId'],axis=1)

X_cible =  data.loc['test'].drop(['Survived','PassengerId'],axis=1)
# après quelques tests je me limite à supprimer 'isEnfant'

X=X.drop(['isEnfant'], axis=1)

X_cible=X_cible.drop(['isEnfant'], axis=1)
# Modules KERAS

import keras

from keras.layers import Dense, Activation, Dropout

from keras.models import Sequential

from numpy.random import seed

from tensorflow import set_random_seed



# définition de n_cols : la dimension des données d'entrée

n_cols = X.shape[1]



# définition du seed (graine des fonctions d'aléas) afin de pouvoir reproduire à l'identique les résultats des calculs

seed(42)

set_random_seed(42)



# Architecture du modèle, un ecouche d'entrée, une de sortie et deux couches "profondes"

# cette architecture est un peu empirique, il n'y a pas vraiment de rêgles permettant de définir

# la meilleur architecture

nb_neurones=47

model = Sequential()

model.add(Dense(nb_neurones, activation='linear', input_shape = (n_cols,)))

model.add(Dense(nb_neurones, activation='linear')) 

model.add(Dense(nb_neurones, activation='linear')) 



# option dropout, cette option évite le surajustement (overfitting) en injectant des aléas sur les données

# d'entrée (une partie des données est aléatoirement fixée à zero)

model.add(Dropout(0.1))

    

# couche de sortie

model.add(Dense(1, activation='sigmoid'))  # output layer



# compilation du modèle

model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])



# exécution du modèle

history = model.fit(X,y,nb_epoch=200,batch_size=512, validation_split=0.2, verbose=0)
plt.plot(history.history['acc'],'red',label='précision sur le jeu d\'entrainement')

plt.plot(history.history['val_acc'],'blue',label='précision sur le jeu de validation')

plt.legend()
model.fit(X,y,nb_epoch=400,verbose=0,batch_size=512)
y_cible = model.predict(X_cible).reshape(-1,)

y_cible = np.rint(y_cible)
y_cible.sum()
sample=pd.DataFrame()

sample['PassengerId'] = data.loc['test','PassengerId']

sample['Survived'] = y_cible

sample.Survived=sample.Survived.astype(int)

sample=sample.reset_index(drop=True)

sample.to_csv('submission.csv', index=False)