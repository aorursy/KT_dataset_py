#On import des librairies pour manipuler les données 

import numpy as np # librairie utilisée généralement pour de l'algèbre linéaire

import pandas as pd # pour le traitement de données sous forme de tableaux (e.g. pd.read_csv)



# Les données sont disponible dans  "../input/" directory.

# Par exemple,rouler cette cellule de code (Shift+Enter) va lister tous les fichiers sous le dossiers 

# input



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Tous les résultats que vous allez générer seront sous le dossiers output
train_data=pd.read_csv("/kaggle/input/semaine-ia-kaggle-intro/data_train.csv") #Importer les données d'entrainement

train_data.head() 
test_data=pd.read_csv("/kaggle/input/semaine-ia-kaggle-intro/data_test.csv")

test_data.head() #Afficher les première lignes des données
women=train_data.loc[train_data.Sex == 'female']["Survived"] 

#.loc : localiser dans les données d'entrainement: parmis les femmes, celles qui on survécues

rate_women=sum(women)/len(women)

print("% of women who survived:", rate_women)
men=train_data.loc[train_data.Sex == 'male']["Survived"] #Même chose pour les hommes 

rate_men=sum(men)/len(men)

print("% of women who survived:", rate_men)
features=["Pclass","Sex","SibSp","Parch"]

X=pd.get_dummies(train_data[features])

X.head()

train_data.head()
from sklearn.ensemble import RandomForestClassifier 

#Librairie scikit-learn permet de créer rapidement des modèles de classification 

y=train_data["Survived"]

features=["Pclass","Sex","SibSp","Parch"]



X=pd.get_dummies(train_data[features])

# Pour transformer les données catégoriques en données numériques. 

#Exemple: https://stackoverflow.com/questions/48170405/is-pd-get-dummies-one-hot-encoding

X_train=train_data[features]

X_test=pd.get_dummies(test_data[features])



model=RandomForestClassifier(n_estimators=200,max_depth=9,random_state=1)

model.fit(X, y) #On entraine le modèle

predictions=model.predict(X_test) #On prédit la survie des passagers dans les données test



output=pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': predictions})

# On strucuture le résultats sous la forme voulue par la soumission dans la compétition

output.to_csv('my_submissionFab2.csv',index=False)

#On sauvegarde les résultats sous format csv

print("Your Submission was succesfully saved !")