# -*- coding: utf-8 -*-

"""

Created on Mon Mar 16 02:36:15 2020



@author: islem

"""

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv('../input/train.csv')

data.columns

print(data.shape)#taille de fichier

data.head()  #affichier les premiers lignes

print(data)#affichier le fchier

print(data.describe())# utilisée pour calculer certaines données statistiques comme le count mean.....

print(data['Age'])

data.dtypes#le nombre de type des variablesdata.columns#afficher les noms des colonnes de ce fichier

data



#Exploration des données:

sns.pairplot(data,hue='Survived')#la correlation entre les différentes caractéristiques,



#Survived 

sns.countplot(data.Survived)#count de Survived en graph

data['Survived'].value_counts()#count de Survived 

data.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)#en porcentage

#Sexe!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

sns.barplot(x='Sex', y='Survived', data=data)#plot SibSp Vs Survived

plt.show()

data.groupby(['Sex']).mean()#regroupe les données en fonction de la moyenne de 'sex'





sns.countplot(data.Sex)#plot Sex 

plt.show()

#Pclass

data.groupby(['Pclass']).mean()#regroupe les données en fonction de la moyenne de 'Pclass''

sns.barplot(x='Pclass', y='Survived', data=data)#plot SibSp Vs Survived

plt.show()

sns.factorplot('Pclass',data=data,hue='Survived',kind='count') #histogram de nombre de passager en fonction de pclass et sex

sns.barplot(x='Sex',y='Pclass',hue='Survived', data=data)#plot Pclass Vs survived

plt.show()



# le nombre passager en fonction  Pclass et Sex et Survived

Survived_Pcalss = sns.factorplot(x="Pclass", y="Survived", 

                                 hue="Sex", data=data,size=6, 

                                 kind="bar", palette="BuGn_r")

Survived_Pcalss.despine(left=True)

Survived_Pcalss = Survived_Pcalss.set_ylabels("survival probability")







#Age



#plot Survived en fonction l'Age

ax = sns.boxplot(x="Parch", y="Age", data=data)

ax = sns.stripplot(x="Survived", y="Age",data=data, jitter=True,edgecolor="gray")

sns.plt.title("Survival by Age",fontsize=12);

type(data)





data.groupby(['Age']).mean()#regroupe les données en fonction de la moyenne de 'Age''



bins = np.arange(0, 80, 5)#plot Survived Vs Age Vs Pclass

g = sns.FacetGrid(data, row='Survived', col='Pclass', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()





bins=[0,10,20,30,40,50,60,70,80]#plot Age

data['AgeBin']=pd.cut(data['Age'],bins)

sns.factorplot('AgeBin',data=data,hue='Survived',kind='count')





#Sibsp

data.SibSp.value_counts(normalize=True).plot(kind="bar",alpha=0.5)#en porcentage



 

sns.barplot(x='SibSp', y='Survived', data=data)#plot SibSp Vs Survived

plt.show()



#Parch*

data.Parch.value_counts(normalize=True).plot(kind="bar",alpha=0.5)#en porcentage



sns.barplot(x='Parch', y='Survived', data=data)#plot Parch Vs Survived

plt.show()





#Embarked

sns.barplot(x='Embarked',y='Survived', data=data)#plot Embarked Vs survived

plt.show()



data.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)#en porcentage







#Fare

grid = sns.FacetGrid(data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()

















#ajouter une colonne

for i in range(len(data)):

    if data.loc[i, "SibSp"] + data.loc[i, "Parch"] == 0:

        data.loc[i, "Alone"] = 1

    else:

        data.loc[i, "Alone"] = 0

sns.barplot('Alone', 'Survived', data=data, color="mediumturquoise")

plt.show()



data['Died']= 1-data['Survived']  # creer une nouvelle variable Died egale a 1 si survived =0 , et vice versa 

data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar')  # grouper par sex, et pour chaque sexe grouper par survived et par died 



data['familymembers']=data.SibSp+data.Parch

data.groupby('familymembers').agg('sum')[['Survived', 'Died']].plot(kind='bar')  





#missing data

data.isna()



print(data.isna().sum())





np.isnan(data['Survived']).sum()#le nombre des données manquantes dans la colonne survived

np.isnan(data['Survived']).sum()/data['Survived'].size #la moyenne des données manquantes dans la colonne survived

np.isnan(data['Pclass']).sum()/data['Pclass'].size

np.isnan(data['Age']).sum()/data['Age'].size

print(100*data.isna().sum()/(len(data)))

sns.heatmap(data.isnull(), cbar=False)



datanotna=data[data["Age"] .notna()]

print(datanotna)

dataisna=data[data["Age"].isna()]



sns.factorplot('Pclass',data=dataisna,hue='Sex',kind='count')#on peut pas eliminer sex et pclass 

sns.factorplot('Parch',data=dataisna,hue='Survived',kind='count')#on peut  eliminer  parch =3,4,5,6



datanotnaf=datanotna[datanotna.Sex == 'female']

datanotnam=datanotna[datanotna.Sex == 'male']

dataisnaf=dataisna[dataisna.Sex == 'female']

dataisnam=dataisna[dataisna.Sex == 'male']

print(datanotna)#714

print(dataisna)#177

print(datanotnaf)#261 femmes

print(datanotnam)#453 hommes

print(dataisnaf)#53 femmes

print(dataisnam)#124 hommes







#parch=1

print(dataisna[dataisna["Parch"]==1])#8 personne

print(datanotna[datanotna["Parch"]==1])#110 personnes

parch1=datanotna[datanotna["Parch"]==1]

x1=parch1['Age'].mean()

print(x1)

parchfinal1=dataisna[dataisna["Parch"]==1]

parchfinal1['Age']=x1

print(parchfinal1['Age'])





#parch=1

#femme:

print(dataisnaf[dataisnaf["Parch"]==1])#5

print(datanotnaf[datanotnaf["Parch"]==1])#55

parch1f=datanotnaf[datanotnaf["Parch"]==1]

x1f=parch1f['Age'].mean()

print(x1f)

parchfinal1f=dataisnaf[dataisnaf["Parch"]==1]

parchfinal1f['Age']=x1f

print(parchfinal1f['Age'])





#homme:

print(dataisnam[dataisnam["Parch"]==1])#3

print(datanotnam[datanotnam["Parch"]==1])#55

parch1m=datanotnam[datanotnam["Parch"]==1]

x1m=parch1m['Age'].mean()

print(x1m)

parchfinal1m=dataisnam[dataisnam["Parch"]==1]

parchfinal1m['Age']=x1m

print(parchfinal1m['Age'])









#parch=2

print(dataisna[dataisna["Parch"]==2])#12 personnes

print(datanotna[datanotna["Parch"]==2])#68 personnes

parch2=datanotna[datanotna["Parch"]==2]

x2=parch2['Age'].mean()

print(x2)

parchfinal2=dataisna[dataisna["Parch"]==2]

parchfinal2['Age']=x2

print(parchfinal2['Age'])

#parch=2

#femme:

print(dataisnaf[dataisnaf["Parch"]==2])#7personnes

print(datanotnaf[datanotnaf["Parch"]==2])#42personnes

parch2f=datanotnaf[datanotnaf["Parch"]==2]

x2f=parch2f['Age'].mean()

print(x2f)

parchfinal2f=dataisnaf[dataisnaf["Parch"]==2]

parchfinal2f['Age']=x2f

print(parchfinal2f['Age'])

#homme:

print(dataisnam[dataisnam["Parch"]==2])#5 personnes

print(datanotnam[datanotnam["Parch"]==2])#26 personnes

parch2m=datanotnam[datanotnam["Parch"]==2]

x2m=parch2m['Age'].mean()

print(x2m)

parchfinal2m=dataisnam[dataisnam["Parch"]==2]

parchfinal2m['Age']=x2m

print(parchfinal2m['Age'])



#parch=0

print(dataisna[dataisna["Parch"]==0])#157 personnes

print(datanotna[datanotna["Parch"]==0])#521 personnes

parch0=datanotna[datanotna["Parch"]==0]

x0=parch0['Age'].mean()

print(x0)

parchfinal0=dataisna[dataisna["Parch"]==0]

parchfinal0['Age']=x0

print(parchfinal0['Age'])



#parch=0

#femme:

print(dataisnaf[dataisnaf["Parch"]==0])#41 personnes

print(datanotnaf[datanotnaf["Parch"]==0])#153 personnes

parch0f=datanotnaf[datanotnaf["Parch"]==0]

x0f=parch0f['Age'].mean()

print(x0f)

parchfinal0f=dataisnaf[dataisnaf["Parch"]==0]

parchfinal0f['Age']=x0f

print(parchfinal0f['Age'])



#homme:

print(dataisnam[dataisnam["Parch"]==0])#116  personnes

print(datanotnam[datanotnam["Parch"]==0])#368 personnes

parch0m=datanotnam[datanotnam["Parch"]==0]

x0m=parch0m['Age'].mean()

print(x0m)

parchfinal0m=dataisnam[dataisnam["Parch"]==0]

parchfinal0m['Age']=x0m

print(parchfinal0m['Age'])



data=pd.concat([parchfinal0,parchfinal1,parchfinal2,datanotna])

print(data)





#supprimer des colonnes

data=data.drop(['Cabin'],axis=1)#supprime la colonne Cabin

data=data.drop(['PassengerId'],axis=1)#supprime la colonne PassengerId

data=data.drop(['Name'],axis=1)#supprime la colonne Name

data=data.drop(['Ticket'],axis=1)#supprime la colonne TIcket

data["Embarked"]=data["Embarked"].fillna("S")#remplace les valeurs manquantes par la plus fréquante



#convertir sex en {0,1} 

data["Sex"][data["Sex"] == "male"] = 0

data["Sex"][data["Sex"] == "female"] = 1

print(data["Sex"].head(28))



# Convertir la colonne Embarked  en entier

data["Embarked"][data["Embarked"] == "S"] = 0

data["Embarked"][data["Embarked"] == "C"] = 1

data["Embarked"][data["Embarked"] == "Q"] = 2

print(data["Embarked"].head(28))

data["Embarked"].value_counts()



#=========================================================================================================================

#                                              Chapitre 2                                                                #

#=========================================================================================================================



###############################################LA bASE 2##################################################################





#Attributs/Etiquettes

datanotna=data[data['Age'] .notna()]



y=datanotna['Age']

x=datanotna[['Survived','Pclass','Sex','familymembers','Embarked','Fare']]



#Base d'apprentissage et base de test

from sklearn.model_selection import train_test_split

x_train,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)

print('Train set',x_train.shape)

print('test set',x_test.shape)



#Modélisation des données

from sklearn.neighbors import KNeighborsRegressor

model=KNeighborsRegressor(n_neighbors=20)



#Entraînement du modèle

model.fit(x_train, y_train)

print('train score:',model.score(x_train,y_train))

print('test score:',model.score(x_test,y_test))



#La partie de prédiction

dataisna=data[data['Age'] .isna()]

y_final=dataisna['Age']

x_final=dataisna[['Survived','Pclass','Sex','familymembers','Embarked','Fare']]



y_final=model.predict(x_final)

dataisna['Age']=y_final



data_sklearn=pd.concat([dataisna,datanotna])

0

#####################################Modelisation de la survie#############################################################

#Attributs/Etiquettes

#?la base1



y=data['Survived']

x=data[['Age','Pclass','Sex','familymembers','Embarked','Fare']]



#la base 2

y=data_sklearn['Survived']

x=data_sklearn.drop(['Survived'],axis=1)#supprime la colonne Cabin

print(x.shape)

#Base d’apprentissage et base de test

from sklearn.model_selection import train_test_split



x_train,x_val ,y_train , y_val =train_test_split(x,y,test_size=0.2)

print('Train set',x_train.shape)

print('test set',x_val.shape)



#############es algorithmmes des modélisations#######

#-----------------Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=3,random_state=3)



random_forest.fit(x_train, y_train)

random_forest.score(x_train,y_train)

random_forest.score(x_val,y_val)



#cross-validation

from sklearn.model_selection import cross_val_score

rf_cv = cross_val_score(random_forest, x_train, y_train, cv=10)

rf_cv.mean()



#le meilleur nombre d'estimateur



from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

ne = np.arange(1,20)

param_grid = {'n_estimators' : ne}

rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)

rf_cv.fit(x_train,y_train)

print('Best value of n_estimators:',rf_cv.best_params_)

print('Best score:',rf_cv.best_score_*100)



#relancement de l'algorithme de Random Forest avec les meilleurs parametres

#base1



from sklearn.ensemble import RandomForestClassifier



random_forest=RandomForestClassifier(n_estimators=12)

random_forest.fit(x_train, y_train)

print('train score:',rf_cv.score(x_train,y_train))

print('test score:',rf_cv.score(x_val,y_val))

from sklearn.model_selection import cross_val_score

rf_scores = cross_val_score(rf_cv, x_train, y_train, cv=10)

print("CrossValMeans",rf_scores.mean())



#base2

from sklearn.ensemble import RandomForestClassifier



random_forest=RandomForestClassifier(n_estimators=13)

random_forest.fit(x_train, y_train)

print('train score:',rf_cv.score(x_train,y_train))

print('test score:',rf_cv.score(x_val,y_val))

from sklearn.model_selection import cross_val_score

rf_scores = cross_val_score(rf_cv, x_train, y_train, cv=10)

print("CrossValMeans",rf_scores.mean())



#-----------------KNN classifier 

#Initialisation du modèle:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)



#l’entrainement du modèle

knn.fit(x_train, y_train)

knn.score(x_train,y_train)

knn.score(x_test,y_test)



#Modèles de validation croisée :





from sklearn.model_selection import cross_val_score

KNN_scores = cross_val_score(random_forest, x_train, y_train, cv=10)

KNN_scores.mean()





#Utilisation de la valeur "optimale" :

#la base1



from sklearn.model_selection import cross_val_score

knn_scores = cross_val_score(knn, x_train, y_train, cv=10)

knn_scores.mean()



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=14)

ne = np.arange(1,20)

param_grid = {'n_neighbors' : ne}

from sklearn.model_selection import GridSearchCV

rf_cv = GridSearchCV(knn, param_grid=param_grid, cv=5)

rf_cv.fit(x_train,y_train)



#la base2



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train, y_train)

print('train score:',knn.score(x_train,y_train))

print('test score:',knn.score(x_val,y_val))

from sklearn.model_selection import cross_val_score

KNN_scores = cross_val_score(knn, x_train, y_train, cv=10)

print("CrossValMeans",KNN_scores.mean())

#----------------------------------SVM

#Initialisation du modèle :



from sklearn.svm import SVC

svm = SVC()



#l’entrainement du modèle

svm.fit(x_train, y_train)

print('train score:',svm.score(x_train,y_train))

print('test score:',svm.score(x_val,y_val))

#Modèles de validation croisée :

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm, x_train, y_train, cv=10)

print("CrossValMeans",svm_scores.mean())

#Modèles de validation croisée

from sklearn.svm import SVC



svm = SVC()

ne = np.arange(1,20)

param_grid = {'C' : ne}

from sklearn.model_selection import GridSearchCV

svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5)

svm_cv.fit(x_train,y_train)

print('Best value of n_neighbors:',svm_cv.best_params_)

print('Best score:',svm_cv.best_score_*100)





#Utilisation de la valeur "optimale"

#la base1

from sklearn.svm import SVC

svm = SVC(C=6)

svm.fit(x_train, y_train)

print('train score:',svm.score(x_train,y_train))

print('test score:',svm.score(x_val,y_val))

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm, x_train, y_train, cv=10)

print("CrossValMeans",svm_scores.mean())



#Base2

from sklearn.svm import SVC

svm = SVC(C=4)

svm.fit(x_train, y_train)

print('train score:',svm.score(x_train,y_train))

print('test score:',svm.score(x_val,y_val))

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm, x_train, y_train, cv=10)

print("CrossValMeans",svm_scores.mean())



#===========================La comparaison du modéles ===================================



models = pd.DataFrame({

    'Model': [ 'SVC', 

              'KNN','Random Forest'],

   "accuracy_score de la Base 2": [acc_svc, acc_knn,acc_randomforest]})

models.sort_values(by=  "accuracy_score de la Base 2",ascending=False)



#visualisation graphique de résultat de code:

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(RandomForestClassifier(random_state = random_state))

cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":['Random Forest' , 'KNN','SVM']})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")



#les facteurs importants:





from sklearn.model_selection import train_test_split

x_train,x_val ,y_train , y_val = train_test_split(x,y,test_size=0.2)

print('Train set',x_train.shape)

print('test set',x_val.shape)



from sklearn.metrics import accuracy_score



# Random Forests

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

random_forest.fit(x_train, y_train)

Y_pred_rf = random_forest.predict(x_val)

random_forest.score(x_train,y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

print("Important features")

pd.Series(random_forest.feature_importances_,x_train.columns).sort_values(ascending=True).plot.barh(width=0.8)



print('__'*30)

print(acc_random_forest)



#Confusion Matrix



from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



predictions = cross_val_predict(random_forest, x_train,y_train, cv=3)

confusion_matrix(y_train, predictions)



#=========================================================================================================================

#                                              Chapitre 3                                                                #

#=========================================================================================================================



data_test=pd.read_csv('../input/test.csv')#lire le fichier excel

print(100*data_test.isna().sum()/(len(data_test)))



#ajouter une colonne



data_test['familymembers']=data_test.SibSp+data_test.Parch



ids = data_test['PassengerId']

print(ids.shape)

#supprimer des colonnes

data_test=data_test.drop(['Cabin'],axis=1)#supprime la colonne Cabin

data_test=data_test.drop(['PassengerId'],axis=1)#supprime la colonne PassengerId

data_test=data_test.drop(['Name'],axis=1)#supprime la colonne Name

data_test=data_test.drop(['Ticket'],axis=1)#supprime la colonne TIcket



print(data_test)

#convertir sex en {0,1} 

genders = {"male": 0, "female": 1}

data_test['Sex'] = data_test['Sex'].map(genders)





# Convertir la colonne Embarked  en entier

ports = {"S": 0, "C": 1, "Q": 2}



data_test['Embarked'] = data_test['Embarked'].map(ports)



#missing data



print(data_test.isna().sum())



#Attributs/Etiquettes

datanotna=data_test[data_test['Age'] .notna()]



y=datanotna['Age']

x=datanotna[['Pclass','Sex','familymembers','Embarked']]



#Base d'apprentissage et base de test

from sklearn.model_selection import train_test_split

x_train,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)

print('Train set',x_train.shape)

print('test set',x_test.shape)



#Modélisation des données

from sklearn.neighbors import KNeighborsRegressor

model=KNeighborsRegressor(n_neighbors=20)



#Entraînement du modèle

model.fit(x_train, y_train)

print('train score:',model.score(x_train,y_train))

print('test score:',model.score(x_test,y_test))



#La partie de prédiction

dataisna=data_test[data_test['Age'] .isna()]

y_final=dataisna['Age']

x_final=dataisna[['Pclass','Sex','familymembers','Embarked']]

print(x_final.shape)

dataisna['Age']=model.predict(x_final)





data_test=pd.concat([dataisna,datanotna])

print(data_test.shape)

#remplacer les valeurs manquantes de "Fare"

data_test['Fare'].fillna(data_test['Fare'].median,inplace=True)

#Prediction 'Survived'

data_test=data_test[['Age','Pclass','Sex','familymembers','Embarked']]

data_test.dtypes#afficher les noms des colonnes de ce fichier



predictions = rf_cv.predict(data_test)

print(predictions.shape)

#Soumission

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

#-------------------------------------------------------------------------------------------------------------------------