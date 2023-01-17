import numpy as np 

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style
dstrain = pd.read_csv("../input/titanic/train.csv")

dstrain.head()
dstest = pd.read_csv("../input/titanic/test.csv")

dstest.head()
# Combinaisons des deux dataset en un seul pour ce faciliter le traitement des données

ds = pd.concat([dstrain, dstest], sort=False, ignore_index=True)
ds.tail()
ds.info()
ds.describe()
# Vérifier les valeurs nulles et afficher leur pourcentage

total = ds.isna().sum().sort_values(ascending=False)

percent_1 = ds.isna().sum()/ds.isna().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data = missing_data[missing_data['%']!=0]

missing_data
# Récupérer les valeurs uniques des différentes colonnes

colsWithUniqueValues =  ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

for col in colsWithUniqueValues:

    try:

        print(f"{col:<10} {sorted(ds[col].dropna().unique())}")

    except:

        print(f"{col:<10} {ds[col].unique()}")
# On peut voir ci-dessus que 38% du dataset a survecu

# Normalement Ticket, PassengerId ne devraient pas avoir d'impact sur le taux de survie

ds.drop(['PassengerId','Ticket'],axis=1,inplace = True)

ds.head()
# les ages ne suivent pas une distribution 'normale'

from scipy.stats import normaltest

# test de normalité

stat, p = normaltest(ds['Age'].dropna())

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpretation

alpha = 0.05

if p > alpha:

    print("l'échantillon semble Gaussien (impossible de rejeter l'hypothése nulle H0)")

else:

    print("l'échantillon ne semble pas Gaussien (rejet de l'hypothése nulle H0)")
# Renvoit un age au hasard pour la colonne Age qui suit la distribution des valeurs de la colonne

# utilisation de sample (Pandas)

ds['Age'].dropna().sample(ds['Age'].isnull().sum(),random_state=0)
def impute_nan(df,variable):

    # on duplique la colonne dans un premier temps (y compris les valeurs nulles qu'elle contient)

    df[variable+"_rnd"]=df[variable]

    # on génére une série de la taille des Age qui sont null et on sample dedans avec la distrib de la colonne Age

    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)

    # Pour réaliser la fusion, on conserve les index des Age null

    random_sample.index=df[df[variable].isnull()].index

    df.loc[df[variable].isnull(),variable+'_rnd']=random_sample
# on remplace dans le dataset les valeurs nulles dans la colonne Age

impute_nan(ds,"Age")

ds.head()
# On voit ci-dessous qu'en remplissant les 263 valeurs age manquantes que l'on n'a pas (peu) modifié la distribution

fig = plt.figure()

ax = fig.add_subplot(111)

ds['Age'].plot(kind='kde', ax=ax)

ds.Age_rnd.plot(kind='kde', ax=ax, color='green')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
# La colonne Age ne sert plus, elle contient des valeurs nulles, on utilisera Age_rnd que l'on vient de créer

ds.drop(['Age'],axis=1,inplace = True)
# Le port d'embarquement le plus fréquent est 'S'

dstrain['Embarked'].describe() 
common_value = 'S'

ds['Embarked'] = ds['Embarked'].fillna(common_value)
# Calcul du nombre de personnes accompagnantes

ds['Relatives'] = ds['SibSp'] + ds['Parch']



# création d'une colonne not alone qui va permettre d'avoir un booléen permettant de savoir

# si le voyageur est seul ou accompagné

ds.loc[ds['Relatives'] > 0, 'Not_alone'] = 1

ds.loc[ds['Relatives'] == 0, 'Not_alone'] = 0

ds['Not_alone'] = ds['Not_alone'].astype(int)
# Taux de survie en fonction du nombre de personnes accompagnantes

axes = sns.catplot('Relatives','Survived', data=ds, aspect = 2.5,kind='point')
# il semblerait que l'on survive plus en voyageant avec 1 à 3 personnes

# nous allons créer trois classes pour représenter ce que l'on peut apprécier visuellement sur le graphe

def family_cat(size):

    if (size >= 1) & (size < 4):

        return 0

    elif ((size >= 4) & (size < 7)) | (size == 0):

        return 1

    elif (size >= 7):

        return 2

    

ds['Famcat'] = ds['Relatives'].apply(family_cat)

ds['Famcat'] = ds['Famcat'].astype(int)

ds.head()
# Graphe du taux de survie en fonction de la catégorie de famille et du sexe

plt.figure(figsize=(8, 8))

sns.barplot(x="Famcat", y="Survived", hue="Sex", data=ds, palette='Blues_d')

plt.show()
ds['Fare_Per_Person'] = ds['Fare'].fillna(0)/(ds['Relatives']+1)

ds.head(10)
# Dans le dataset on a un Fare inconnu. Mais on sait que le voyageur était seul

ds[ds['Fare'].isna()]
# on va donc utiliser la moyenne de la classe 3 pour son Fare et Fare_Per_Person

fare = ds[ds['Pclass']==3]['Fare_Per_Person'].mean()

fare
ds.loc[ds['Fare'].isna(),'Fare_Per_Person'] = fare

ds.loc[ds['Fare'].isna(),'Fare'] = fare
# vérification

ds.iloc[1043,:]
# Taux de survie en fonction du Fare_Per_Person et du sexe

dstemp = ds.copy()

plt.figure(figsize=(8, 8))

# on crée des tranches de 'Fare' équilibrées avec qcut pour le graphe

dstemp['Fare_Per_Person'] = pd.qcut(dstemp['Fare_Per_Person'], 5)

sns.barplot(x="Fare_Per_Person", y="Survived", data=dstemp, hue ="Sex", palette='Blues_d')

plt.show()
# Creating 'Title' column

ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

ds['Title'].unique().tolist()
# Pourcentage de passagers en fonction du titre

ds['Title'].value_counts(normalize=True)*100
# Taux de survie en fonction du Titre

ds.groupby(['Title'])['Survived'].mean().sort_values(ascending=False) * 100
# pas de null dans la colonne crée

ds['Title'].isna().sum()
# Création de catégories pour regrouper les Titre qui ont un taux de survie équivalent

# Dona est surement un titre du dataset de test, mais c'est un titre de noblesse, on le met en catégorie Top



ds['Title'] = ds['Title'].replace(['Sir', 'Countess', 'Mme', 'Mlle', 'Dona' , 'Lady'], 'Top')

ds['Title'] = ds['Title'].replace(['Mrs', 'Miss'], 'High')

ds['Title'] = ds['Title'].replace(['Master', 'Dr', 'Col', 'Major', 'Ms'], 'Mid')

ds['Title'] = ds['Title'].replace(['Mr'], 'Low')

ds['Title'] = ds['Title'].replace(['Jonkheer', 'Rev', 'Don', 'Capt'], 'Bottom')



ds['Title'].value_counts()
# Taux de survie en fonction des catégories et du sexe 

plt.figure(figsize=(8, 8))

sns.barplot(x="Title", y="Survived", data=ds, order = ['Bottom','Low','Mid','High','Top'], hue ="Sex", palette='Blues_d')

plt.show()
# beaucoup de valeurs nulles dans la feature Cabin, mais pour le cabines renseignées

# on peut voir qu'elles commencent toutes par une lettre

ds[ds['Cabin'].isna() == False].head(5) 
ds['Cabin'] = ds['Cabin'].fillna('Unknown')

ds['Deck']=ds['Cabin'].str.get(0)



ds[ds['Cabin']!='Unknown'].head(5)
sorted(ds['Deck'].unique())
#visualisation du taux de survie par pont (Deck) et par sexe

plt.figure(figsize=(8, 8))

sns.barplot(x='Deck', y='Survived', data=ds, hue = 'Sex' ,palette='ocean', order = sorted(ds['Deck'].unique()))

plt.show()
ds['Pclass'] = ds['Pclass'].astype(str)

ds = pd.get_dummies(ds, columns=['Pclass','Embarked','Famcat','Title','Sex','Deck'],drop_first=False)

ds.head()
ds.columns
# drop des colonnes qui ne servent plus

ds.drop(['Name','SibSp','Parch','Cabin'],axis=1,inplace = True)

ds.head()
# Séparation des données Train / Test

train = ds[:len(dstrain)]



# on récupére la matrice de corrélation générale

corr = train.corr()

# on l'affiche avec sns

plt.figure(figsize=(15,15))

sns.heatmap(corr,annot = True, fmt='.1g',vmin=-1, vmax=1, center= 0, square = True, cbar = None, cmap= 'coolwarm')



# on peut voir ci dessous que le sexe est la feature la plus fortement correlée à la survie avec le titre

# mais les autres (Fare_Per_Person, Fare, Embarked, Pclass, Famcat, Title, Not_alone) le sont aussi dans une moindre mesure
# Normalisation des données numériques

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

ds[['Age_rnd', 'Fare', 'Fare_Per_Person', 'Relatives']] = scaler.fit_transform(ds[['Age_rnd', 'Fare', 'Fare_Per_Person', 'Relatives']])

ds.head()
# Séparation des données Train / Test

train = ds[:len(dstrain)]



# Splitting dataset into test

test = ds[len(dstrain):]
# Premier test rapide avec RandomForestClassifier   

from sklearn.ensemble import RandomForestClassifier

X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]



random_forest = RandomForestClassifier(n_estimators=100,random_state=0)

random_forest.fit(X_train, Y_train)



random_forest.score(X_train, Y_train)

print("Train score : ",round(random_forest.score(X_train, Y_train) * 100, 2))



# le score nous montre qu'on est probablement en overfitting...
# On peut récuperer les features qui ont été importante pour ce modéle

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.nlargest(20,'importance').set_index('feature')

importances
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()

model.fit(X_train,Y_train)

ranked_features=pd.Series(model.feature_importances_,index=X_train.columns)

plt.figure(figsize=(8, 8))

ranked_features.nlargest(len(X_train.columns)).sort_values(ascending=True).plot(kind='barh')

plt.show()
model.score(X_train, Y_train)

print("Train score : ",round(model.score(X_train, Y_train) * 100, 2))
from sklearn.tree import DecisionTreeClassifier,plot_tree

model=DecisionTreeClassifier()

model.fit(X_train,Y_train)

ranked_features=pd.Series(model.feature_importances_,index=X_train.columns)

plt.figure(figsize=(8, 8))

ranked_features.nlargest(len(X_train.columns)).sort_values(ascending=True).plot(kind='barh')

plt.show()
model.score(X_train, Y_train)

print("Train score : ",round(model.score(X_train, Y_train) * 100, 2))
# Avec un DecisionTree (CART), on peut afficher l'abre de décision qui a été crée

'''

plt.figure(figsize=(100,100))

plot_tree(model,feature_names=X_train.columns,class_names="Survived", filled=True,fontsize=6)

plt.savefig("dt.jpg",dpi = 100)

'''
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
# SelectKBest Algorithm

ordered_rank_features=SelectKBest(score_func=chi2,k=20)

ordered_feature=ordered_rank_features.fit(X_train,Y_train)

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])

dfcolumns=pd.DataFrame(X_train.columns)

features_rank=pd.concat([dfcolumns,dfscores],axis=1)

features_rank.columns=['Features','Score']

features_rank = features_rank.sort_values('Score',ascending=False).set_index('Features')

features_rank



# les features les plus liées au taux de survie, sont le sexe, le titre, la classe, la catégorie de famille (telle qu'on l'a crée).

# on peut voit notamment que l'age et le prix du billet ne sont pas si important
ranked_features=pd.Series(ordered_feature.scores_,index=X_train.columns)

plt.figure(figsize=(8, 8))

ranked_features.nlargest(len(X_train.columns)).sort_values(ascending=True).plot(kind='barh')

plt.show()
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier



seed = 47

# preparation des modéles

models = []

models.append(('LR', LogisticRegression()))

models.append(('SGD', SGDClassifier()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier(3)))

models.append(('CART', DecisionTreeClassifier()))

models.append(('EXT', ExtraTreesClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('ADAB', AdaBoostClassifier()))

models.append(('GDB', GradientBoostingClassifier()))

models.append(('SVM', SVC()))

models.append(('LSVC', LinearSVC()))

models.append(('XGB', XGBClassifier()))

# évaluation des modéles

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle=True)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring,n_jobs=-1)

    results.append(cv_results)

    names.append(name)

    msg = f"{name:<5}: Mean={cv_results.mean():-<10.3f}Median={np.median(cv_results):-<10.3f}std={cv_results.std():.4f}"

    print(msg)

# graphe de comparaison en boxplot 

fig = plt.figure()

fig.suptitle('Comparaison des Algorithmes')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV



#Modéles que l'on va essayer

clf_lr = LogisticRegression()

clf_lda = LinearDiscriminantAnalysis()

clf_rf = RandomForestClassifier()

clf_gdb = GradientBoostingClassifier()

clf_svm = SVC()

clf_lsvc = LinearSVC()

clf_xgb = XGBClassifier()



classifiers = [clf_lr, clf_lda, clf_rf, clf_gdb,clf_svm,clf_lsvc,clf_xgb]



### paramétres de départ pour RandomizedSearchCV



### ------------------

### LogisticRegression

### ------------------

param_lr = {"penalty" :          ["l1","l2"],

            "tol" :              [0.0001,0.0002,0.0003],

            "max_iter":          [100,300,500,800,1000],

            "C" :                [0.01, 0.1, 1, 10, 100],

            "intercept_scaling": [1, 2, 3, 4],

            "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}



### --------------------------

### LinearDiscriminantAnalysis

### --------------------------

param_lda = {"solver" : ['svd', 'lsqr', 'eigen']}



### ----------------------

### RandomForestClassifier

### ----------------------

# Nombre d'abre dans RandomForest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Nombre de features à considérer à chaque split

max_features = ['auto', 'sqrt']

# Nombre maximum de niveau dans les abres

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Combien d'echantillon au minimum pour séparer un noeud

min_samples_split = [2, 5, 10]

# Nombre minimum d'échantillons dans chaque feuille

min_samples_leaf = [1, 2, 4]

# Méthode de séléction des échantillons

bootstrap = [True, False]



param_rf = {'n_estimators':      n_estimators,

            'criterion':         ['entropy', 'gini'],

            'max_features':      max_features,

            'max_depth':         max_depth,

            'min_samples_split': min_samples_split,

            'min_samples_leaf':  min_samples_leaf,

            'bootstrap':         bootstrap}



### --------------------------

### GradientBoostingClassifier

### --------------------------



param_gdb = {

    "loss":              ["deviance"],

    "learning_rate":     [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "min_samples_split": np.linspace(0.1, 0.5, 12),

    "min_samples_leaf":  np.linspace(0.1, 0.5, 12),

    "max_depth":         [3,5,8],

    "max_features":      ["log2","sqrt"],

    "criterion":         ["friedman_mse",  "mae"],

    "subsample":         [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":      [10,100,500]}





### -------

### XGBoost

### -------



param_xgb = {

    'n_estimators':     [10,100,500],

    'colsample_bytree': [0.75,0.8,0.85],

    'max_depth':        [10,50,100,None],

    'reg_alpha':        [1],

    'reg_lambda':       [2, 5, 10],

    'subsample':        [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    'learning_rate':    [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    'gamma':            [.5,1,2],

    'min_child_weight': [0.01],

    'sampling_method':  ['uniform']}





### ---------------

### SVC 

### ---------------



param_svc = {'gamma':  [1e-3, 1e-2,0.1, 1, 10,100],

             'C':      [1e-2,0.1,1, 10, 100, 1000],

             'kernel': ['linear', 'rbf'],

             'degree': [1,2,3]}



### ---------------

### LinearSVC 

### ---------------



param_lsvc = {'C':        [1e-2,0.1,1, 10, 100, 1000],

              'tol' :     [0.0001,0.0002,0.0003],

              'max_iter': [100,300,500,800,1000]}





parameters = [param_lr, param_lda, param_rf, param_gdb, param_svc, param_lsvc, param_xgb]





clf_best_acc = []

clf_best_params = []

clf_best_estimator = []

rnd_searchs = [] 



#Recherche sur tous les classifiers

for i in range(len(classifiers)):

    rnd_searchs.append(RandomizedSearchCV(estimator = classifiers[i],

                                 param_distributions = parameters[i],

                                 scoring = 'accuracy',

                                 n_iter = 40,

                                 cv = 5,

                                 verbose=0,

                                 random_state=47,

                                 n_jobs = -1))

    

    print(classifiers[i].__class__.__name__)

    underline=['-']*len(classifiers[i].__class__.__name__)

    print(''.join(underline))

    rnd_searchs[i].fit(X_train, Y_train)

    print("Meilleurs paramétres :",rnd_searchs[i].best_params_)

    print("Score : ",rnd_searchs[i].best_score_)

    clf_best_acc.append(rnd_searchs[i].best_score_)

    clf_best_params.append(rnd_searchs[i].best_params_)

    clf_best_estimator.append(rnd_searchs[i].best_estimator_)



print("")    

print("RandomSearchCV Terminé...")
modelNames=[]

for model in classifiers:

    modelNames.append(model.__class__.__name__)

rndtunedScores=pd.Series(clf_best_acc,index=modelNames)

rndtunedScores=rndtunedScores-0.8

plt.figure(figsize=(8, 8))

rndtunedScores.sort_values(ascending=True).plot(kind='barh',left = 0.8)

plt.show()

# comme on peut le voir tous les modéles sont dans un mouchoir de poche...
from sklearn.model_selection import learning_curve

import warnings

warnings.filterwarnings('ignore')
### ------------------

### LogisticRegression

### ------------------

param_lr = {  "penalty" : ["l2"],

              "tol" : [0.0001,0.00015,0.0002],

              "max_iter": [1,2,5,10,100,200,600,800],

              "C" :[0.01, 0.1, 1],

              "intercept_scaling": [2,3,4],

              "solver":['sag']}



clf_lr = LogisticRegression()
# CV=5 / petite amélioration 

best_clf_lr = GridSearchCV(clf_lr, param_grid = param_lr, cv = 5, verbose = False, n_jobs = -1).fit(X_train,Y_train)

print(best_clf_lr.best_params_)

print(best_clf_lr.best_score_)
# CV = 10 améliore encore le résultat

best_clf_lr = GridSearchCV(clf_lr, param_grid = param_lr, cv = 10, verbose = False, n_jobs = -1).fit(X_train,Y_train)

print(best_clf_lr.best_params_)

print(best_clf_lr.best_score_)
def plotLearningCurve(model):

    N, train_score, val_score = learning_curve(model, X_train, Y_train,

                                                  train_sizes=np.linspace(0.1, 1, 10), cv=10)



    # N contient le nombre d'éléments retenus pour faire l'entrainement

    print(N)

    # train_score contient pour toutes les itérations de train_sizes (10 ici), les résultat pour chaque cv sur le jeu de train

    # val_score contient pour toutes les itérations de train_sizes (10 ici), les résultat pour chaque cv sur le jeu de validation

    plt.plot(N, train_score.mean(axis=1), label='train')

    plt.plot(N, val_score.mean(axis=1), label='validation')

    plt.xlabel('train_sizes')

    plt.legend() 





# on récupére le meilleur modéle trouvé grâce à GridSearchCV 

model = best_clf_lr.best_estimator_ 

plotLearningCurve(model)
from sklearn.model_selection import train_test_split



# précédemment on a vu grace à learning curves, que la meilleur taille du dataset pour l'entrainement était la taille de 720

# on réinjecte les paramétres trouvés avec GridSearchCV dans notre modéle, avec 10% de test_size



model_lr = LogisticRegression(C=1,

                           intercept_scaling=4,

                           max_iter=10,

                           penalty='l2',

                           solver='sag',

                           tol=0.00015,

                           random_state=47)                          

X_tr, X_te, y_tr, y_te = train_test_split(X_train, Y_train, test_size=0.1)

model_lr.fit(X_tr, y_tr)

print("train : ",model_lr.score(X_tr, y_tr))

print("test  : ",model_lr.score(X_te, y_te))
### ----------------------

### RandomForestClassifier

### ----------------------

n_estimators = [int(x) for x in np.linspace(start = 1300, stop = 1500, num = 10)]

max_features = ['sqrt']

max_depth = [int(x) for x in np.linspace(20, 40, num = 10)]

min_samples_split = [4,5,6]

min_samples_leaf = [3,4,5]

bootstrap = [False]

criterion = ['entropy']

# Create the random grid

param_rf = {'n_estimators': n_estimators,

            'criterion': criterion,

            'max_features': max_features,

            'max_depth': max_depth,

            'min_samples_split': min_samples_split,

            'min_samples_leaf': min_samples_leaf,

            'bootstrap': bootstrap}
# désactivé car prends trop de temps sur Kaggle

'''

clf_rf = RandomForestClassifier()

best_clf_rf = GridSearchCV(clf_rf, param_grid = param_rf, cv = 5, verbose = True, n_jobs = -1).fit(X_train,Y_train)

print(best_clf_rf.best_params_)

print(best_clf_rf.best_score_)

'''
'''

# on récupére le meilleur modéle trouvé grâce à GridSearchCV 

model = best_clf_rf.best_estimator_ 

plotLearningCurve(model)

'''
model_rf = RandomForestClassifier(bootstrap=False,

                                  criterion='entropy',

                                  max_depth=28,

                                  max_features='sqrt',

                                  min_samples_leaf=3,

                                  min_samples_split=5,

                                  n_estimators= 1477,

                                  random_state=47)

X_tr, X_te, y_tr, y_te = train_test_split(X_train, Y_train, test_size=0.3)

model_rf.fit(X_tr, y_tr)

print("train : ",model_rf.score(X_tr, y_tr))

print("test  : ",model_rf.score(X_te, y_te))
### -------

### XGBoost

### -------



param_xgb = {

    'n_estimators':     [100],

    'colsample_bytree': [0.85],

    'max_depth':        [80,90],

    'reg_alpha':        [1],

    'reg_lambda':       [9,10,11],

    'subsample':        [0.95],

    'learning_rate':    [0.05],

    'gamma':            [.75,1],

    'min_child_weight': [0.01],

    'sampling_method':  ['uniform']}
clf_xgb = XGBClassifier()

best_clf_xgb = GridSearchCV(clf_xgb, param_grid = param_xgb, cv = 5, verbose = False, n_jobs = -1).fit(X_train,Y_train)

print(best_clf_xgb.best_params_)

print(best_clf_xgb.best_score_)
best_clf_xgb = GridSearchCV(clf_xgb, param_grid = param_xgb, cv = 10, verbose = False, n_jobs = -1).fit(X_train,Y_train)

print(best_clf_xgb.best_params_)

print(best_clf_xgb.best_score_)
# on récupére le meilleur modéle trouvé grâce à GridSearchCV 

model = best_clf_xgb.best_estimator_ 

plotLearningCurve(model)
model_xgb = XGBClassifier(n_estimators =100,

                            colsample_bytree = 0.85,

                            max_depth=80,

                            reg_alpha=1,

                            reg_lambda=10,

                            subsample=0.95,

                            learning_rate=0.05,

                            gamma=1,

                            min_child_weight=0.01,

                            sampling_method='uniform',

                            random_state=47)

X_tr, X_te, y_tr, y_te = train_test_split(X_train, Y_train, test_size=0.05)

model_xgb.fit(X_tr, y_tr)

print("train : ",model_xgb.score(X_tr, y_tr))

print("test  : ",model_xgb.score(X_te, y_te))
temp = pd.read_csv("../input/titanic/test.csv")

Id = temp.PassengerId



# Splitting dataset into test

test = ds[len(dstrain):]

X_test = test.drop("Survived", axis=1)



final_predictions_lr = model_lr.predict(X_test)

final_predictions_rf = model_rf.predict(X_test)

final_predictions_xgb = model_xgb.predict(X_test)





outputlr = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions_lr.astype(int)})

outputrf = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions_rf.astype(int)})

outputxgb = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions_xgb.astype(int)})

#outputlr.to_csv('../output/submissionlr.csv', index=False)

#outputrf.to_csv('../output/submissionrf.csv', index=False)

#outputxgb.to_csv('../output/submissionxgb.csv', index=False)
from sklearn.ensemble import VotingClassifier
model_vt = VotingClassifier([('LR', model_lr),

                            ('RF', model_rf),

                            ('XGB', model_xgb)],

                            voting='hard')

# Différence entre voting = soft / hard :

# voting = soft => on additionne les probabilités de chaque classe pour tous les classifiers, on prend la classe qui a la somme

# de proba la plus importante (fonctionne bien si pas trop de disparités entre classifiers)

# voting = hard => on prend la classe qui a été prédite par le plus de classfiers



model_vt.fit(X_train, Y_train)

print(model.score(X_train, Y_train))
final_predictions_vt = model_vt.predict(X_test)

outputvt = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions_vt.astype(int)})

#outputvt.to_csv('../output/submissionvt.csv', index=False)