import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets
from sklearn import tree
import statistics as stat
import seaborn as sns
import collections

from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import discriminant_analysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
import xgboost as xgb

#Lees bijde sets om later samen te voegen om pre-processing op uit te voeren
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


#haal de lijst van survivers eruit en sla het in een panda 
#data frame colom
df_survived = df_train.Survived
df_train = df_train.drop('Survived',axis= 1)
#Voeg bijde lijsten samen voor preprocessing
pre_data = pd.concat([df_train, df_test])

#Haal de ticket colom eruit
pre_data = pre_data.drop(['Ticket'], 1)

pre_data.info()
#plot leeftijd in combinatie met features om de feature te vinden die
#een goede voorspellende waarden geven over leeftijd
f,ax = plt.subplots(2,2,figsize=(10,10))
sns.boxplot(x='Sex', y='Age',data=df_train, ax = ax[0,0])
sns.boxplot(x='Pclass', y='Age',data=df_train, ax = ax[0,1])
sns.boxplot(x='Embarked', y='Age', data=df_train, ax = ax[1,0])
#Voeg overleeft colom voor plotting 
train_with_surv = pd.concat([df_train, df_survived],axis=1)
sns.boxplot(x='Survived', y='Age', data=train_with_surv, ax = ax[1,1])
plt.show()
# row and column sharing
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,10))

##############################################
###### Standard model whole training set #####
##############################################
#Stop in een lijst alle leeftijden die voorkomen in training en test
ages = [value for value in pre_data.get('Age').as_matrix() if not math.isnan(value)]
nanfreq = 0

#Tel het hoeveeltijd missende leeftijden er zijn in de totale data
for value in pre_data.get('Age').as_matrix():
    if math.isnan(value):
        nanfreq += 1 
#bereken het gemiddelde en standaard standaarddeviatie van 
#de distributie van leeftijden van test en train samen
sigma = np.std(ages)
mu = np.mean(ages)

#Plot de distributie
count, bins, ignored = ax1.hist(ages, 30, normed=True)
ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
          linewidth=2, color='r')
ax1.title.set_text('Standard model complete trainingset')

######################################
##### First class standard model #####
######################################
#Stop in een lijst alle leeftijden die voorkomen in klassen 1 van de training
fc_ages = [value for value in df_train[df_train.Pclass == 1].get('Age') if not math.isnan(value)]
fc_nan = 0

#Tel het hoeveeltijd missende leeftijden er zijn als iemand in klassen 1 zit
for value in pre_data[pre_data.Pclass == 1].get('Age'):
    if math.isnan(value):
        fc_nan = fc_nan + 1 
        
#Bereken het gemiddelde en standaard standaarddeviatie van 
#de distributie van leeftijden van mensen die in klassen 1 zitten
fc_sigma = np.std(fc_ages)
fc_mu = np.mean(fc_ages)

#Plot de distributie
count, bins, ignored = ax2.hist(fc_ages, 30, normed=True)
ax2.plot(bins, 1/(fc_sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - fc_mu)**2 / (2 * fc_sigma**2) ),
          linewidth=2, color='r')
ax2.title.set_text('Standard model First Class')

#######################################
##### Second class standard model #####
#######################################
#Stop in een lijst alle leeftijden die voorkomen in klassen 2 van de training
sc_ages = [value for value in df_train[df_train.Pclass == 2].get('Age') if not math.isnan(value)]
sc_nan = 0

#Tel het hoeveeltijd missende leeftijden er zijn als iemand in klassen 2 zit
for value in pre_data[pre_data.Pclass == 2].get('Age'):
    if math.isnan(value):
        sc_nan = sc_nan + 1 

#Bereken het gemiddelde en standaard standaarddeviatie van 
#de distributie van leeftijden van mensen die in klassen 2 zitten
sc_sigma = np.std(sc_ages)
sc_mu = np.mean(sc_ages)

#Plot de distributie
count, bins, ignored = ax3.hist(sc_ages, 30, normed=True)
ax3.plot(bins, 1/(sc_sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - sc_mu)**2 / (2 * sc_sigma**2) ),
          linewidth=2, color='r')
ax3.title.set_text('Standard model Second Class')

######################################
##### Third class standard model #####
######################################
#Stop in een lijst alle leeftijden die voorkomen in klassen 3 van de training
tc_ages = [value for value in df_train[df_train.Pclass == 3].get('Age') if not math.isnan(value)]
tc_nan = 0

#Tel het hoeveeltijd missende leeftijden er zijn als iemand in klassen 3 zit
for value in pre_data[pre_data.Pclass == 3].get('Age'):
    if math.isnan(value):
        tc_nan = tc_nan + 1 

#Bereken het gemiddelde en standaard standaarddeviatie van 
#de distributie van leeftijden van mensen die in klassen 3 zitten
tc_sigma = np.std(tc_ages)
tc_mu = np.mean(tc_ages)

#Plot de distributie
count, bins, ignored = ax4.hist(tc_ages, 30, normed=True)
ax4.plot(bins, 1/(tc_sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - tc_mu)**2 / (2 * tc_sigma**2) ),
          linewidth=2, color='r')
ax4.title.set_text('Standard model Third Class')
plt.show()
#Maak lijsten aan per klassen met willekeurige gekozen leeftijden(als element) binnen 
#de distributies van die klassen
random_ages = np.random.normal(mu, sigma, nanfreq)
fc_random_ages = np.random.normal(fc_mu, fc_sigma, fc_nan)
sc_random_ages = np.random.normal(sc_mu, sc_sigma, sc_nan)
tc_random_ages = np.random.normal(tc_mu, tc_sigma, tc_nan)

#Maak ints van de floats
random_ages_round = [abs(round(i)) for i in random_ages]
fc_random_ages_round = [abs(round(i)) for i in fc_random_ages]
sc_random_ages_round = [abs(round(i)) for i in sc_random_ages]
tc_random_ages_round = [abs(round(i)) for i in tc_random_ages]

#Vul een string in alle missende leeftijden
age_data = pre_data.fillna(value={'Age':'NaN'})

numpy_data = age_data.values

#Vervang per persoon de missende leeftijd aan 
#gebaseert op in welke klassen ze zitten
for row in numpy_data:
    if row[1] == 1 and row[4]=='NaN':
        row[4] = fc_random_ages_round[0]
        fc_random_ages_round.pop(0)
    if row[1] == 2 and row[4] == 'NaN':
        row[4] = sc_random_ages_round[0]
        sc_random_ages_round.pop(0)
    if row[1] == 3 and row[4] =='NaN':
        row[4] = tc_random_ages_round[0]
        tc_random_ages_round.pop(0)
        
data = pd.DataFrame(numpy_data,columns=['PassengerId', 
                                         'Pclass', 'Name', 'Sex', 
                                         'Age', 'SibSp', 'Parch',
                                         'Fare', 'Cabin', 'Embarked'])

#Bereken gemiddelde en standaarddeviatie met ingevulde leeftijden
new_ages = [value for value in data.get('Age').as_matrix() if not math.isnan(value)]
newsigma = np.std(new_ages)
newmu = np.mean(new_ages)
print('Old sigma:', sigma, '\nNew sigma:', newsigma, '\nOld mu:', mu,'\nNew mu:', newmu)
data = data.fillna(value={'Cabin':'NaN','Embarked': 'S','Fare':stat.mode(data['Fare'])})
############# age ##############

#Feature engineeer de test en training tegelijk
#Splits de continue waarden van leeftijd op in groupen voor betere visualizatie
#en machine learning resultaten
data.loc[data['Age']<=10, 'AgeGroups'] = 'AgeGroup0'
data.loc[(data['Age']>10) & (data['Age']<=20), 'AgeGroups'] = 'AgeGroup1'
data.loc[(data['Age']>20) & (data['Age']<=30), 'AgeGroups'] = 'AgeGroup2'
data.loc[(data['Age']>30) & (data['Age']<=40), 'AgeGroups'] = 'AgeGroup3'
data.loc[(data['Age']>40) & (data['Age']<=50), 'AgeGroups'] = 'AgeGroup4'
data.loc[data['Age']>50, 'AgeGroups'] = 'AgeGroup5'

############# bloedverwanten ##############

#Maak 1 colom aan met hoeveelheid bloedverwanten.
#en 1 met of er bloedverwanten aan boord zijn of niet
data['NumberRelatives']= data.loc[:,'Parch']+ data.loc[:,'SibSp']
data['Relatives'] = 0
data['Relatives'] = data['NumberRelatives'].mask(data['NumberRelatives'] > 0, 1)

############# ticketprijs ##############

#maak groepen aan gebaseert op frequenties van fares
Fare_categories = pd.qcut(data['Fare'], 4)
#print(Fare_categories)

#gebaseert op Fare_categories uitkomsten maak numerieke groepen aan 
data.loc[ data['Fare'] <= 8, 'FareGroups'] = 'FareGroup0'
data.loc[(data['Fare'] > 8) & (data['Fare'] <= 15), 'FareGroups'] = 'FareGroup1'
data.loc[(data['Fare'] > 15) & (data['Fare'] <= 31), 'FareGroups']= 'FareGroup2'
data.loc[ data['Fare'] > 31, 'FareGroups'] = 'FareGroup3'

############# titels ##############

#Maak een colom aan met Titles gevonden in de Name colom
cnt = collections.Counter()
names = ['Dr', 'Rev', 'Col', 'Major', 'Mme', 'Sir', 'Lady', 'the Countess', 'Don', 'Ms', 'Capt', 'Jonkheer']
titles = []

#split titles
for i in data.get('Name'):
    title = i.split(', ')[1].split('.')[0]
    if title == 'Mlle':
        title = 'Miss'
    elif title in names:
        title = 'Special'
    titles.append(title)

    cnt[title] += 1
#maak van titles een colom 
titleseries = pd.Series(titles)
data['Titles'] = titleseries
#format style voor seaborn
sns.axes_style()
sns.set_style("dark")
sns.despine()
sns.set_palette("pastel")
sns.set_context('notebook')

#Split de training en test
training = data.iloc[:891]
test = data.iloc[891:]

#voeg survived colom toe aan clean training data
training = pd.concat([training, df_survived],axis=1)

#Hoeveel overleeft : grafiek en percentage
sns.countplot(x='Survived',data=training)
plt.title('Histogram by Survival')
plt.ylabel('Passengers')
plt.show()
print('Percentage survive',training['Survived'].mean())
print('Percentage not survive',1-training['Survived'].mean())

#Hoeveel overleeft gebaseert op klassen : grafiek en percentage
sns.countplot('Pclass',hue='Survived',data=training)
plt.title('Class Histogram by Survival')
plt.xlabel('Class')
plt.ylabel('Passengers')
plt.show()
print (training[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean())

#Hoeveel overleeft gebaseert op sex : grafiek en percentage
sns.countplot('Sex',hue='Survived',data=training)
plt.title('Gender Histogram by Survival')
plt.xlabel('Gender)')
plt.ylabel('Passengers')
plt.show()
print (training[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())


#Hoeveel overleeft gebaseert op leeftijds interval : grafiek en percentage
plt.title('Age Histogram by Survival')
sns.countplot('AgeGroups',hue='Survived',data=training)
plt.title('Age groups Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('Passengers')
plt.show()
print (training[['AgeGroups','Survived']].groupby(['AgeGroups'], as_index=False).mean())

#Hoeveel overleeft gebaseert op (stief)broer,(stief)zus : grafiek en percentage
sns.countplot('SibSp',hue='Survived',data=training)
plt.title('Sibling Histogram by Survival')
plt.xlabel('Siblings')
plt.ylabel('Passengers')
plt.show()
print (training[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean())

#Hoeveel overleeft gebaseert op (stief)broer,(stief)zus : grafiek en percentage
sns.countplot('Parch',hue='Survived',data=training)
plt.title('Parent Histogram by Survival')
plt.xlabel('Children')
plt.ylabel('Passengers')
plt.show()
print (training[['Parch','Survived']].groupby(['Parch'], as_index=False).mean())

#Hoeveel overleeft gebaseert op bloedverwant : grafiek en percentage
sns.countplot('Relatives',hue='Survived',data=training)
plt.title('Relatives Histogram by Survival')
plt.xlabel('Relatives')
plt.ylabel('Passengers')
plt.show()
print (training[['Relatives','Survived']].groupby(['Relatives'], as_index=False).mean())

#Hoeveel overleeft gebaseert op prijs groepen : grafiek en percentage    
sns.countplot('FareGroups',hue='Survived',data=training)
plt.title('Fare Histogram')
plt.xlabel('Fare')
plt.ylabel('Passengers')
plt.show()
print (training[['FareGroups','Survived']].groupby(['FareGroups'], as_index=False).mean())

#Hoeveel overleeft gebaseert op haven : grafiek en percentage
sns.countplot('Embarked',hue='Survived',data=training)
plt.title('Embarked Histogram by Survival')
plt.xlabel('Ports')
plt.ylabel('Passengers')
plt.show()
print (training[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean())
#maak een dummie colom van embarked voeg toe aan training
embark_dummies = pd.get_dummies(data.get('Embarked'))
data = pd.concat([data, embark_dummies], axis=1)
#del result['Embarked']

#maak een dummie colom van pclass voeg toe aan training
pclass_dummies = pd.get_dummies(data.get('Pclass'))
data = pd.concat([data, pclass_dummies], axis=1)
#del result['Pclass']

#maak een dummie colom van Faregroups voeg toe aan training
faregroups_dummies = pd.get_dummies(data.get('FareGroups'))
data = pd.concat([data, faregroups_dummies], axis=1)
#del result['AgeGroups']

#maak een dummie colom van Faregroups voeg toe aan training
agegroups_dummies = pd.get_dummies(data.get('AgeGroups'))
data = pd.concat([data, agegroups_dummies], axis=1)
#del result['AgeGroups']

#maak een dummie colom van titlesvoeg toe aan training
title_dummies = pd.get_dummies(titles)
data = pd.concat([data, title_dummies], axis=1)

#Vervang de string male met 0 en female met 1 om
#machine learning uit te kunnen voeren
data = data.replace(['male', 'female'], [0, 1])

data.info()
#reset sns style
sns.reset_orig()
#kies de colommen van dataframe 1 
possible_data = data[['Sex', 'Relatives', 'C','Q','S',1,2,3,
                     'FareGroup0','FareGroup1','FareGroup2','FareGroup3','AgeGroup0','AgeGroup1','AgeGroup2','AgeGroup3',
                      'AgeGroup4','AgeGroup5',
                     'Master','Miss','Mr','Mrs','Special']]

#kies de colommen van datafram 2
possible_treedata = data[['Pclass','Sex','Embarked','AgeGroups','Relatives','NumberRelatives','FareGroups','Titles']]
#embarked: S=0,C=1,Q=2, FAREGROUPS: FareGroup0 = 0,FareGroup1 = 1,FareGroup2 = 2,FareGroup3 = 3
#Titles :Mr=0,Mrs = 1,Miss = 2, Master = 4,Dona = 3,Special = 5
possible_treedata = possible_treedata.replace(['S', 'C','Q','FareGroup0','FareGroup1','FareGroup2','FareGroup3',
                                               'AgeGroup0','AgeGroup1','AgeGroup2','AgeGroup3','AgeGroup4','AgeGroup5',
                                              'Mr','Mrs','Miss','Dona','Master','Special'], 
                                              [0,1,2,0,1,2,3,0,1,2,3,4,5,0,1,2,3,4,5])
#possible_treedata = possible_treedata[['Pclass','Sex','Embarked','AgeGroups','Relatives','NumberRelatives','FareGroups']]


#Split de training en test voor tree algoritme
treedata_training = possible_treedata.iloc[:891]
treedata_test = possible_treedata.iloc[891:]

#Split de training en test
training = possible_data.iloc[:891]
test = possible_data.iloc[891:]

#zet om naar np.array
X = training.values
test = test.values

Xtree = treedata_training.values
testtree = treedata_test.values

y = df_survived.values

treedata_training.info()
print('|----------|')
training.info()
#Namen van de te testen algoritme
test_names = ['DecisionTree','DecisionTree_depth5','RandomForest','RandomForest_GS','RandomForest_Z']

#De te testen algoritme
test_clf = [tree.DecisionTreeClassifier(),
            tree.DecisionTreeClassifier(max_depth=5),
            ensemble.RandomForestClassifier(n_jobs=-1),
            ensemble.RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=400, n_jobs=-1, random_state=1),
            ensemble.RandomForestClassifier(criterion='entropy',random_state=1,min_samples_leaf=1, min_samples_split=16, n_estimators=50, n_jobs=-1)
           ]

names = ["DecisionTree","ExtraTree","RandomForest","AdaBoost","Bagging",
         "GradientBoost","GaussianProcess","LogisticRegressionCV","PassiveAggressive","Ridge","SGD","Perceptron","BernoulliNB",
         "GaussianNB","LinearDiscriminant","QuadraticDiscriminant",'XGB',"KNN","NeuralNet","linear SVM","RBF SVM"]

clfl = [tree.DecisionTreeClassifier(max_depth=3),
        tree.ExtraTreeClassifier(max_depth=3),
        
        ensemble.RandomForestClassifier(),
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.GradientBoostingClassifier(),
        
        gaussian_process.GaussianProcessClassifier(),
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifier(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),
    
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
        
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        
        xgb.XGBClassifier(),
        
        KNeighborsClassifier(),
        MLPClassifier(alpha=1),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1)]

# Een consistente split om te gebruiken voor cross validatie
cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

#Maakt een array aan met daarin lijsten van cross validation scores van algoritme.
#Input is een numpy array versie van de training data die je wil gebruiken, de 
#numpy versie van de survived colom, panda versie van de training data, namen van de algoritme,
#de algorime zelf, hoeveelheid cross validation scores je wil en de cv split 
def multi_cross_val(training_data_np, training_y, training_data_panda,names, clfl,iterations, cross_split):
    total_algo_array = []
    for i in range(iterations):
        algo_array = []
        for name, clf in zip(names, clfl):
            scores = cross_val_score(clf, training_data_panda, training_y, cv=cross_split)
            algo_array.append(scores)
        total_algo_array.append(algo_array)
    return total_algo_array

#Plot de cross validation score van een lijst van algoritme 
#Accuracy op de y-as en hoeveelheid iteraties van een algoritme
#gebaseert op mulit_cross_val op de x-as
def plot_ML_algo(lijstvanmlnamen, array):
    namendict = {}
    namen_means =[]
    index = 0
    for naam in lijstvanmlnamen:
        namendict[naam] = []
        for i in range(len(array)):
            namendict[naam].append(array[i][index])
        index+=1
    for_one = []
    for naam in namendict:
        mean_array = []
        for i in range(len(array)):
            mean = np.mean(namendict[naam][i])
            print(naam,mean)
            mean_array.append(mean)
            for_one.append(mean)
        #print(naam,np.mean(mean_array))
        if np.mean(mean_array)> 0.70:
            if len(array) != 1:
                plt.plot(np.linspace(0, len(array), len(array)), mean_array, label=naam)
                plt.legend(loc='best')
    if len(array) == 1:
        s = sns.barplot(lijstvanmlnamen, for_one)
        s.set_ylim(0.5, 1.0)
        plt.xticks(rotation=80)
    plt.ylabel('Cross Validation Accuracy')
    plt.show()
            
#Output een array met daarin een array waarvan elke lijst 
#de cross val scores zijn van een algoritme in de lijst van algoritme
scorearray = multi_cross_val(X,y,training,test_names,test_clf,1,cv_split)

#Plot de array
plot_ML_algo(test_names,scorearray)


#Output een array met daarin een array waarvan elke lijst 
#de cross val scores zijn van een algoritme in de lijst van algoritme
treescorearray = multi_cross_val(Xtree,y,treedata_training,test_names,test_clf,1,cv_split)

#Plot de array
plot_ML_algo(test_names,treescorearray)

#
# scoresarray = multi_cross_val(X,y,training,names,clfl,1,cv_split)
# plot_ML_algo(names,scoresarray)

# tscoresarray = multi_cross_val(Xtree,y,training,names,clfl,1,cv_split)
# plot_ML_algo(names,tscoresarray)
# voor het maken van een afbeelding van een DecisionTree
# import graphviz as gv
# Code om een afbeelding te maken van een booms
# clf = test_clf[1]
# clf.fit(Xtree,y)

# dot_data = tree.export_graphviz(clf, out_file=None)

# graph = gv.Source(dot_data)
# graph.render()
# Voor de cross validation splitting
cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

# classifiers
c = [ensemble.AdaBoostClassifier(),
     ensemble.ExtraTreesClassifier(n_jobs=-1),
     ensemble.GradientBoostingClassifier(),
     ensemble.RandomForestClassifier(n_jobs=-1),
     linear_model.LogisticRegression(n_jobs=-1),
     KNeighborsClassifier(n_jobs=-1),
     SVC(),
     MLPClassifier()
    ]

# parameters voor de verschillende classifiers
grid_n_estimator = [50,100,200,400]
grid_learn = [1]
grid_seed = [0, 1]
grid_criterion = ['gini','entropy']
grid_max_depth = [None]
grid_bool = [True, False]

grid_min_samples_leaf=[1,5]
grid_min_samples_split=[2,4,10,16]

grid_param = [
            [{
            #ADA Boost
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }],
            [{
            #ET
            'n_estimators': grid_n_estimator, 
            'criterion': grid_criterion, 
            'max_depth': grid_max_depth, 
            'random_state': grid_seed,
             }],
            [{
            #GradientBoost
            'learning_rate': grid_learn, 
            'n_estimators': [100,200,300], 
            'max_depth': grid_max_depth,
            'random_state': grid_seed,
            'min_samples_leaf': [100,150],
            'max_features': [0.3, 0.1] 
             }],
            [{
            #RandomForest
            'n_estimators': grid_n_estimator,
            'criterion': grid_criterion, 
            'max_depth': grid_max_depth,
            'oob_score': [True],
            'random_state': grid_seed,
            'min_samples_leaf': grid_min_samples_leaf,
            'min_samples_split': grid_min_samples_split,
             }],   

            [{
            #LR
            'fit_intercept': grid_bool, 
            'C': [0.001,0.01,0.1,1,10,100,1000],
            'penalty': ['l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
            'random_state': grid_seed
             }],
    
            [{
            #KNN
            'n_neighbors': [1,2,3,4,5,6,7], 
            'weights': ['uniform', 'distance'], 
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }],
            
    
            [{
            #SVM
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1, 10, 100],
            'gamma': [ 0.001, 0.01, 0.1, 1], 
            'decision_function_shape': ['ovo', 'ovr'],
            'probability': [True],
            'random_state': grid_seed
             }],
                [{
            #NN
            'learning_rate':['constant', 'invscaling', 'adaptive'],
            'solver': ["lbfgs", "sgd", "adam"],
            "activation" : ["identity", "logistic", "tanh", "relu"],
            "learning_rate_init":[0.001,0.01,0.1],
            'random_state': grid_seed,
            "hidden_layer_sizes":[100,50,150,200],
             "max_iter": [200,400,800]
             }],  
        ]

# vind de beste parameters voor een classifier
def hyperparameters(estimator, data, y, parameters, cv_split):
    base_results = cross_val_score(estimator, data, y, cv=cv_split)
    estimator.fit(data, y)
    print('voor:')
    print(estimator.get_params())
    print('m',base_results.mean())
    print('s',base_results.std()*3)
    print('searching...')
    grid = GridSearchCV(estimator, parameters, cv=cv_split, verbose=1, n_jobs=-1)
    grid = grid.fit(data, y)
    estimator.set_params(**grid.best_params_)
    edited_results = cross_val_score(estimator, X, y, cv=cv_split)
    print('Na:')
    print(estimator.get_params())
    print('m',edited_results.mean())
    print('s',edited_results.std()*3)
    

# Duurt lang om te runnen! 
#hyperparameters(c[0], X, y, grid_param[0], cv_split)
#hyperparameters(c[1], X, y, grid_param[1], cv_split)
#hyperparameters(c[2], Xtree, y, grid_param[2], cv_split) #stock beste
#hyperparameters(c[3], Xtree, y, grid_param[3], cv_split) #{'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_split': 1e-07, 'min_samples_leaf': 1, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': -1, 'oob_score': True, 'random_state': 0, 'verbose': 0, 'warm_start': False}
#hyperparameters(c[4], X, y, grid_param[4], cv_split) #{'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'solver': 'newton-cg', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
#hyperparameters(c[4], Xtree, y, grid_param[4], cv_split) #{'C': 10, 'class_weight': None, 'dual': False, 'fit_intercept': False, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'solver': 'sag', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
#hyperparameters(c[5], X, y, grid_param[5], cv_split)
#hyperparameters(c[6], X, y, grid_param[6], cv_split)
#hyperparameters(c[7], Xtree, y, grid_param[7], cv_split) # {'activation': 'tanh', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 100, 'learning_rate': 'adaptive', 'learning_rate_init': 0.1, 'max_iter': 200, 'momentum': 0.9, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'sgd', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}
estimator = tree.DecisionTreeClassifier()

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

# Test welke combinatie van features het meest uitmaken voor de accuracy
def feature_elimination(estimator, X, y, cv_split):
    rfe = RFECV(estimator, step = 1, scoring = 'accuracy', cv=cv_split)
    rfe.fit(X,y)

    print(rfe.n_features_) #Het nummer van selected features with cross-validation.
    print(rfe.support_) #The mask of selected features.
    print(rfe.ranking_) #rank van de features, beste is 1
    print(rfe.grid_scores_ )#cross-validation scores: grid_scores_[i] = CV score of the i-th subset of features.
    #print(rfe.estimator_)

    print("Optimal number of features : %d" % rfe.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.show()

feature_elimination(estimator, X, y, cv_split)
feature_elimination(estimator, Xtree, y, cv_split)
binair = training.columns.values.tolist()
numeriek = treedata_training.columns.values.tolist()
print(binair, numeriek)

# feature importance
forest = ensemble.RandomForestClassifier(n_jobs=-1)

def fi(forest,X,y, namen):
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    lijst = []
    for f in range(X.shape[1]):
        lijst.append(namen[indices[f]])
        
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    for i in range(len(lijst)):
        if isinstance(lijst[i], int):
            lijst[i] = str(lijst[i])

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    sns.barplot(lijst, importances[indices])
    plt.xticks(rotation=80)
    plt.show()

fi(forest,X,y,binair)
fi(forest,Xtree,y,numeriek)
names = ['RandomForest','LogisticRegression','Neural Network','GradientBoosting']
namesv = ['RandomForest','LogisticRegression','Neural Network','GradientBoosting','VotingClassifier']

clfl = [ensemble.RandomForestClassifier(),
        linear_model.LogisticRegression(),
        MLPClassifier(max_iter=800),
        ensemble.GradientBoostingClassifier()]

clflHX = [ensemble.RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=400, random_state=1),
        linear_model.LogisticRegression(**{'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 0, 'solver': 'newton-cg', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}),
        MLPClassifier(**{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 50, 'learning_rate': 'adaptive', 'learning_rate_init': 0.01, 'max_iter': 800, 'momentum': 0.9, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 1, 'shuffle': True, 'solver': 'sgd', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}),
        ensemble.GradientBoostingClassifier()]#stock was best

clflHXtree = [ensemble.RandomForestClassifier(**{'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_split': 1e-07, 'min_samples_leaf': 1, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'oob_score': True, 'random_state': 0, 'verbose': 0, 'warm_start': False}),
        linear_model.LogisticRegression(**{'C': 10, 'class_weight': None, 'dual': False, 'fit_intercept': False, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 0, 'solver': 'sag', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}),
        MLPClassifier(**{'activation': 'tanh', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 100, 'learning_rate': 'adaptive', 'learning_rate_init': 0.1, 'max_iter': 800, 'momentum': 0.9, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'sgd', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}),
        ensemble.GradientBoostingClassifier()]#stock was best

# test de voting classifier van sklearn
def test_Voting(names, clfl, X, y, cv_split):
    means = []
    stds = []
    for name, clf in zip(names, clfl):
        score = cross_val_score(clf, X, y, cv=cv_split)
        means.append(score.mean())
        stds.append(score.std())
        print("Accuracy: %f (+/- %f) [%s]" % (score.mean(), score.std(), name))
    est = []
    for i in range(len(names)):
        est.append((names[i],clfl[i]))

    #voting classifier
    eclf = ensemble.VotingClassifier(estimators=est, voting='soft')#weights 
    vc_results = cross_val_score(eclf, X, y, cv=cv_split)
    print("Accuracy: %f (+/- %f) [%s]" % (vc_results.mean(), vc_results.std(), 'VotingClassifier'))
    means.append(vc_results.mean())
    stds.append(vc_results.std())
    return means, stds

def plot(means, stds, names):
    s = sns.barplot(names, means, yerr=stds)
    s.set_ylim(0.5, 1.0)
    plt.xticks(rotation=80)
    plt.ylabel('Cross Validation Accuracy')
    plt.show()
    
# zonder hyperparameters
print('Binaire data:')
a,b = test_Voting(names, clfl, X, y, cv_split)
plot(a, b, namesv)
# print('\n')
# print('Numerieke data:')
# a,b = test_Voting(names, clfl, Xtree, y, cv_split)
# plot(a, b, namesv)
# print('\n')
# # met hyperparameters
# print('Binaire data:')
# a,b = test_Voting(names, clflHX, X, y, cv_split)
# plot(a, b, namesv)
# print('\n')
# print('Numerieke data:')
# a,b = test_Voting(names, clflHXtree, Xtree, y, cv_split)
# plot(a, b, namesv)
names = ['RandomForest','LogisticRegression','Neural Network','StackingClassifier']

clfl = [ensemble.RandomForestClassifier(),
        linear_model.LogisticRegression(),
        MLPClassifier(max_iter=800),
        ensemble.GradientBoostingClassifier()] #lvl 2 classifier

clflHX = [ensemble.RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=400, random_state=1),
        linear_model.LogisticRegression(**{'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 0, 'solver': 'newton-cg', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}),
        MLPClassifier(**{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 50, 'learning_rate': 'adaptive', 'learning_rate_init': 0.01, 'max_iter': 800, 'momentum': 0.9, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 1, 'shuffle': True, 'solver': 'sgd', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}),
        ensemble.GradientBoostingClassifier()] #lvl 2 classifier

clflHXtree = [ensemble.RandomForestClassifier(**{'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_split': 1e-07, 'min_samples_leaf': 1, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'oob_score': True, 'random_state': 0, 'verbose': 0, 'warm_start': False}),
        linear_model.LogisticRegression(**{'C': 10, 'class_weight': None, 'dual': False, 'fit_intercept': False, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'solver': 'sag', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}),
        MLPClassifier(**{'activation': 'tanh', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': 100, 'learning_rate': 'adaptive', 'learning_rate_init': 0.1, 'max_iter': 800, 'momentum': 0.9, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 0, 'shuffle': True, 'solver': 'sgd', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}),
        ensemble.GradientBoostingClassifier()] #lvl 2 classifier

#traint een meta classifier
def stacking_test(names, clfs, X, y, cv_split):
    sclf = StackingClassifier(classifiers=clfs, use_probas=True, average_probas=False, meta_classifier=clfs[-1])
    means = []
    stds = []
    for clf, name in zip(clfs, names):
        scores = cross_val_score(clf, X, y, cv=cv_split, scoring='accuracy', n_jobs=-1)
        print("Accuracy: %f (+/- %f) [%s]" % (scores.mean(), scores.std(), name))
        means.append(scores.mean())
        stds.append(scores.std())
    return means, stds

# zonder hyperparameters
# print('Binaire data:')
# a,b = stacking_test(names, clfl, X, y, cv_split)
# plot(a, b, names)
# print('\n')
# print('Numerieke data:')
# a,b = stacking_test(names, clfl, Xtree, y, cv_split)
# plot(a, b, names)
# print('\n')
# # met hyperparameters
# print('Binaire data:')
# a,b = stacking_test(names, clflHX, X, y, cv_split)
# plot(a, b, names)
# print('\n')
# print('Numerieke data:')
# a,b = stacking_test(names, clflHXtree, Xtree, y, cv_split)
# plot(a, b, names)
# Maak csv file om voor de kaggle output
def make_kaggle_submission(estimator, X, y, test, filename):
    estimator.fit(X,y)
    Y_pred = estimator.predict(test)
    df_test['Survived'] = Y_pred

    print(filename, df_test[['PassengerId', 'Survived']].shape)
    df_test[['PassengerId', 'Survived']].to_csv(filename, index=False)

# zo hebben we onze submissions vooral gedaan:
#make_kaggle_submission(clf, X, y, test, 'normal.csv')
#make_kaggle_submission(clf, Xtree, y, testtree, 'tree.csv')
# gridsearch alternative, the TPOT.
from tpot import TPOTClassifier
from deap.tools import _hypervolume

def tpot_test(X, y, test, cv_split):
    tpot = TPOTClassifier(verbosity=2, max_time_mins=240, max_eval_time_mins=2, population_size=100, cv=cv_split, n_jobs=-1, periodic_checkpoint_folder='C:/Users/Michael/OneDrive/KI')
    tpot.fit(X,y)
    #make_kaggle_submission(tpot, Xtree, y, test, 'tpot.csv')
    clf = tpot.fitted_pipeline_
    print(clf)
    eval_dict = tpot.evaluated_individuals_
    #np.save('eval_dict.npy', eval_dict) #lukt niet op Kaggle
    tpot.export('tpot_exported_pipeline.py')
    return clf

# duurt 4uur!
#clf = tpot_test(Xtree, y, test, cv_split)
# run pipelines (van TPOT)
from tpot.builtins import ZeroCount
from tpot.builtins import CombineDFs
from tpot.builtins import OneHotEncoder
from tpot.builtins import StackingEstimator
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import RobustScaler, Normalizer, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from xgboost import XGBClassifier

# Score on the training set was:0.8376865671641791
exported_pipelineX = make_pipeline(
    ZeroCount(),
    FastICA(tol=0.4),
    Normalizer(norm="l2"),
    RobustScaler(),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.45, min_samples_leaf=5, min_samples_split=10, n_estimators=100)
)

# Generation 88 - Current best internal CV score: 0.8343283582089553 tree data
exported_pipelineY = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.35000000000000003, n_estimators=100), step=0.2),
    ZeroCount(),
    XGBClassifier(learning_rate=0.01, max_depth=2, min_child_weight=3, n_estimators=100, nthread=1, subsample=1.0)
)
# die op iteratie 100ish van gen88
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=6, max_features=0.2, min_samples_leaf=5, min_samples_split=7, n_estimators=100, subsample=0.6000000000000001)),
    StackingEstimator(estimator=GaussianNB()),
    ZeroCount(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.2, min_samples_leaf=4, min_samples_split=8, n_estimators=100, subsample=0.8500000000000001)
)
#of deze
exported_pipeline = make_pipeline(
RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=7, min_samples_split=11, n_estimators=100)
)
print(Xtree.shape)
score = cross_val_score(exported_pipeline, Xtree, y, cv=cv_split, scoring='accuracy', n_jobs=-1)
print("Accuracy: %f (+/- %f) [%s]" % (score.mean(), score.std(), 'tpot'))
exported_pipeline.fit(Xtree,y)
Y_pred = exported_pipeline.predict(testtree)
df_test['Survived'] = Y_pred

print('pipe.csv', df_test[['PassengerId', 'Survived']].shape)
#Maak csv file om upteloaden naar kaggle
df_test[['PassengerId', 'Survived']].to_csv('pipe.csv', index=False)

print("Accuracy: %f (+/- %f) [%s]" % (score.mean(), score.std(), 'tpot'))
#print het dict van de TPOTClassifier, dict is niet geupload naar kaggle want dat kan niet.
# eval_dict = np.load('eval_dictt.npy').item() #- de laatste t voor binair
# tpls = list(eval_dict.values())

# x = []
# y = [0]
# for i in range(len(tpls)):
#      x.append(i)
#      y.append(tpls[i]['internal_cv_score'])
# y.pop(0)
# plt.plot(x,y)
# plt.xlabel('Iteratie')
# plt.ylabel('internal_cv_score')
# plt.show()

# x = []
# y = [0]
# for i in range(len(tpls)):
#     if tpls[i]['internal_cv_score'] > y[-1]:
#         x.append(i)
#         y.append(tpls[i]['internal_cv_score'])
# y.pop(0)
# plt.plot(x,y,'o',x,y,'k')
# plt.xlabel('Iteratie')
# plt.ylabel('internal_cv_score')
# plt.show()


