import sklearn

print (sklearn.__version__)
# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression,Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.linear_model.stochastic_gradient import SGDClassifier



from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split , StratifiedKFold, cross_val_score

from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

from sklearn.metrics import roc_curve, auc



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



import re



# Configurações de visualização

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6

pd.set_option('display.max_rows', 20000)

pd.set_option('display.max_columns', 50)

import subprocess as sub

print(sub.check_output(["pwd"]).decode("utf8"))

print(sub.check_output(["ls", ".."]).decode("utf8"))
#Carga dos arquivos



#train_df = pd.read_csv("input/SERPRO/titanic-train.csv"titanic-train.csv")

#test_df = pd.read_csv("input/SERPRO/titanic-test.csv"titanic-test.csv")

train_df = pd.read_csv('/kaggle/input/serpro-titanic/titanic-train.csv')

test_df = pd.read_csv('/kaggle/input/serpro-titanic/titanic-test.csv')



#obtem os id para usar na submissão

persons = test_df.person

print(train_df.shape)

print(test_df.shape)

print(persons.shape)
# une os conjuntos de treino e teste

full = train_df.append(test_df, ignore_index=True)



# cria indices para separação entre os conjuntos, posteriormente

train_idx = len(train_df)

print(len(full))

test_idx = len(full) - len(test_df) # 1309 - 437 =872

print("train_idx="+str(train_idx))

print("test_idx="+str(test_idx) + '\n')



print(full.shape)

full.info()

def display_all(full):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(full)

           

display_all(full.describe(include='all').T)
full['survived'].value_counts()
# Fare não existentes: usada mediana da classe do passageiro

class_fares = dict(full.groupby('pclass')['fare'].median())



# cria coluna com mediana dos preços

full['fare_med'] = full['pclass'].apply(lambda x: class_fares[x])



# substitui as fares com os valores desta coluna

full['fare'].fillna(full['fare_med'], inplace=True, )

del full['fare_med']
#Transforma survived e sex para numérico

full['survived'] = full['survived'].map({'yes': 1, 'no': 0})

full['sex'] = full['sex'].map({'male': 1, 'female': 0}) 



full.head()
# extraindo os títulos de cada nome

full[ 'tratamento' ] = full[ 'name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

    
# Mapeando títulos em grupos categóricos

title_dic = {

                    "Capt":       "Tripulacao",

                    "Col":        "Tripulacao",

                    "Major":      "Tripulacao",

                    "Jonkheer":   "Nobreza",

                    "Don":        "Nobreza",

                    "Sir" :       "Nobreza",

                    "Dr":         "Tripulacao",

                    "Rev":        "Tripulacao",

                    "the Countess":"Nobreza",

                    "Dona":       "Nobreza",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Nobreza"



                    }



# mapeando os títulos

full['title'] = full.tratamento.map( title_dic )

full['title'].shape
# Cria variaveis numericas para cada title

titledummies = pd.get_dummies( full.title , prefix='title' )

full = pd.concat([ full, titledummies ], axis=1) 

full.shape
# Abordagem inicial, não utilizada!

# Ages não existentes: uso mediana das idades dos passageiros

#full['age'] = full['age'].fillna(full['age'].median())

#display_all(full.describe(include='all').T)
# Ve a média da idade de cada tipo de título

full.groupby('title')['age'].mean()

# Tratando valores ausentes de idade com a media da idade de cada title

full.loc[(full.age.isnull())&(full.title=='Master'),'age']=8

full.loc[(full.age.isnull())&(full.title=='Miss'),'age']=23

full.loc[(full.age.isnull())&(full.title=='Mr'),'age']=31

full.loc[(full.age.isnull())&(full.title=='Mrs'),'age']=36

full.loc[(full.age.isnull())&(full.title=='Tripulacao'),'age']=45

full.loc[(full.age.isnull())&(full.title=='Nobreza'),'age']=43

full['familysize'] = full['parch'] + full['sibsp']
# Cria variaveis numericas para embarked

embarked = pd.get_dummies( full.embarked , prefix='embarked' )
# Insere no dataset

full = pd.concat( [ full,embarked ] , axis=1)

full.head()

full.info()

full_bak = full.copy()

full_bak = full_bak.drop(['cabin','embarked','home_destination', 'name', 'person', 'ticket', 'title', 'tratamento' ], axis = 1)

full = full.drop(['cabin','embarked','home_destination', 'name', 'person', 'ticket', 'title', 'tratamento', 'parch', 'sibsp', 'title_Master', 'title_Miss',  'title_Mrs',  'title_Tripulacao',  'title_Nobreza',  'embarked_C',  'embarked_Q',  'embarked_S' ], axis = 1) 

full.head()
full.info()
# Separando dados de treino e teste para inicialmente explorar a importância de features, antes de drops

train = full_bak[ :train_idx]

test = full_bak[test_idx: ]



print(test_idx)

print(train.shape)

print(test.shape)
# converte survived novamente para int

train.survived = train.survived.astype(int)

print(train.survived.shape)
# cria X e y de treino



# Coloca survived como target

X_train= train.drop('survived', axis=1).values

y_train= train.survived.values



# obs: X_test não tem valores para "survived" : NaN

X_test = test.drop('survived', axis=1).values



print(X_train.shape, X_test.shape, y_train.shape)
#model = RandomForestClassifier(n_estimators=100)

model = GradientBoostingClassifier(n_estimators=250)

model.fit( X_train, y_train)

y_pred = model.predict( X_test )
features = pd.DataFrame()

features['feature'] = (train.drop('survived', axis=1)).columns

features['importance'] = model.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(5,5))
# cria dados de treino e teste

# lembrete: em full.bak temos mais features, que foram usadas na estimativa geral das suas importâncias



train = full[ :train_idx]

test = full[test_idx: ]



print(test_idx)

print(train.shape)

print(test.shape)
# converte survived novament epara int

train.survived = train.survived.astype(int)

print(train.survived.shape)
# cria X e y de treino



# Coloca survived como target

X_train= train.drop('survived', axis=1).values

y_train= train.survived.values



# obs: X_test não tem valores para "survived" : NaN

X_test = test.drop('survived', axis=1).values



print(X_train.shape, X_test.shape, y_train.shape)
# preparação dos modeloss

seed = 7

models = []



models.append(('Logistic Regression', LogisticRegression()))

models.append(('Decision Tree', DecisionTreeClassifier()))

models.append(('K-Nearest Neighbours (3)', KNeighborsClassifier(n_neighbors=3)))

models.append(('K-Nearest Neighbours (7)', KNeighborsClassifier(n_neighbors=7)))

models.append(('K-Nearest Neighbours (11)', KNeighborsClassifier(n_neighbors=11)))

models.append(('Random Forest', RandomForestClassifier()))

models.append(('Random Forest (10)', RandomForestClassifier(n_estimators=10)))

models.append(('Random Forest (100)', RandomForestClassifier(n_estimators=100)))

models.append(('Gaussian Naïve Bayes', GaussianNB()))

models.append(('Stochastic Gradient Decent (SGD)', SGDClassifier(max_iter=50)))

models.append(('Linear SVC', LinearSVC()))

models.append(('Perceptron (5)', Perceptron(max_iter=5)))

models.append(('Perceptron (10)', Perceptron(max_iter=10)))

models.append(('Perceptron (50)', Perceptron(max_iter=50)))



models.append(('Support Vector Machines (SVM)', SVC()))



models.append(('GradientBoostingClassifier (GBC)',GradientBoostingClassifier()))

models.append(('GradientBoostingClassifier (GBC20)',GradientBoostingClassifier(n_estimators=20)))

models.append(('GradientBoostingClassifier (GBC40)',GradientBoostingClassifier(n_estimators=40)))

models.append(('GradientBoostingClassifier (GBC70)',GradientBoostingClassifier(n_estimators=70)))

models.append(('GradientBoostingClassifier (GBC100)',GradientBoostingClassifier(n_estimators=100)))

models.append(('GradientBoostingClassifier (GBC150)',GradientBoostingClassifier(n_estimators=150)))

models.append(('GradientBoostingClassifier (GBC180)',GradientBoostingClassifier(n_estimators=180)))

models.append(('GradientBoostingClassifier (GBC200)',GradientBoostingClassifier(n_estimators=200)))

models.append(('GradientBoostingClassifier (GBC220)',GradientBoostingClassifier(n_estimators=220)))

models.append(('GradientBoostingClassifier (GBC250)',GradientBoostingClassifier(n_estimators=250)))

models.append(('GradientBoostingClassifier (GBC280)',GradientBoostingClassifier(n_estimators=280)))

models.append(('GradientBoostingClassifier (GBC300)',GradientBoostingClassifier(n_estimators=300)))



# avaliar um modelo por vez

results = []

names = []

scores = []

scoring = 'accuracy'



for name, model in models:

    kfold = sklearn.model_selection.KFold(n_splits=10, shuffle=False, random_state=seed)

    cv_results = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    scores.append(cv_results.mean())

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



resultsdf = pd.DataFrame({'Model': names, 'Score': scores})

print("\n")

resultsdf = resultsdf.sort_values(by='Score', ascending=False)

print(resultsdf)

# boxplot para comparação dos algoritmos

fig = plt.figure(figsize = (15, 8))

fig.suptitle('Comparação dos algorítmos')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.xticks(rotation=70)

plt.show()
train.describe()
# Seleciona um modelo para estimar a importância, usando as features selecionadas

#model = RandomForestClassifier(n_estimators=100)

model = GradientBoostingClassifier(n_estimators=220)

model.fit( X_train, y_train)

y_pred = model.predict( X_test )
features = pd.DataFrame()

features['feature'] = (train.drop('survived', axis=1)).columns

features['importance'] = model.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(5,5))
model =  GradientBoostingClassifier(n_estimators=250)

model.fit(X_train, y_train)



y_test = model.predict( X_test )

y_test.shape
submissiondf = pd.DataFrame( { 'person': persons , 'survived': y_test } )

submissiondf['survived'] = submissiondf['survived'].map({1: 'yes', 0: 'no'})



print(submissiondf.shape)

print(submissiondf.head())
submissiondf.to_csv( 'RG_titanic_sub_09a.csv' , index = False )