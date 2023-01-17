import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder



adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        engine='python',

        sep=r'\s*,\s*',

        na_values="?")

adult.shape
adult.head()
adult.info()
adult.isnull().sum()
print("workclass")

print(adult['workclass'].describe())

print()



print("occupation")

print(adult['occupation'].describe())

print()



print("native.country")

print(adult['native.country'].describe())

print()
print(adult['workclass'].value_counts())

print()



print(adult['native.country'].value_counts())

print()

      

print(adult['occupation'].value_counts())

print()
value = adult['workclass'].describe().top

adult['workclass'] = adult['workclass'].fillna(value)



value = adult['occupation'].describe().top

adult['occupation'] = adult['occupation'].fillna(value)
#Para facilitar a visualização dos gráficos, mudaremos momentaneamente a variável income para uma numérica

adult['income']=adult['income'].map({'<=50K':0, '>50K':1})
#Contagem de quantos ganham <=50K e >50K

sns.countplot(adult['income'])

plt.ylabel('Quantidade')
#Relação entre o quanto ganha e o sexo

g = sns.barplot(x='sex',y='income',data=adult)

plt.ylabel('Probabilidade de ganhar >50k')

plt.xlabel('Sexo')

plt.show()
#Matriz de correlação entre as variáveis numéricas

variaveis_numericas = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']

g = sns.heatmap(adult[variaveis_numericas].corr(),annot=True, fmt='.2f', cmap='coolwarm')

plt.show(g)
#Relação entre o quanto se ganha e o nível de educação

g = sns.catplot(x='education.num',y='income',data=adult,kind='bar',height=6,palette='muted')

g.despine(left=True)

plt.ylabel('Probabilidade de ganhar >50K')

plt.show(g)
#Idade das pessoas avaliadas pelo censo 

plt.figure(figsize=(13,7))

sns.distplot(adult['age'], bins=70)

plt.ylabel('Quantidade')

plt.xlabel('Idade')
#Quantidade de horas trabalhadas

plt.figure(figsize=(13,7))

adult['hours.per.week'].hist()

plt.xlabel('Horas por semana')

plt.ylabel('Quantidade')
#Relação entre o quanto se ganha e a etnia

plt.figure(figsize=(17,10))

b=sns.countplot(x='race',hue='income',data=adult)
#Juntando as variáveis capital gain e capital loss de modo a evitar sobre ajuste

adult['capital']=adult['capital.gain']-adult['capital.loss']

adult['capital'].describe()
#Relação entre o esatado civil e quanto se ganha 

plt.figure(figsize=(17,10))

b=sns.countplot(x='marital.status',hue='income',data=adult)
#Relação entre o relação familiar e quanto se ganha 

plt.figure(figsize=(17,10))

b=sns.countplot(x='relationship',hue='income',data=adult)
#Revertendo o processo usado pra analisar a base

adult['income']=adult['income'].map({0:'<=50K', 1:'>50K'})
#Removendo as variáveis que não serão utilizadas e tratando das que serão

copy=adult['income']

adult=adult.drop(['native.country','capital.gain','capital.loss','education','fnlwgt','Id','income'],axis=1)

adult['income']=copy



colums_to_encode=['workclass','marital.status','sex','race','relationship','occupation']

for feature in colums_to_encode:

    le=preprocessing.LabelEncoder()

    adult[feature]=le.fit_transform(adult[feature])
x_adult=adult[['age','workclass','education.num','marital.status','sex','race','relationship','occupation','hours.per.week','capital']]

y_adult=adult.income
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_adult,y_adult, test_size=0.3 , random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

pca=PCA()

x_train = pca.fit_transform(x_train)
pca.explained_variance_ratio_
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

accuracy_score(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier

import sklearn.ensemble

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
#Tentando uma Árvore qualquer e uma floresta qualquer

cv_res=cross_val_score(DecisionTreeClassifier(),x_train,y_train,cv=10)

print('Árvore',cv_res.mean())

cv_res=cross_val_score(sklearn.ensemble.RandomForestClassifier(n_estimators=100),x_train,y_train,cv=10)

print('Floresta aleátória', cv_res.mean())
#Busca gridge demorando mais que o normal





#clf=sklearn.ensemble.RandomForestClassifier()

#kf=KFold(n_splits=3)

#max_features=np.array([2,3,4,5])

#n_estimators=np.array([25,50,100,150])

#min_samples_leaf=np.array([50,75,100])

#param_grid=dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf)

#grid=GridSearchCV(estimator=clf,param_grid=param_grid,cv=kf)

#gres=grid.fit(x_train,y_train)

#print("Melhor",gres.best_score_)

#print("Parâmetros",gres.best_params_)
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(solver="adam", alpha=0.0001, hidden_layer_sizes=(5,),

                   random_state=1, learning_rate='constant', learning_rate_init=0.01,

                   max_iter=50, activation='logistic', momentum=0.9,verbose=True,

                   tol=0.0001)

cv_res=cross_val_score(mlp,x_train,y_train,cv=10)

cv_res.mean()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 



C = pd.read_csv("/kaggle/input/atividade3/train.csv",

        engine='python',

        sep=r'\s*,\s*',

        na_values="?")
C.head()
#Criando variavel para guardar o target e retirando Id

y_C=C.median_house_value

C=C.drop(['Id','median_house_value'],axis=1)
C.info()
C["persons/bedrooms"] = C["population"]/C["total_bedrooms"]

C["rooms/households"] = C["total_rooms"]/C["households"]

C=C.drop(['population','total_bedrooms','total_rooms','households'],axis=1)
C.hist(bins=200, figsize=(25,20))
plt.figure(figsize=(17,10))

plt.xlabel('Pessoas por quarto')

C["persons/bedrooms"].hist(bins=200, range=(0,6))
plt.figure(figsize=(17,10))

plt.xlabel('Cômodos por casa')

C["rooms/households"].hist(bins=200, range=(0,10))
plt.figure(figsize=(10,10))

plt.title("Matriz de correlação")

sns.heatmap(C.corr(), annot=True, linewidths=0.2)
C['median_age'].describe()
plt.figure(figsize=(17,10))

plt.xlabel('Idade média')

C["median_age"].hist(bins=200, range=(0,53))
from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor

from sklearn import neural_network

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
LR = Ridge()

cv_res=cross_val_score(LR,C,y_C,cv=10)

cv_res.mean()
LR = Lasso()

cv_res=cross_val_score(LR,C,y_C,cv=10)

cv_res.mean()
neural_net = neural_network.MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',learning_rate='adaptive', max_iter=800,learning_rate_init=0.01, warm_start = True, alpha=0.01)

cv_res=cross_val_score(neural_net,C,y_C,cv=10)

cv_res.mean()