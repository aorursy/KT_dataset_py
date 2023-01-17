import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from plotnine import *

%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split as ttsplit

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Normalizer

from sklearn.compose import ColumnTransformer 

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate
import warnings

warnings.filterwarnings('ignore')
def matrix_corr(x):       #Matriz de correlação linear

    corr_matrix=x.corr()

    corr_matrix=corr_matrix['income'].sort_values(ascending=False)

    return corr_matrix



def histo_gg(dados,eixo_x,grupo):       #Histograma na forma ggplot

    fig=(ggplot(dados, aes(x=eixo_x,fill=grupo,group=grupo))

     +geom_histogram(binwidth=0.5, color='black')

     + coord_flip())

    return fig



def plotaErro(error):       #Plotar o gráfico do erro

    plt.figure(figsize=(12, 6))

    plt.plot(range(1, len(error)+1), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

    plt.title('Error Rate K Value')

    plt.xlabel('K Value')

    plt.ylabel('Mean Error')

    plt.show()

    return
new_names=["Id","age", "workclass", "fnlwgt", "education", "education_num", "martial_status", "occupation", 

        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "country",'income']



adult_train = pd.read_csv("../input/adult-pmr3508/train_data.csv",header=0,

        names=new_names,

        sep=r'\s*,\s*',

        engine='python',

        index_col=['Id'],

        na_values="?")

adult_train.head(10)
adult_train.isna().sum()   #verificando presença de nan
sns.set(font_scale=1)

sns.set_style(style='ticks')

adult_train.hist(bins=50,figsize=(10,10), color='black', histtype='bar');

plt.show();



#o histograma mostra como estão distribuídos os dados

#os atributos capita_gain/capital_loss estão mal distribuídos

#o atributo fnlwgt possui valores alto em comparação com os demais
adult_train.describe()

#o atributo fnlwgt é da ordem de 10^4
sns.set(font_scale=1)

fig=plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2,2,1)

cg0 = adult_train['capital_gain'].map(lambda x: x==0)

cg0.value_counts().plot(kind='pie',autopct='%1.0f%%');

plt.subplot(2,2,2)

cl0 = adult_train['capital_loss'].map(lambda x: x==0)

cl0.value_counts().plot(kind='pie',autopct='%1.0f%%');

#Notamos que capital_gain/capital_loss possuem poucos dados não nulos
adult_train.corr()

#visualizando alguma correlação linear

#Nada muito destacado
le=LabelEncoder()

categorical_cols=['income']

var1=adult_train.copy()

var1[categorical_cols] = var1[categorical_cols].apply(lambda col: le.fit_transform(col))

#var1['income']=var1.apply(LabelEncoder().fit_transform)

#Labe encoder no atributo income

var1
matrix_corr(var1)

#Podemos notar algumas corrrelações lineares

#fnlwgt possui baixa correlação
#Vamos tentar aumentar a correação de fnlwgt

var2=var1.copy()

std=Pipeline(steps=[('standard',StandardScaler())])

mmt=Pipeline(steps=[('minmax',MinMaxScaler())])

ct = ColumnTransformer([("std",std, ['fnlwgt'])])

var2['fnlwgt']= ct.fit_transform(var2)

corr_matrix=matrix_corr(var2)

corr_matrix
var3=var1.copy()

ct = ColumnTransformer([("mmt",mmt, ['fnlwgt'])])

var3['fnlwgt']= ct.fit_transform(var3)

corr_matrix=matrix_corr(var3)

corr_matrix
var4=var1.copy()

ct = ColumnTransformer([("norma",Normalizer(), ['fnlwgt'])])

var3['fnlwgt']= ct.fit_transform(var4)

corr_matrix=matrix_corr(var4)

corr_matrix
adult_train=adult_train.drop(['fnlwgt'], axis=1)

#removendo fnlwgt
histo_gg(adult_train,'relationship','income')
histo_gg(adult_train,'occupation','income')
histo_gg(adult_train,'country','income')
sns.set(font_scale=1)

coutry = adult_train['country'].map(lambda x: x=='United-States')

coutry.value_counts().plot(kind='pie',autopct='%1.0f%%');
histo_gg(adult_train,'martial_status','income')
histo_gg(adult_train,'workclass','income')
histo_gg(adult_train,'race','income')
x=adult_train.copy()

coluns_name=x.columns

x.head(10)
x=x.drop(['education','country','workclass','occupation','race'], axis=1)
x

x.isna().sum()
categorical_cols=['relationship','sex','martial_status','income']

var=x.copy()

var[categorical_cols] = var[categorical_cols].apply(lambda col: le.fit_transform(col))

matrix_corr(var)
#Criando dataset de treino/test

X=adult_train.copy()

Y=pd.DataFrame(X['income'])

X=var.drop(['income'],axis=1)
xtrain, xtest, ytrain, ytest = ttsplit(X, Y, test_size = 0.25, stratify = Y, random_state=42)

xtrain
error = []

for i in range(1, 20):

    knn = KNeighborsClassifier(n_neighbors=i,n_jobs=-1,algorithm='kd_tree')

    knn.fit(xtrain, ytrain)

    pred_i = knn.predict(xtest)

    pred_i= pred_i.reshape(len(pred_i),1)

    error.append(np.mean(pred_i != ytest))

plotaErro(error)
model = KNeighborsClassifier(n_neighbors=14,n_jobs=-1,algorithm='kd_tree')

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(model, X=xtrain, y=ytrain, cv=kfold)

print("Average accuracy: %f (%f)" %(results['test_score'].mean(), results['test_score'].std()))
scores = cross_val_score(knn, xtrain, ytrain, cv=10)

scores.mean()
new_names=["Id","age", "workclass", "fnlwgt", "education", "education_num", "martial_status", "occupation", 

        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "country"]



df = pd.read_csv("../input/adult-pmr3508/test_data.csv",header=0,

        names=new_names,

        sep=r'\s*,\s*',

        engine='python',

       index_col=['Id'],

        na_values="?")

df.head(10)
df_test=df
dropcol=['fnlwgt','race','education','country','workclass','occupation']

df_test=df_test.drop(columns=dropcol)
categorical_cols=['relationship','sex','martial_status']

df_test[categorical_cols] = df_test[categorical_cols].apply(lambda col: le.fit_transform(col))

df_test.head(10)
ypred=knn.predict(df_test)
submission = pd.DataFrame()

submission[0] = df.index

submission[1] = ypred

submission.columns = ['Id','income']

submission['income'].value_counts()
submission.to_csv('submission.csv',index = False)