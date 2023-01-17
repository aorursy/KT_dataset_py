##bibliotecas 

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import scikitplot as skplt

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score

from sklearn.utils import resample

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

## visualização

pd.set_option('max_columns', 140)

pd.set_option('max_colwidth', 5000)

pd.set_option('display.max_rows', 140)

%matplotlib inline

plt.rcParams['figure.figsize'] = (12,8)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## carregando e juntando datasets

train = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')

codebook = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/codebook.csv')



df = train.append(test)



train.shape,test.shape, df.shape

## procurando valores com grande quantidade de nulos

df.isna().sum().sort_values(ascending=False).head(10)
## verificando variáveis que apresentaram muitos nulos e não são a Target

print(codebook[codebook['Variable name']=='rez_esc'])

print('----------------------------------')

print(codebook[codebook['Variable name']=='v18q1'])

print('----------------------------------')

print(codebook[codebook['Variable name']=='v2a1'])

print('----------------------------------')

print(codebook[codebook['Variable name']=='meaneduc'])

print('----------------------------------')

print(codebook[codebook['Variable name']=='SQBmeaned'])
df.v18q1.value_counts()
df.rez_esc.value_counts()
df.v2a1.value_counts()
df.meaneduc.value_counts()
df.SQBmeaned.value_counts()
#realizando a imputação e dropando coluna

df['v2a1'].fillna(-1, inplace=True)

df['v18q1'].fillna(0, inplace=True)

df['SQBmeaned'].fillna(-1, inplace=True)

df['rez_esc'].fillna(-1, inplace=True)

df.meaneduc.fillna(df.meaneduc.median(), inplace=True)

## realizando o drop de outras colunas que assim como SQBmeaned são provenientes de outras

# df.drop(['SQBescolari'], axis=1, inplace=True)

# df.drop(['SQBage'], axis=1, inplace=True)

# df.drop(['SQBhogar_total'], axis=1, inplace=True)

# df.drop(['SQBedjefe'], axis=1, inplace=True)

# df.drop(['SQBhogar_nin'], axis=1, inplace=True)

# df.drop(['SQBovercrowding'], axis=1, inplace=True)

# df.drop(['SQBdependency'], axis=1, inplace=True)

# df.drop(['agesq'], axis=1, inplace=True)

df.isna().sum().sort_values(ascending=False).head(10)
##verificando se existem colunas do tipo objeto

df.select_dtypes('object').describe()
df.dependency.value_counts().head(5)
df.edjefe.value_counts().head(5)
df.edjefa.value_counts().head(5)
#realizando um replace

valores_replace = {'yes': 1, 'no': 0}

#df.drop(['Id'], axis=1, inplace=True)

#df.drop(['idhogar'], axis=1, inplace=True)

df['dependency'] = df['dependency'].replace(valores_replace).astype(float)

df['edjefe'] = df['edjefe'].replace(valores_replace).astype(int)

df['edjefa'] = df['edjefa'].replace(valores_replace).astype(int)
df.select_dtypes('object')
## verificando a correlação entre as variáveis e a Target



df[df['Target'].notnull()].corr()['Target'].sort_values(ascending=False)
## após a criação das colunas ocorreu uma leve melhoria no score, as colunas comentadas foram testadas e diminuiram o score geral.



#df["telperpessoa"]=df["qmobilephone"]/df["tamviv"]

df["m2perpessoa"]=df["tamhog"]/df["tamviv"]

# df['tabletsperpessoa'] = df['v18q1'] / df['tamviv']

# df['roomsperpessoa'] = df['rooms'] / df['tamviv']

df['rentperpessoa'] = df['v2a1'] / df['tamviv']

# df['hsizeperpessoa'] = df['hhsize'] / df['tamviv']



## média de aluguel paga por quantidade de quartos 

df[df['tipovivi3']==1].groupby(['rooms'])['v2a1'].mean().plot(kind='bar')
#df['v2a1'].value_counts.plot.bar()
## quantidade de moradias com teto (1) e sem teto (0)

df.groupby(['cielorazo'])['idhogar'].nunique().plot(kind='bar')
## porcentagem de moradias sem teto

print(round(df[df['cielorazo']==0]['idhogar'].nunique()/df[df['cielorazo']==1]['idhogar'].nunique()*100,2),'%')
#abastaguano

plt.figure(figsize=(15,9))

fig, axes = plt.subplots(nrows=2, ncols=2)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

fig.suptitle('0 (possuem) 1 (não possuem)')



ax=axes[0,0].title.set_text('Distribuição água.')

ax=axes[0,1].title.set_text('Distribuição energia.')

ax=axes[1,0].title.set_text('Possui Sanitario.')

ax=axes[1,1].title.set_text('Possui piso.')



df.groupby(['abastaguano'])['idhogar'].nunique().plot(kind='bar',ax=axes[0,0])

df.groupby(['noelec'])['idhogar'].nunique().plot(kind='bar',ax=axes[0,1])

df.groupby(['pisonotiene'])['idhogar'].nunique().plot(kind='bar',ax=axes[1,0])

df.groupby(['sanitario1'])['idhogar'].nunique().plot(kind='bar',ax=axes[1,1])



quantidade = mpatches.Patch(label='Quantidade')

plt.legend(fancybox=True, framealpha=1,handles=[quantidade] ,shadow=True, borderpad=1 )



plt.show()


print('** Métodos de se jogar lixo fora **')

print('Tanker truck:',df[df['elimbasu1']==1].groupby(['elimbasu1'])['idhogar'].nunique().iloc[0])

print('Botan hollow or buried:',df[df['elimbasu2']==1].groupby(['elimbasu2'])['idhogar'].nunique().iloc[0])

print('Burning:',df[df['elimbasu3']==1].groupby(['elimbasu3'])['idhogar'].nunique().iloc[0])

print('Throwing in an unoccupied space:',df[df['elimbasu4']==1].groupby(['elimbasu4'])['idhogar'].nunique().iloc[0])

print('Throwing in river,  creek or sea:',df[df['elimbasu5']==1].groupby(['elimbasu5'])['idhogar'].nunique().iloc[0])

print('Other:',df[df['elimbasu6']==1].groupby(['elimbasu6'])['idhogar'].nunique().iloc[0])

## zonas de moradia

rural = round(df[df['area2']==1].groupby(['area2'])['idhogar'].nunique().iloc[0]/df[df['area1']==1].groupby(['area1'])['idhogar'].nunique().iloc[0],2)

urbana = 1-rural

plt.figure(figsize=(15,9))

labels = [r'Urbana('+str(urbana)+')', r'Rural ('+str(rural)+')']

sizes = [88.4, 10.6, 0.7, 0.3]

colors = ['orange', 'blue']

patches, texts = plt.pie(sizes, colors=colors, startangle=90)

plt.legend(patches, labels, loc="best")

plt.pie(df.groupby('area1')['idhogar'].nunique())

plt.show()
## contando a quantidade de lideres de familia masculinos e femininos

print('quantidade de lideres de família homens:',df[(df['parentesco1']==1) & (df['male']==1)].shape[0])

print('quantidade de lideres de família mulheres:',df[(df['parentesco1']==1) & (df['female']==1)].shape[0])
## verificando a distribuição da classe a ser predita

df[df['Target'].notnull()].groupby(['Target'])['idhogar'].nunique().plot(kind='bar', title='1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households')
## separando datasets em treino e teste para aplicação no modelo

feats = [c for c in df.columns if c not in ['Id', 'idhogar', 'Target']]

train, test = df[~df['Target'].isnull()], df[df['Target'].isnull()]

train.shape, test.shape
#Criando vários parâmetros  afim de buscar o melhor caso

param_grid = {

    'criterion' : ['gini', 'entropy'],

    'class_weight' : ['balanced','balanced_subsample'],

     'n_estimators': [100, 200, 300, 400]

 }

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                         cv = 3, n_jobs = -1, verbose = 2)
## comentado para não rodar toda vez já que demora mais de 10 minutos

# grid_search.fit(train[feats], train['Target'])

# grid_search.best_params_
## mesmo após aplicar o grid_search, o resultado ainda não ficou melhor que apenas com o n_estimators 200, por isso foi mantido ele.

rf = RandomForestClassifier(n_jobs=-1, n_estimators=200 ,random_state=42)

rf.fit(train[feats], train['Target'])
## realizando a predição de valores

test['Target'] = rf.predict(test[feats]).astype(int)

## Demonstrando a importância das variáveis e seus nomes 

print(rf.feature_importances_)

print(train.columns)
#distribuição da variável 

test['Target'].value_counts(normalize=True)
pd.Series(rf.feature_importances_, index=feats).sort_values()
skplt.metrics.plot_confusion_matrix(train['Target'], rf.predict(train[feats])) ## Matriz de confusão

accuracy_score(train['Target'], rf.predict(train[feats])) ## score do modelo
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission1.csv', index=False)
## pegando chefe de família

heads2 = train[train['parentesco1'] == 1]

## treinando modelo

# max_depth : profundidade de árvore, utilizando none para ir expandido até todas estarem puras

# n_jobs : quantidade de tarefas rodando em paralelo

# n_estimators : quantidade de árvores na floresta, 700 foi o melhor valor entre os testados

# min_impurity_decrease : o nó vai se dividir se a impureza da divisão for maior ou igual ao valor utilizado

# min_samples_leaf : quantidade minima de amostras para ser considerado um nó folha

# verbose : demonstrar menos informações na hora de rodar o modelo

# class_weight : Foi utilizado balanced, balanceando o peso das classes

#

#

rf4 = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')

rf4.fit(heads2[feats], heads2['Target'])

## predizendo

test['Target'] = rf4.predict(test[feats]).astype(int)

# criando arquivo

test[['Id', 'Target']].to_csv('submission.csv', index=False)
skplt.metrics.plot_confusion_matrix(heads2['Target'], rf4.predict(heads2[feats])) ## Matriz de confusão

accuracy_score(heads2['Target'], rf4.predict(heads2[feats])) ## Score do modelo
xgb = XGBClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')

xgb.fit(heads2[feats], heads2['Target'])

test['Target'] = xgb.predict(test[feats]).astype(int)

test[['Id', 'Target']].to_csv('submission2.csv', index=False)

skplt.metrics.plot_confusion_matrix(heads2['Target'], xgb.predict(heads2[feats])) ## Matriz de confusão

accuracy_score(heads2['Target'], xgb.predict(heads2[feats])) ## Score do modelo
## learning_rate : controla quanto vai ser a contribuição para o novo modelo, utilizando o atual

abc = AdaBoostClassifier(random_state=42, n_estimators=700,learning_rate=1.0)

abc.fit(heads2[feats], heads2['Target'])

test['Target'] = abc.predict(test[feats]).astype(int)

test[['Id', 'Target']].to_csv('submission3.csv', index=False)
skplt.metrics.plot_confusion_matrix(heads2['Target'], abc.predict(heads2[feats])) ## Matriz de confusão

accuracy_score(heads2['Target'], abc.predict(heads2[feats])) ## Score do modelo
# Criando um modelo de RF Classifier e usando o Cross Validation

rfc = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')

scores = cross_val_score(rfc, heads2[feats], heads2['Target'], cv=5, n_jobs=-1)

scores
scores.mean()
train['Target'].value_counts()

df_1 = train[train['Target'] == 1]

df_2 = train[train['Target'] == 2]

df_3 = train[train['Target'] == 3]

df_4 = train[train['Target'] == 4]

df_1.shape,df_2.shape,df_3.shape,df_4.shape,
## aumentando as classes menores até igualar a quantidade com df_4



df_1 = resample(df_1, 

                       replace=True,

                       n_samples=len(df_4),

                       random_state=42)



df_2 = resample(df_2, 

                       replace=True,

                       n_samples=len(df_4),

                       random_state=42)



df_3 = resample(df_3, 

                       replace=True,

                       n_samples=len(df_4),

                       random_state=42)
df_1.shape,df_2.shape,df_3.shape,df_4.shape,
## juntando as 4 tabelas

train = pd.concat([df_1, df_2, df_3, df_4])
train.shape
## aplicando o mesmo treinamento que alcançou o maior score

heads2 = train[train['parentesco1'] == 1]

rf4 = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')

rf4.fit(heads2[feats], heads2['Target'])

## predizendo

test['Target'] = rf4.predict(test[feats]).astype(int)

# criando arquivo

test[['Id', 'Target']].to_csv('submission4.csv', index=False)
skplt.metrics.plot_confusion_matrix(heads2['Target'], rf4.predict(heads2[feats])) ## Matriz de confusão

accuracy_score(heads2['Target'], rf4.predict(heads2[feats])) ## Score do modelo
## verificando a melhoria do primeiro modelo do testo, após aplicação de balanceamento 

rf4 = RandomForestClassifier(n_jobs=-1, n_estimators=200 ,random_state=42)

rf4.fit(heads2[feats], heads2['Target'])

## predizendo

test['Target'] = rf4.predict(test[feats]).astype(int)

# criando arquivo

test[['Id', 'Target']].to_csv('submission5.csv', index=False)
skplt.metrics.plot_confusion_matrix(heads2['Target'], rf4.predict(heads2[feats])) ## Matriz de confusão

accuracy_score(heads2['Target'], rf4.predict(heads2[feats])) ## Score do modelo