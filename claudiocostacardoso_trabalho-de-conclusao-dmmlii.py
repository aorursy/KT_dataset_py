# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.head()
# Verificando os tipos e os valores nulos

df.info()
# Estatísticas descritivas para as principais componentes do risco de credito

df_d= df[['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']]

df_d.describe()
# Importando a biblioteca gráfica matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
# Histogramas das variávis explicativa numericas 

numeric_feats = [c for c in df.columns if df[c].dtype != 'object'and c not in ['BAD']]

df_numeric_feats = df[numeric_feats]



df_numeric_feats.hist(figsize=(20,8), bins=30)
# BoxPlot das variavéis explicativas por condição do emprestimo

plt.figure(figsize=(16,16))

c = 1

for i in df_numeric_feats.columns:

    if c < len(df_numeric_feats.columns):

        plt.subplot(3,3,c)

        sns.boxplot(x='BAD' , y= i, data=df)

        c+=1

    else:

        sns.boxplot(x='BAD' , y= i, data=df)
# BoxPlot da dsitribuição do tipo de trabalho em função das variáveis explicativas e condição do cliente(BAD)



plt.figure(figsize=(16,16))

c = 1

for i in df_numeric_feats.columns:

    if c < len(df_numeric_feats.columns):

        plt.subplot(3,3,c)

        sns.boxplot(x='JOB' , y= i, data=df)

        c+=1

    else:

        sns.boxplot(x='JOB' , y= i, data=df)
# Os cinco maiores volumse de emprestimo



df.nlargest(5,'LOAN')
# O Gráfico de barras abaixo, avalia o total de registros por emprestimo (BAD)

df["BAD"].value_counts().plot.bar(title='BAD')
# Avaliando os empresitimos com relacao a profisão do pagador



JOB=pd.crosstab(df['JOB'],df['BAD'])

JOB.div(JOB.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Tipos de Ocupação e Condição do Empréstimo', figsize=(8,4))
# Avaliando os empresitimos com relacao a consolidação da dívida (DebtCon) e melhoria da casa (HomeImp)

JOB=pd.crosstab(df['REASON'],df['BAD'])

JOB.div(JOB.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Tipos de Ocupação e Condição do Empréstimo', figsize=(8,4))
# Correlação entre as 10 variáveis determinantes para o risco de credito.



df_c= df[['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']]

df_c.corr()



# Plotando a correlação

# Aumentando a area do gráfico

f, ax =plt.subplots(figsize=(9,5))

sns.heatmap(df_c.corr(), annot=True, fmt='.2f', linecolor='black', lw=.7, ax=ax, cmap=plt.cm.Blues)

plt.title('Correlação entre as 10 variáveis determinantes para o risco de credito')

plt.show()
# Verificando o percental de valores missing por variável

def missing_values(data_frame):

    total = data_frame.isnull().count()

    missing = data_frame.isnull().sum()

    missing_percent = missing/total * 100

    display(missing_percent)

    

missing_values(df)
df = df.dropna(thresh=df.shape[1]/2, axis=0)

df.info()
# Verificando o percental de valores missing após a eliminação dos registros que tem mais da metada das colunas nulas

def missing_values(data_frame):

    total = data_frame.isnull().count()

    missing = data_frame.isnull().sum()

    missing_percent = missing/total * 100

    display(missing_percent)

    

missing_values(df)
# Tratando colunas categóricas

for col in df.select_dtypes(include='object').columns:

    if df[col].isna().sum() > 0:

         df[col].fillna(df[col].mode()[0], inplace=True)   
# Tratando colunas numéricas

for col in df.select_dtypes(exclude='object').columns:

    if df[col].isna().sum() > 0:

        df[col].fillna(df[col].median(), inplace=True) 
df.sample(5)
df.info()
def showBalance(df, col):

    for c in col:

        print('Distribuição da Coluna: ', c,'\n',df[c].value_counts(normalize=True),'\n')

    else:

       pass

        

showBalance(df, col=['REASON','JOB','BAD'])
# Importando a biblioteca resample

from sklearn.utils import resample



# Função para realizar balanceamento

def balance(df, col):

    df_maior = df[df[col] == df[col].mode()[0]]

    df_menor = df[df[col] != df[col].mode()[0]]

 

    # Upsample da menor classe

    df_menor_upsampled = resample(df_menor, 

                                  replace=True,     

                                  n_samples=df_maior.shape[0],

                                  random_state=42) 

 

    # Combinar as classe predominante com a menor classe aumentada

    df_upsample = pd.concat([df_maior, df_menor_upsampled])



    # Display new class counts

    print('Contagem de registros')

    print(df_upsample['BAD'].value_counts())

    print('\nDistribuição dos registros')

    print(df_upsample['BAD'].value_counts(normalize=True))



    return df_upsample

    

df_upsample = balance(df, 'BAD')

print('\n')

showBalance(df_upsample, col=['REASON','JOB','BAD'])
# Criando dummys para todas as colunas

df_upsample = pd.get_dummies(df_upsample, columns=['REASON', 'JOB'])
# Verificando os tipos dos dados e os tamanhos

df_upsample.info()
# Olhando os dados aleatoriamente

df_upsample.sample(5)
# Separando o dataframe



from sklearn.model_selection import train_test_split



# Separando treino e teste

# Em 70% para treino e 30% para teste

train, test = train_test_split(df_upsample, test_size=0.20, random_state=42)



# Não vamos mais usar o dataset de validação



train.shape, test.shape
# definindo colunas de entrada

feats = [c for c in df_upsample.columns if c not in ['BAD']]
# # Trabalhando com Random Forest e oob_score

# Importando o modelo

from sklearn.ensemble import RandomForestClassifier



# Instanciar o modelo

rf = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=200, random_state=42, criterion='gini')



# Usar o cross validation

from sklearn.model_selection import cross_val_score



scores = cross_val_score(rf, train[feats], train['BAD'], n_jobs=-1, cv=5)



scores, scores.mean()
# Importação bibliotecas

# Importação GridSearchCV.

from sklearn.model_selection import GridSearchCV



#instanciando o modelo

rf2=RandomForestClassifier(n_jobs=-1)



#setando parametros para o gridSearchCV

param_dict = {'n_estimators':[100,400,800,1000],  'criterion': ['gini','entropy'] }



grid2 = GridSearchCV(rf2, param_dict, cv=10)



# Usar o cross validation

from sklearn.model_selection import cross_val_score



scores = cross_val_score(rf2, train[feats], train['BAD'], n_jobs=-1, cv=5)



scores, scores.mean()
# Trabalhando com GBM

from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)
# Usar o cross validation

from sklearn.model_selection import cross_val_score



scores = cross_val_score(gbm, train[feats], train['BAD'], n_jobs=-1, cv=5)



scores, scores.mean()
# Trabalhando com XGBoost



# Importar o modelo

from xgboost import XGBClassifier



# Instanciar o modelo

xgb = XGBClassifier(n_jobs=-1, oob_score=True, n_estimators=200, random_state=42, learning_rate=0.05)
# Usando o cross validation

scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5)



scores, scores.mean()
# Treinar o modelo,usando o Random Forest para treinamento e predição

rf.fit(train[feats], train['BAD'])



# Fazendo predições

preds = rf.predict(test[feats])
# Avaliando o desempenho do modelo (Accuracy)



# Importando a metrica

from sklearn.metrics import accuracy_score



accuracy_score(test['BAD'], preds)
# matriz de cofussão

from sklearn.metrics import confusion_matrix

confusion_matrix(test['BAD'], rf.predict(test[feats]))
cm = confusion_matrix(test['BAD'], rf.predict(test[feats]))



fig, ax = plt.subplots(figsize=(7,6))

sns.heatmap(cm, annot=True, ax=ax, fmt='.0f',cmap=plt.cm.Blues); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('Real labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['1', '0'])

ax.yaxis.set_ticklabels(['1', '0']);

plt.show()



sens = cm[0,0] / (cm[0,0] + cm[1,0])

esp = cm[1,1] / (cm[0,1] + cm[1,1])

efi = (sens+esp)/2



print('Sensibilidade: ', round(sens,4))

print('Especificidade: ', round(esp,4))

print('Esficiência: ', round(efi,4))

# Avaliando a importancia de cada coluna de entrada

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()


# Treinar o modelo,usando o Random Forest para treinamento e predição

rf2.fit(train[feats], train['BAD'])



# Fazendo predições

preds = rf2.predict(test[feats])



accuracy_score(test['BAD'], preds)
# matriz de cofussão

from sklearn.metrics import confusion_matrix

confusion_matrix(test['BAD'], rf2.predict(test[feats]))
cm = confusion_matrix(test['BAD'], rf2.predict(test[feats]))



fig, ax = plt.subplots(figsize=(7,6))

sns.heatmap(cm, annot=True, ax=ax, fmt='.0f',cmap=plt.cm.Blues); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('Real labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['1', '0'])

ax.yaxis.set_ticklabels(['1', '0']);

plt.show()



sens = cm[0,0] / (cm[0,0] + cm[1,0])

esp = cm[1,1] / (cm[0,1] + cm[1,1])

efi = (sens+esp)/2



print('Sensibilidade: ', round(sens,4))

print('Especificidade: ', round(esp,4))

print('Esficiência: ', round(efi,4))

# Avaliando a importancia de cada coluna de entrada

pd.Series(rf2.feature_importances_, index=feats).sort_values().plot.barh()