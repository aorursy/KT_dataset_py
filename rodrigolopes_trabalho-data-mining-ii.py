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
# Carregando o Arquivo



df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')



df.shape
# Visualizando qtde e tipos

df.info()
#Visualizando os valores do DataFrame

df.head()
#Empréstimos Adimplentes

df[df['BAD']==0].describe()
#Emprestimos Inadimplentes

df[df['BAD']==1].describe()
import matplotlib.pyplot as plt

import seaborn as sns

fig, axs = plt.subplots(1,2,figsize=(14,7))

sns.countplot(x='BAD',data=df,ax=axs[0])

axs[0].set_title("Frequência do Status de Pagamento")

df.BAD.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')

axs[1].set_title("Porcentagem de Pagamento")

plt.show()

#Tipo de emprego X Pagou emprestimo

sns.catplot(x='JOB', hue='BAD', data=df, kind='count')

#Motivo Emprestimo X Pagou emprestimo

sns.catplot(x='REASON', hue='BAD', data=df, kind='count')

# Relação PAGOU x VALOR EMPRESTIMO + REASON

sns.catplot(x='BAD', y='LOAN', hue='REASON', data=df, height=7, aspect=.8).set(title="Valor X Situação do Emprestimo e Razão")

sns.catplot(x='BAD', y='LOAN', hue='JOB', data=df, height=7, aspect=.8).set(title="Valor X Situação do Emprestimo e Trabalho")
# Relação do Trabalho X Tempo de Trabalho e Situação do Emprestimo

plt.figure(figsize=(15,5))

sns.violinplot(x='JOB', y='YOJ', hue='BAD',split=True, inner="quart", data=df)  

# Relação do Trabalho X Tempo do Emprestimo Mais Antigo e Situação do Emprestimo

plt.figure(figsize=(15,5))

sns.violinplot(x='JOB', y='CLAGE', hue='BAD',split=True, inner="quart",data=df)   

# Relação do Trabalho X Valor da Hipoteca e Situação do Emprestimo

plt.figure(figsize=(15,5))

sns.violinplot(x='JOB', y='MORTDUE', hue='BAD', split=True, inner="quart",data=df)

# Analisando campos Nulos

df.isna().sum()

#Analisando os registros com mais da metade de valores nulos das YOJ a DEBTINC

df[df.iloc[:,6:].isnull().all(axis=1)]
#Excluidno do dataFrame dados com 7 ou mais colunas com valores nulos

df = df.dropna(axis=0,thresh=df.shape[1]-6)

df[df.iloc[:,6:].isnull().all(axis=1)]
# Analisando campos Nulos

df.isna().sum(), df.shape

#Analisando colunas REASON do Tipo Object

df['REASON'].value_counts(),print("Nulos Campo REASON:", df['REASON'].isna().sum())

#Analisando colunas JOB do Tipo Object

df['JOB'].value_counts(),print("Nulos Campo JOB:", df['JOB'].isna().sum())

#Criando categoria Dummie para coluna REASON e JOB



dumies_reason=pd.get_dummies(df['REASON'],prefix='REASON')

df = df.merge(dumies_reason,left_index=True, right_index=True)

dumies_job=pd.get_dummies(df['JOB'],prefix='JOB')

df = df.merge(dumies_job,left_index=True, right_index=True)

df.head()
#preenchendo campos nulos que restaram com 0

df.fillna(0,inplace=True)
df.isna().sum(), df.shape
# Correlação das variáveis numéricas

plt.figure(figsize= (15, 15))



sns.heatmap(df.corr(), square=True, annot=True, linewidth=0.5)

# Dividindo o DataFrame

from sklearn.model_selection import train_test_split



# Treino e teste

train, test = train_test_split(df, test_size=0.199, random_state=42)



# Veificando o tanho dos DataFrames

train.shape, test.shape
# definindo colunas de entrada para a predição

feats = [c for c in df.columns if c not in ['BAD','JOB', 'REASON']]
# Bibliotecas RandomForest

from sklearn.ensemble import RandomForestClassifier

# Bibliotecas GBM

from sklearn.ensemble import GradientBoostingClassifier

# Trabalhando com XGBoost

from xgboost import XGBClassifier



#Validação do Modelo, Acurácia

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score



# importando a bilbioteca para plotar o gráfico de Matriz de Confusão

import scikitplot as skplt

#Trabalhando com Random Forest



rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, oob_score=True,max_depth=4, random_state=42)

rf.fit(train[feats], train['BAD'])

rf_predict=rf.predict(test[feats])

rf_accuracy=accuracy_score(test['BAD'], rf_predict)

rf_scores = cross_val_score(rf, test[feats], rf_predict, n_jobs=-1, cv=5)

rf_model_f1=cross_validate(rf, test[feats] ,rf_predict, scoring='f1',n_jobs=-1, cv=5)

temp = pd.Series([rf_accuracy, rf_scores.mean(), rf_model_f1['test_score'].mean()], index=['ACCURACY', 'K-FOLD', 'F1'])

val_model_rf = pd.DataFrame(temp, columns=['Resultado_RF'])
# Importancia das Variáveis - Modelo Random Forest

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()



#Matrix de Confusão - Modelo Random Forest

skplt.metrics.plot_confusion_matrix(test['BAD'] ,rf_predict, normalize=True)

#Validação Modelo Random Forest



val_model_rf.head()

# Trabalhando com GBM



gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)

gbm.fit(train[feats], train['BAD'])

gbm_predict=gbm.predict(test[feats])

gbm_accuracy = accuracy_score(test['BAD'], gbm_predict)

gbm_scores = cross_val_score(gbm, test[feats], gbm_predict, n_jobs=-1, cv=5)

gbm_model_f1=cross_validate(gbm, test[feats] ,gbm_predict, scoring='f1',n_jobs=-1, cv=5)



temp = pd.Series([gbm_accuracy, gbm_scores.mean(), gbm_model_f1['test_score'].mean()], index=['ACCURACY', 'K-FOLD', 'F1'])

val_model_gbm = pd.DataFrame(temp, columns=['Resultado_GBM'])
# Importancia das Variáveis - GBM

pd.Series(gbm.feature_importances_, index=feats).sort_values().plot.barh()

#Matrix de Confusão - Modelo GBM

skplt.metrics.plot_confusion_matrix(test['BAD'] ,gbm_predict, normalize=True)

#Validação Modelo GBM



val_model_gbm.head()
# Trabalhando com XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=200, learning_rate=0.09, random_state=42)

xgb.fit(train[feats], train['BAD'])

xgb_predict=xgb.predict(test[feats])

xgb_accuracy=accuracy_score(test['BAD'], xgb_predict)

xgb_model_f1=cross_validate(xgb, test[feats] ,xgb_predict, scoring='f1',n_jobs=-1, cv=5)

xgb_scores = cross_val_score(xgb, test[feats], xgb_predict, n_jobs=-1, cv=5)



temp = pd.Series([xgb_accuracy, xgb_scores.mean(), xgb_model_f1['test_score'].mean()], index=['ACCURACY', 'K-FOLD', 'F1'])

val_model_xgb = pd.DataFrame(temp, columns=['Resultado_XGB'])
# Importancia das Variáveis - XGB

pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()

#Matrix de Confusão - Modelo XGB

skplt.metrics.plot_confusion_matrix(test['BAD'] ,xgb_predict, normalize=True)



#Validação Modelo XGB



val_model_xgb.head()
# Compilação dos Resultados da Cross Validation dos modelos utilizados



final_result = pd.concat([val_model_rf,val_model_gbm,val_model_xgb],axis=1)



final_result.head().T