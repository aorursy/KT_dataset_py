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
#-- carregando as libs

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE



sns.set(rc={'figure.figsize':(25, 18)})
#-- loading dataset

db = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

db.head()
#-- checking dataset dimensionality

print(f'Dimensões do dataset - \n linhas = {db.shape[0]} \n colunas = {db.shape[1]}')
#-- checking variables type

db.info()
#-- checking missing values

missing_values = pd.DataFrame(db.isnull().sum().sort_values(ascending=False) / len(db))

missing_values.reset_index(inplace=True)

missing_values.columns = ['features', 'porcentagem']

missing_values.head()
#-- print columns by missing values

for i in range(0, len(missing_values)):

    print(f'{missing_values.iloc[i, 0]} - sua pocentagem é: {missing_values.iloc[i, 1]}')
#-- checking missing values

_ = sns.heatmap(db.isnull(), cbar=False)
#-- creatings a list with columns names

var_columns = list(db.columns)
#-- missing values

_ = plt.bar(missing_values['features'], missing_values['porcentagem'], width = 0.6)

_ = plt.xticks(rotation=90, fontsize=6)

_ = plt.axhline(y=0.8, color='r', linestyle='--', lw=2)
#-- filtrando colunas com missing values

columns_filt = list(missing_values[missing_values['porcentagem'] <= 0.8]['features'])
#-- dropando as linhas com missing values

db_filt = db[columns_filt].dropna()

db_filt.reset_index(inplace=True)
#-- verificando o tipo dos dados

db_filt.info()
#-- selecionando os dados numéricos

db_filt_num = db_filt.select_dtypes(include=['int64'])
#-- verificando o tipo dos dados

db_filt_num.info()
#-- printando métricas de avaliação 

db_filt_num.iloc[:, 1:].describe()
#-- verificando as correlações entre os dados numéricos

_ = sns.heatmap(db_filt_num.iloc[:, 1:].corr())
#-- printando o pairplot

_ = sns.pairplot(db_filt_num.iloc[:, 1:], corner=True, height=4)
#-- selecionadno os dados categóricos

db_filt_cat = db_filt.select_dtypes(include='object')
#-- printando algumas métricas

db_filt_cat.describe()
db_filt.columns
#-- dropando a coluna resposta

db_filt = db_filt.drop(columns=['Patient ID'])
#-- feature engineering

db_filt_dummies = pd.get_dummies(db_filt)
#-- printando os heads

db_filt_dummies.head()
#-- criando as variáveis do modelo

X = db_filt_dummies.drop(columns=['SARS-Cov-2 exam result_negative', 'SARS-Cov-2 exam result_positive'])

y = db_filt['SARS-Cov-2 exam result']
#-- verificando o balaceamento dos dadso

y.value_counts()
#-- alterando nome da coluna

y.columns = ['SARS-Cov-2 exam result']
for i in range(0, len(y)):

    if y[i] == 'negative':

        y[i] = 0

    else:

        y[i] = 1
y[:] = y[:].apply(pd.to_numeric)
#-- criando o balanceamento

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns

os_data_X, os_data_y = os.fit_sample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X, columns=columns )

os_data_y = pd.DataFrame(data=os_data_y)



#-- checando

print("tamanho do dataset ",len(os_data_X))

print("número de paciente sem covid",len(os_data_y[os_data_y[:] == 0]))

print("número de paciente com covid",len(os_data_y[os_data_y[:] == 1]))

print("proporção de sem covid ",len(os_data_y[os_data_y[:] == 0])/len(os_data_X))

print("proporção de com covid ",len(os_data_y[os_data_y[:] == 1])/len(os_data_X))
#-- criando o modelo

logistic = LogisticRegression(random_state=0)
#-- aplicando o modelo nos dados de treino

logistic.fit(os_data_X, os_data_y)
#-- aplicando o modelo nos dados de teste

y_pred = logistic.predict(X_test)
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,

                             roc_auc_score, log_loss, plot_confusion_matrix, roc_curve, auc, plot_roc_curve)
#-- calculando a acuracidade do modelo

accuracy_score(y_test, y_pred)
#-- printando a matrix de confusão

confusion_matrix(y_test, y_pred)
#-- printando os indicadores de recall

print(classification_report(y_test, y_pred))
#-- plotando a curva roc

_ = plot_roc_curve(logistic, X_test, y_test)
#-- calculando a probabilidade

y_sca_proba = logistic.predict_proba(X_test)

y_sca_proba[:5]
#-- selecionando a coluna de probabilidade ser um bom techlead

y_sca_proba = y_sca_proba[:, 1]

y_sca_proba[:5]
#-- selecionando a probabilidade maior do que 70%

y_pred_customizado = y_sca_proba >= 0.80
#-- printando a classificação do modelo

print(classification_report(y_test, y_pred_customizado))