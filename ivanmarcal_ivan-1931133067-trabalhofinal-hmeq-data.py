# Importando as Bibliotecas
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import itertools

from sklearn.metrics import balanced_accuracy_score
from imblearn.datasets import fetch_datasets
from imblearn.ensemble import BalancedRandomForestClassifier

from imblearn.metrics import geometric_mean_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Importando os dados para um dataframe
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
# Visualizando quantidade de colunas e linhas
print('df:', df.shape)
# Visualizando os tipos das variáveis
df.info()
# Visualizando os 5 priemiros registros
df.sample(5).T
# Analisando os dados - Coluna "BAD"
df['BAD'].value_counts()
# Gráfico de barras - Coluna "BAD"
# Valores: 1 = cliente inadimplente no empréstimo 0 = empréstimo recebido
df['BAD'].value_counts().plot.bar()
# Analisando os dados - Coluna "JOB"
# Valores JOB:
#     Mgr -> trabalho de gerente
#     Office -> trabalho de escritório
#     ProfExe -> trabalho profissional e/ou executivo
#     Sales -> trabalho com vendas
#     Self -> trabalho por conta própria
#     Other -> outros trabalhos
df['JOB'].value_counts()
# Gráfico de barras - Coluna "BAD"
df['JOB'].value_counts().plot.bar()
# Gráfico de barras - Colunas "JOB" x "BAD"
SitEmpre_Trabalho = pd.crosstab(df['JOB'],df['BAD'])
SitEmpre_Trabalho.div(SitEmpre_Trabalho.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Situação Emprestimo x Trabalho', figsize=(8,8))
# Analisando os dados - Coluna "REASON"
df['REASON'].value_counts()
# Gráfico de barras - Colunas "REASON"
# Valores de REASON: 
#      DebtCon = consolidação da dívida 
#      HomeImp = melhoria da casa
df['REASON'].value_counts().plot.bar()
# Gráfico de barras - Colunas "BAD" x "REASON"
SitEmpre_Razao = pd.crosstab(df['REASON'],df['BAD'])
SitEmpre_Razao.div(SitEmpre_Razao.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Situação Empréstimo x Motivo do Empréstimo', figsize=(8,8))
# Visualizando dados estatíticos básicos apenas de BAD = 1 Inadimplência
df[df['BAD']==1].drop('BAD', axis=1).describe().style.format("{:.2f}")
# Visualizando dados estatísticos básicos apenas de BAD = 0 Normalidade
df[df['BAD']==0].drop('BAD', axis=1).describe().style.format("{:.2f}")
# Preenchendo os valores nulos com:
# Média: VALUE, YOJ E DEBTINC
# Valores Fixos: REASON e JOB
# Zero: para os demais campos
df = df.fillna({"VALUE": df['VALUE'].mean()//1, 
                            "MORTDUE": 0,  
                            "DEROG": 0, 
                            "DELINQ": 0, 
                            "CLAGE": 0, 
                            "NINQ": 0, 
                            "CLNO": 0, 
                            "YOJ": df['YOJ'].mean()//1, 
                            "DEBTINC": df['DEBTINC'].mean()//1,
                            "REASON": 'Debtcon', 
                            "JOB": 'Other'})
# Reexibindo os dados após o preebchimento dos nulos
df.info()
# Transformando os tipos object em categoricos
for col in df.columns:
    if df[col].dtype == 'object':
        df[col]= df[col].astype('category').cat.codes
# Reexibindo os dados após a trabsformação
df.info()
# Visualizando dados estatísticos de todas variáveis
df.describe().style.format("{:.2f}")
# Exibindo primeiros registros de forma transposta
# Visualizando os primeiros apresentados de forma TRANSPOSTA .T
df.head().T
# Criando matriz de correlação
df_matriz = df.corr()
# Exibindo correlação das colunas com coluna JOB
df_matriz["JOB"].sort_values(ascending=False)
# Criando nova coluna 
# Razão entre montante do pedido de empréstimo e o valor da propriedade atual
# LOAN - Montante do pedido de empréstimo
# VALUE - Valor da propriedade atual
df["RAZAO_LOAN_VALUE"] = df["LOAN"]/df["VALUE"]

# Visualizando alguns dados da nova coluna
df['RAZAO_LOAN_VALUE'].sample(5)
# Com a criação do novo ("RAZAO_LOAN_VALUE") é necessário recriar matriz de correlação
d_matriz = df.corr()
# Visualizando a correlação com a coluna "JOB"
df_matriz["JOB"].sort_values(ascending=True)
# Criando uma cópia do dataframe "df"
df2 = df.copy()
# Selecionando as colunas para uso no modelo (exceto "BAD")
feats = [c for c in df2.columns if c not in ['BAD']]

# Exibindo as colunas selecionadas
feats
# Criando as bases de TEST e TRAIN
train, test = df2[feats], df2['BAD']
x_train, x_test, y_train, y_test = train_test_split(train, test, stratify=test, random_state=0)

# Treinando os modelos para predição: trabalhando com XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200, learning_rate=0.4, random_state=42)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
print('XGB performance:')
print('Accuracy: {:.4f}'
      .format(accuracy_score(y_test, y_pred_xgb)))
print('Balanced accuracy: {:.4f}'
      .format(balanced_accuracy_score(y_test, y_pred_xgb)))
print('F1 Score: {:.4f}'
      .format(f1_score(y_test, y_pred_xgb)))

# Visualizando as colunas em ordem decrescente em grau de importância - XGB
pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()
# Treinando os modelos para predição: trabalhando com BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=50, random_state=0,
                                     n_jobs=-1)
brf.fit(x_train, y_train)
y_pred_brf = brf.predict(x_test)
print('BalancedRandomForestClassifier:')
print('Accuracy: {:.4f}'
      .format(accuracy_score(y_test, y_pred_brf)))
print('Balanced accuracy: {:.4f}'
      .format(balanced_accuracy_score(y_test, y_pred_brf)))
print('F1 Score: {:.4f}'
      .format(f1_score(y_test, y_pred_brf)))
            
# Treinando os modelos para predição: trabalhando com RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print('RandomForestClassifier:')
print('Accuracy: {:.4f}'
      .format(accuracy_score(y_test, y_pred_rf)))
print('Balanced accuracy: {:.4f}'
      .format(balanced_accuracy_score(y_test, y_pred_rf)))
print('F1 Score: {:.4f}'
      .format(f1_score(y_test, y_pred_rf)))
