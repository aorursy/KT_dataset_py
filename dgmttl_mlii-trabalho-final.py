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
# Importando arquivo csv para o dataframe
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

# Conferindo os tipos de dados 
df.info()
# Traduzindo o nome das colunas
df.rename(columns={'BAD': 'INADIMPLENTE',
                  'LOAN': 'VALOR_EMPRESTIMO',
                  'MORTDUE': 'VALOR_HIPOTECA',
                  'VALUE': 'VALOR_PROPRIEDADE',
                  'REASON': 'MOTIVO',
                  'JOB': 'AREA_OCUPACAO',
                  'YOJ': 'ANOS_TRABALHO',
                  'DEROG': 'QTD_PROTESTOS',
                  'DELINQ': 'QTD_CALOTES',
                  'CLAGE': 'QTD_MESES_PRIM_EMPRES',
                  'NINQ': 'QTD_LINHAS_CRED_REC',
                  'CLNO': 'QTD_LINHAS_CRED',
                  'DEBTINC': 'IND_COMPROMET_SAL'}, inplace = True)


# Olhando os dados aleatóriamente
df.sample(15).T
# Importando bibliotecas gráficas
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (11,7)

# Gerando gráfico para análise de distribuição da variável 'VALOR_EMPRÉSTIMO'
f, ax = plt.subplots(figsize=(15,6))
sns.distplot(df['VALOR_EMPRESTIMO'], hist = True, kde = True, label='Valor Empréstimo')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Distribuição de Valores do Empréstimo')
plt.xlabel('Valor em US$')
plt.ylabel('Frequência')
df[df.columns[1:2]].describe().style.format("{:.2f}")
# Gerando gráfico para análise de distribuição das variáveis 'VALOR_HIPOTECA' e 'VALOR_PROPRIEDADE'
f, ax = plt.subplots(figsize=(15,6))
sns.distplot(df['VALOR_HIPOTECA'], hist = True, kde = True, label='Valor Hipoteca')
sns.distplot(df['VALOR_PROPRIEDADE'], hist = True, kde = True, label='Valor da Propriedade')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Distribuição de Variáveis')
plt.xlabel('Valor em US$')
plt.ylabel('Frequência')
df_rel_hip_prop = df[df.columns[2:4]].describe()

df_rel_hip_prop['rel_hip_prop'] = df_rel_hip_prop['VALOR_HIPOTECA'] / df_rel_hip_prop['VALOR_PROPRIEDADE']

df_rel_hip_prop.style.format("{:.2f}")
# Boxplot para análise da relação entre o valor do emprestimo e o bom e mal pagador.
sns.set_style("whitegrid") 

sns.boxplot(x = 'INADIMPLENTE', y = 'VALOR_EMPRESTIMO', data = df)
df_agrup_inad = df[['VALOR_EMPRESTIMO']].groupby(df['INADIMPLENTE'])
df_agrup_inad.describe().T
df_agrup_inad = pd.crosstab(df.AREA_OCUPACAO, df.INADIMPLENTE)

df_agrup_inad.div(df_agrup_inad.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,
                                                                 title='Área de Ocupação X Tipo de Pagadores',
                                                                 figsize=(10,5))

df_prof_inad = pd.crosstab(df['INADIMPLENTE'],df['AREA_OCUPACAO'])
df_prof_inad = df_prof_inad.T
df_prof_inad = df_prof_inad[1]


plt.pie(df_prof_inad, colors=['b', 'g', 'r', 'c', 'm', 'y'], 
        labels= df_prof_inad.index,explode=(0, 0, 0.2, 0.2, 0, 0),
        autopct='%1.1f%%',
        counterclock=False, shadow=True)

plt.title('Proporção de Maus Pagadores por Área de Ocupação')
plt.legend(df_prof_inad.index,loc=3)
plt.show()
# Explorando a correlação entre as variáveis

f, ax = plt.subplots(figsize=(15,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', linecolor='white', ax=ax, lw=.7)
# Gerando gráfico para análise de distribuição da variável 'INADIMPLENTE'

ax = sns.countplot(y="INADIMPLENTE", data=df)
plt.title('Análise da quantidade de inadimplentes')
plt.xlabel('QUANTIDADE')

total = len(df['INADIMPLENTE'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()
df_inad_1 = df[df['INADIMPLENTE'] == 1]
df_inad_2 = df[df['INADIMPLENTE'] == 0]

med_1 = pd.DataFrame(df_inad_1.mean())
med_1.rename(columns = {0:'Mau Pagador'}, inplace=True)
med_2 = pd.DataFrame(df_inad_2.mean())
med_2.rename(columns = {0:'Bom Pagador'}, inplace=True)

df_med_inad = pd.concat([med_1, med_2], axis=1, join='inner')


df_med_inad["rel_mau_bom"] = df_med_inad['Mau Pagador'] / df_med_inad['Bom Pagador']

df_med_inad
# Substituindo valores nulos

df_subst_na = df.copy()

df_subst_na['VALOR_HIPOTECA'] = df_subst_na['VALOR_HIPOTECA'].fillna(0)
df_subst_na['VALOR_PROPRIEDADE'] = df_subst_na['VALOR_PROPRIEDADE'].fillna(0)
df_subst_na['ANOS_TRABALHO'] = df_subst_na['ANOS_TRABALHO'].fillna(0)
df_subst_na['QTD_PROTESTOS'] = df_subst_na['QTD_PROTESTOS'].fillna(0)
df_subst_na['QTD_CALOTES'] = df_subst_na['QTD_CALOTES'].fillna(0)
df_subst_na['QTD_MESES_PRIM_EMPRES'] = df_subst_na['QTD_MESES_PRIM_EMPRES'].fillna(0)
df_subst_na['QTD_LINHAS_CRED_REC'] = df_subst_na['QTD_LINHAS_CRED_REC'].fillna(0)
df_subst_na['QTD_LINHAS_CRED'] = df_subst_na['QTD_LINHAS_CRED'].fillna(0)
df_subst_na['IND_COMPROMET_SAL'] = df_subst_na['IND_COMPROMET_SAL'].fillna(0)

df_subst_na['MOTIVO'] = df_subst_na['MOTIVO'].fillna('Not filled')
df_subst_na['AREA_OCUPACAO'] = df_subst_na['AREA_OCUPACAO'].fillna('Not filled')


# Convertendo colunas de valores para float
df_subst_na['VALOR_EMPRESTIMO'] = df_subst_na['VALOR_EMPRESTIMO'].astype(float)

# Convertendo colunas para inteiro
df_subst_na['QTD_PROTESTOS'] = df_subst_na['QTD_PROTESTOS'].astype(int)
df_subst_na['QTD_CALOTES'] = df_subst_na['QTD_CALOTES'].astype(int)
df_subst_na['QTD_LINHAS_CRED_REC'] = df_subst_na['QTD_LINHAS_CRED_REC'].astype(int)
df_subst_na['QTD_LINHAS_CRED'] = df_subst_na['QTD_LINHAS_CRED'].astype(int)
# Identificando valores NA
df_subst_na.isna().sum()
df_subst_na.info()
# Convertendo as colunas categóricas em dummies
df_subst_na = pd.get_dummies(df_subst_na, columns=['MOTIVO','AREA_OCUPACAO'])

df_subst_na.info()
# Separando o dataframe em train e test

# Importando o train_test_split
from sklearn.model_selection import train_test_split

# primeiro, train e test
train, test = train_test_split(df_subst_na, test_size=0.2, random_state=42)


train.shape, test.shape
# definindo as colunas de entrada

feats = [c for c in df_subst_na.columns if c not in ['INADIMPLENTE']]
# Importando bibliotecas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Instanciando o modelo
rf = RandomForestClassifier(n_estimators=150, random_state=42)

# treinar o modelo
rf.fit(train[feats], train['INADIMPLENTE'])

# Verificar a acurácia com a massa de teste
resultado = accuracy_score(test['INADIMPLENTE'], rf.predict(test[feats]))

# Guardando resultado no dataframe
resultados = pd.DataFrame([{'tipo':'rf', 'modo':'single','resultado':resultado}])



# Feature Importance com RF
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()


# Importando bibliotecas
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)
gbm.fit(train[feats], train['INADIMPLENTE'])

# Verificando a acurácia com a massa de teste
resultado = accuracy_score(test['INADIMPLENTE'], gbm.predict(test[feats]))

# Guardando resultado no dataframe
resultados.loc[1] = ['gbm', 'single',resultado]



# Feature Importance com GBM
pd.Series(gbm.feature_importances_, index=feats).sort_values().plot.barh()


# Importando Bibliotecas
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=200, learning_rate=0.09, random_state=42)
xgb.fit(train[feats], train['INADIMPLENTE'])

# Verificando a acurácia com a massa de teste
resultado = accuracy_score(test['INADIMPLENTE'], xgb.predict(test[feats]))

# Guardando resultado no dataframe
resultados.loc[2] = ['xgb','single',resultado]


# Feature Importance com XGBoost
pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()

# Importando bibliotecas
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, train[feats], train['INADIMPLENTE'], n_jobs=-1, cv=8)

# Guardando resultado no dataframe
resultados.loc[3] = ['rf','cross-validation',scores.mean()]

# Feature Importance com RF
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
scores = cross_val_score(gbm, train[feats], train['INADIMPLENTE'], n_jobs=-1, cv=7)

# Guardando resultado no dataframe
resultados.loc[4] = ['gbm','cross-validation',scores.mean()]


# Feature Importance com GBM
pd.Series(gbm.feature_importances_, index=feats).sort_values().plot.barh()
scores = cross_val_score(xgb, train[feats], train['INADIMPLENTE'], n_jobs=-1, cv=9)

# Guardando resultado no dataframe
resultados.loc[5] = ['xgb','cross-validation',scores.mean()]


# Feature Importance com XGBoost
pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()

resultados.sort_values('resultado', ascending=False)
# importando a biblioteca para plotar o gráfico de Matriz de Confusão
import scikitplot as skplt

preds_test = xgb.predict(test[feats])

# Matriz de confusão - Dados de test
skplt.metrics.plot_confusion_matrix(test['INADIMPLENTE'], preds_test)