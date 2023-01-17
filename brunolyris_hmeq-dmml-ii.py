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
import seaborn as sns
import matplotlib.pyplot as plt
#Importando dataset
df= pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.shape

df.head()
df.info()
graf1 = sns.barplot(x=df.BAD, y=df.BAD, data=df, estimator=lambda x: len(x) / len(df) * 100)
graf1.set(ylabel="Percent")
sns.distplot(df.LOAN.dropna())
plt.title('Distribuição de quantidade de emprestimo')
plt.xlabel('Valor')
plt.ylabel('Quantidade')

sns.distplot(df.MORTDUE.dropna())
plt.title('Distribição do valor do empréstimo ')
plt.ylabel('Quantidade')
plt.xlabel('Valor da divida')
sns.distplot(df.VALUE.dropna())
plt.title('Distribuição do valor do imovel')
plt.xlabel('Valor')
plt.ylabel('Quantidade')
plt.title('Distribuiçao de anos de trabalho')
sns.distplot(df.YOJ.dropna())
plt.xlabel('Anos trabalhaos')
plt.ylabel('Quantidade')
a = df.REASON.copy()
a[pd.isnull(a)==True] = 'Missing'
sns.countplot(a, hue = df.BAD)
plt.title('Status de Emprestimo para pagar outro emprestimo')
plt.xlabel('Motivo')
plt.ylabel('Quantidade')
b = df.JOB.copy()
b[pd.isnull(b)==True] = 'Missing'
sns.countplot(b, hue= df.BAD)
plt.title('Emprestimo por Cargo')
plt.ylabel('Quantidade')
plt.xlabel('Cargo')
def dist_boxplot(x, **kwargs):
    ax = sns.distplot(x, hist_kws=dict(alpha=0.2))
    ax2 = ax.twinx()
    sns.boxplot(x=x, ax=ax2)
    ax2.set(ylim=(-5, 5))

g = sns.FacetGrid(df, col="BAD")
g1 = g.map(dist_boxplot, 'CLAGE', data = df)

g = sns.FacetGrid(df, col="BAD")
g2 = g.map(dist_boxplot, 'NINQ', data = df)

g = sns.FacetGrid(df, col="BAD")
g3 = g.map(dist_boxplot, 'CLNO', data = df)

g = sns.FacetGrid(df, col="BAD")
g4 = g.map(dist_boxplot, 'DEBTINC', data = df)
# input de valores na varial DEBTINC pela media de acordo com a descrição da coluna REASON
df.loc[(df.REASON == 'HomeImp') & (pd.isnull(df.DEBTINC) == True),'DEBTINC'] = np.mean(df.loc[df.REASON == 'HomeImp','DEBTINC'])
df.loc[(df.REASON == 'DebtCon') & (pd.isnull(df.DEBTINC) == True),'DEBTINC'] = np.mean(df.loc[df.REASON == 'DebtCon','DEBTINC'])
df.loc[(pd.isnull(df.REASON) == True) & (pd.isnull(df.DEBTINC) == True),'DEBTINC'] = np.mean(df.loc[pd.isnull(df.REASON) == True,'DEBTINC'])
df.info()
#input de 0 para valores faltantes na coluna de credito inadimplente 
df.loc[pd.isnull(df.DELINQ)==True, 'DELINQ'] = 0
df.loc[pd.isnull(df.JOB)==True, 'JOB'] = 'Missing'
df.isnull().sum()
df = df.fillna(0)
df.shape
df.info()

import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scikitplot as skplt
from sklearn.model_selection import cross_val_score
df_rf = pd.get_dummies(df, df.dtypes[(df.dtypes==np.object) | (df.dtypes=='category')].index.values, drop_first=True)
df_rf.head()
train, test = train_test_split(df_rf, test_size=0.20, random_state=42)
train, valid = train_test_split(train, test_size=0.20, random_state=42)

train.shape, valid.shape, test.shape
df_rf.info()
rf = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators = 150, random_state=150)
feats = [c for c in df_rf.columns if c not in ['BAD']]
feats
rf.fit(train[feats],train['BAD'])
# Prevendo os dados de validação
preds_val = rf.predict(valid[feats])

accuracy_score(valid['BAD'], preds_val)
preds_test = rf.predict(test[feats])

preds_test
accuracy_score(test['BAD'], preds_test)
skplt.metrics.plot_confusion_matrix(test['BAD'], preds_test)
train, test = train_test_split(df_rf, test_size=0.20, random_state=42)
train.shape, test.shape
scores = cross_val_score(rf, train[feats], train['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
scores = cross_val_score(rf, test[feats], test['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()