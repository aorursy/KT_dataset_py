import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
dados_train = pd.read_csv('../input/csv-files/train.csv')
dados_test = pd.read_csv('../input/csv-files/test.csv')
dados_train.head()
dados_train.drop(columns='Unnamed: 0', inplace = True)
train_nulos = pd.DataFrame(dados_train.isna().sum(), columns=['Nulos'])
train_nulos['Nan_percentual'] = ((train_nulos['Nulos'] / dados_train.shape[0]) * 100).round(2)
train_nulos = train_nulos[train_nulos['Nulos'] != 0]
train_nulos.sort_values(by = 'Nan_percentual', ascending= False, inplace =True)
train_nulos
dados_train.shape
dados_train = dados_train[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NU_INSCRICAO']]
dados_test = dados_test[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_INSCRICAO']]
dados_train.dropna(subset=['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'], inplace = True)
dados_train[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']].isna().sum()
dados_train.shape
ax = sns.heatmap(dados_train.corr(),annot = True, annot_kws = {'size': 20})
ax.figure.set_size_inches(18,12)
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.boxplot(dados_train.NU_NOTA_MT, orient= 'h', width=0.3)
ax.figure.set_size_inches(20, 6)
ax.set_title('Nota de Matemática', fontsize=20)
ax.set_xlabel('Valor', fontsize=16)
ax
dados_train.dtypes
dados_train.describe()
Q1 = dados_train.NU_NOTA_CH.quantile(0.25)
Q3 = dados_train.NU_NOTA_CH.quantile(0.75)
IIQ =  Q3 - Q1
lim_inf = Q1 -(1.5 * IIQ)
lim_sup = Q3 +(1.5 * IIQ)
dados_train = dados_train[(dados_train['NU_NOTA_CH'] >= lim_inf) & (dados_train['NU_NOTA_CH'] <= lim_sup)]
Q1 = dados_train.NU_NOTA_CN.quantile(0.25)
Q3 = dados_train.NU_NOTA_CN.quantile(0.75)
IIQ =  Q3 - Q1
lim_inf = Q1 -(1.5 * IIQ)
lim_sup = Q3 +(1.5 * IIQ)
dados_train = dados_train[(dados_train['NU_NOTA_CN'] >= lim_inf) & (dados_train['NU_NOTA_CN'] <= lim_sup)]
Q1 = dados_train.NU_NOTA_LC.quantile(0.25)
Q3 = dados_train.NU_NOTA_LC.quantile(0.75)
IIQ =  Q3 - Q1
lim_inf = Q1 -(1.5 * IIQ)
lim_sup = Q3 +(1.5 * IIQ)
dados_train = dados_train[(dados_train['NU_NOTA_LC'] >= lim_inf) & (dados_train['NU_NOTA_LC'] <= lim_sup)]
Q1 = dados_train.NU_NOTA_MT.quantile(0.25)
Q3 = dados_train.NU_NOTA_MT.quantile(0.75)
IIQ =  Q3 - Q1
lim_inf = Q1 -(1.5 * IIQ)
lim_sup = Q3 +(1.5 * IIQ)
dados_train = dados_train[(dados_train['NU_NOTA_MT'] >= lim_inf) & (dados_train['NU_NOTA_MT'] <= lim_sup)]

Q1 = dados_train.NU_NOTA_REDACAO.quantile(0.25)
Q3 = dados_train.NU_NOTA_REDACAO.quantile(0.75)
IIQ =  Q3 - Q1
lim_inf = Q1 -(1.5 * IIQ)
lim_sup = Q3 +(1.5 * IIQ)
dados_train = dados_train[(dados_train['NU_NOTA_REDACAO'] >= lim_inf) & (dados_train['NU_NOTA_REDACAO'] <= lim_sup)]
dados_train.shape
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.boxplot(dados_train.NU_NOTA_MT, orient= 'h', width=0.3)
ax.figure.set_size_inches(20, 6)
ax.set_title('Nota de Matemática', fontsize=20)
ax.set_xlabel('Valor', fontsize=16)
ax = ax
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.distplot(dados_train.NU_NOTA_MT)
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Notas de Matemática', fontsize=16)
ax = ax
dados_train.replace(0, 1, inplace = True)
dados_test.replace(0, 1, inplace = True)
dados_train_log = np.log(dados_train[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']])
dados_test_log = np.log(dados_test[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']])
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.distplot(dados_train_log.NU_NOTA_MT)
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Notas de Matemática', fontsize=16)
ax = ax
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.boxplot(dados_train_log.NU_NOTA_MT, orient= 'h', width=0.3)
ax.figure.set_size_inches(20, 6)
ax.set_title('Nota de Matemática', fontsize=20)
ax.set_xlabel('Valor', fontsize=16)
ax
y = dados_train_log['NU_NOTA_MT']
y.head()
X = dados_train_log[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']]
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_score(y_test, y_pred)
X_train.shape
y_train.shape
