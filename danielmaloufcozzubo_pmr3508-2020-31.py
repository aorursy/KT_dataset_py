import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
dados_treino = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv', na_values='?')
dados_treino
null_val = dados_treino.isnull().sum()
porc = (null_val/len(dados_treino))*100
null_val = pd.concat([null_val, porc], axis=1).sort_values(by=1, ascending=False)
null_val.columns = ['Total', 'Porcentagem']
null_val.head(5)
null_total = null_val['Total'].sum()
null_linhas = dados_treino.isnull().any(axis=1).sum()
lin_per = '%.2f'%(null_linhas/len(dados_treino)*100)

print(f'O dataset apresenta {null_total} dados faltantes, sendo que {null_linhas} linhas apresentam ao menos um valor faltante.\nIsso repesenta {lin_per}% das linhas com dados faltantes.')
dados_treino.fillna('Missing', inplace=True)
dados_treino.describe()
dados_treino.describe(include=['O'])
sns.relplot('capital.loss', 'capital.gain', data=dados_treino)
plt.xlabel('capital.loss')
plt.ylabel('capital.gain')
plt.show()
dados_treino['capital.loss'].hist()
plt.title('Distribuição capital.loss')
plt.show()

dados_treino['capital.gain'].hist()
plt.title('Distribuição capital.gain')
plt.show()
out_gain = dados_treino[dados_treino['capital.gain']==99999]

per_out = '%.2f'%((len(out_gain)/len(dados_treino))*100)

out_gain
print(f'Frente ao volume dos dados, as {len(out_gain)} linhas que contêm o outlier de 99999 no capital gain representam uma parcela muito pequena dos dados ({per_out}%).')
avg_capital = dados_treino['capital.gain'].mean()
dados_treino['capital.gain'].replace(99999, avg_capital, inplace=True)
dados_treino['capital.change'] = dados_treino['capital.gain'] - dados_treino['capital.loss']

dados_treino['capital.change'].hist()
plt.title('Distribuição capital.change')
plt.show()

dados_treino['capital.change'].describe()
dados_maior50 = dados_treino[dados_treino['income'] == '>50K']['capital.change'].values.tolist()
dados_menor50 = dados_treino[dados_treino['income'] == '<=50K']['capital.change'].values.tolist()

t_test, pval = ttest_ind(random.sample(dados_maior50, 100), random.sample(dados_menor50, 100), equal_var=0)
print(f'O p-value do teste foi de {pval}.')
dados_treino['age'].hist()
plt.title('Distribuição das observações pela idade')
plt.show()
wgt_maior = dados_treino[dados_treino['income'] == '>50K']['fnlwgt'].values.tolist()
wgt_menor = dados_treino[dados_treino['income'] == '<=50K']['fnlwgt'].values.tolist()

t_test_wgt, pval_wgt = ttest_ind(random.sample(wgt_maior, 100), random.sample(wgt_menor, 100), equal_var=0)
print(f'O p-value do teste foi de {pval_wgt}.')
hr_maior = dados_treino[dados_treino['income'] == '>50K']['hours.per.week'].values.tolist()
hr_menor = dados_treino[dados_treino['income'] == '<=50K']['hours.per.week'].values.tolist()

plt.hist(hr_menor, alpha=0.5, label='<=50K')
plt.hist(hr_maior, alpha=0.5, label='>50K')
plt.legend(loc='upper right')
plt.title('Distribuição das horas trabalhadas na semana filtrado pela renda')
plt.show()
pais = dados_treino[['native.country','Id']].groupby('native.country').count().reset_index(drop=False).sort_values('Id')

plt.figure(figsize=(15,10))
plt.barh('native.country', 'Id', data=pais)
plt.title('Distribuição de observações por nacionalidade')
pct_usa = '%.2f'%(pais.iloc[-1, 1]/sum(pais.iloc[:, 1])*100)
print(f'A porcentagem de observações de pessoas nascidas nos EUA é de {pct_usa}%.')
g = sns.countplot('sex', data=dados_treino, hue='income')
plt.xlabel('Gênero')
plt.ylabel('')
g.axes.get_legend().set_title('Renda Anual')
plt.title('Distribuição dos gêneros filtrados pela renda')
plt.show()
plt.figure(figsize=(20, 10))
g = sns.countplot('occupation', data=dados_treino, hue='sex')
plt.xlabel('Ocupação')
plt.ylabel('')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.axes.get_legend().set_title('Gênero')
plt.title('Distribuição das ocupações filtrados por gênero')
plt.show()
plt.figure(figsize=(20, 10))
g = sns.countplot('occupation', data=dados_treino, hue='income')
plt.xlabel('Ocupação')
plt.ylabel('')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.axes.get_legend().set_title('Renda Anual')
plt.title('Distribuição das ocupações filtrados pela renda anual')
plt.show()
dados_treino['native.country.grouped'] = [nc if nc == 'United-States' else 'Others' for nc in dados_treino['native.country']]
le = LabelEncoder()

dados_treino['workclass.num'] = le.fit_transform(dados_treino['workclass'])
dados_treino['marital.status.num'] = le.fit_transform(dados_treino['marital.status'])
dados_treino['occupation.num'] = le.fit_transform(dados_treino['occupation'])
dados_treino['relationship.num'] = le.fit_transform(dados_treino['relationship'])
dados_treino['sex.num'] = le.fit_transform(dados_treino['sex'])
dados_treino['native.country.num'] = le.fit_transform(dados_treino['native.country'])
dados_treino['native.country.grouped.num'] = le.fit_transform(dados_treino['native.country.grouped'])
dados_treino['race.num'] = le.fit_transform(dados_treino['race'])
dados_treino['income.num'] = le.fit_transform(dados_treino['income'])
dados_treino[['income', 'income.num']]
corr = dados_treino.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True)
plt.show()
X_treino = dados_treino[['age', 'education.num', 'hours.per.week', 
                         'capital.change', 'marital.status.num',
                         'relationship.num', 'sex.num']]
Y_treino = dados_treino['income.num']
acc = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i, metric = 'manhattan')
    knn.fit(X_treino, Y_treino)
    scores = cross_val_score(knn, X_treino, Y_treino, cv=10)
    acc.append(scores.mean())
plt.figure(figsize=(15,10))
plt.plot(range(1, 40), acc)
plt.title('Acurácia do modelo')
plt.xlabel('Número de vizinhos')
plt.grid(True)
plt.xticks()
plt.show()
knn = KNeighborsClassifier(n_neighbors=24, metric = 'manhattan')

knn.fit(X_treino, Y_treino)
scores = cross_val_score(knn, X_treino, Y_treino, cv=10)
scores.mean()
dados_teste = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv', na_values='?')
dados_teste
dados_teste['capital.change'] = dados_teste['capital.gain'] - dados_teste['capital.loss']
dados_teste['marital.status.num'] = le.fit_transform(dados_teste['marital.status'])
dados_teste['relationship.num'] = le.fit_transform(dados_teste['relationship'])
dados_teste['sex.num'] = le.fit_transform(dados_teste['sex'])
dados_teste_sel = dados_teste[['age', 'education.num', 'hours.per.week', 
                         'capital.change', 'marital.status.num',
                         'relationship.num', 'sex.num']]
null_val = dados_teste_sel.isnull().sum()
porc = (null_val/len(dados_teste_sel))*100
null_val = pd.concat([null_val, porc], axis=1).sort_values(by=1, ascending=False)
null_val.columns = ['Total', 'Porcentagem']
null_val.head(5)
Y_pred = knn.predict_proba(dados_teste_sel)
#transformação das probabilidades em classes
class_pred = []
for i in range(len(Y_pred)):
    if Y_pred[i][0] >= Y_pred[i][1]:
        class_pred.append('<=50K')
    else:
        class_pred.append('>50K')
submission = pd.DataFrame()
submission['Id'] = dados_teste['Id']
submission['income'] = class_pred
submission.to_csv('./submission.csv', index=False)