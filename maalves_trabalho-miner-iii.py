import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.info()
df.sample(10)
df['BAD'].value_counts()
df.isnull().sum()
def set_mode(df, var):

    df.loc[(df[var].isnull()) & (df['BAD'] == 1), var] = df.loc[(df[var].notnull()) & (df['BAD'] == 1), var].mode()[0]

    df.loc[(df[var].isnull()) & (df['BAD'] == 0), var] = df.loc[(df[var].notnull()) & (df['BAD'] == 0), var].mode()[0]

set_mode(df, 'NINQ')

set_mode(df, 'DELINQ')

set_mode(df, 'DEROG')

set_mode(df, 'CLNO')
df.fillna(df.mean(), inplace=True)
fig, ax = plt.subplots(figsize=(12, 5))



group = df.groupby('JOB').sum()

plt.subplot(1, 2, 1)

sns.barplot(x=group.index, y=group.BAD)

plt.subplot(1, 2, 2)



sns.barplot(x=df.JOB, y=df.LOAN)
def distplot(x, titulo, df):

    summary = pd.DataFrame(df[x].describe()).T

    summary.columns = ['Quantidade', 'Média', 'Desvio Padrão', 'Minímo','25%','50%','75%', 'Máximo']

    summary = summary.T



    fig, ax = plt.subplots(figsize=(12, 5))

    plt.subplot(1, 2, 1)



    sns.distplot(df[x])



    plt.subplot(1, 2, 2)



    table = plt.table(cellText=summary.values,

              rowLabels=summary.index,

              colLabels=summary.columns,

              cellLoc = 'right', rowLoc = 'center',

              loc='right', bbox=[.5,.05,.78,.78])



    plt.axis('off')



    table.set_fontsize(22)

    table.scale(3, 3)  # may help

    plt.title(titulo)
distplot('VALUE', 'Distribuição da variável VALUE', df)
def proriedade(x):

    if x <= 60000:

        return 'Padrão Baixo'

    if x <= 90000:

        return 'Padrão Médio'

    if x <= 120000:

        return 'Padrão Alto'

    if x > 120000:

        return 'Padrão Muito Alto'
df['TIPO_PROPRIEDADE'] = df['VALUE'].transform(proriedade)
fig, ax = plt.subplots(figsize=(18, 5))





plt.subplot(1, 3, 1)

count = df.groupby('TIPO_PROPRIEDADE').count()

plt.xticks(rotation=30);

plt.title("Quantidade de Empréstimos", fontsize=20)

sns.barplot(x=count.index, y=count.BAD)



plt.subplot(1, 3, 2)

plt.xticks(rotation=30);

mean = df.groupby('TIPO_PROPRIEDADE').mean()

plt.title("Média de Valores de Empréstimos", fontsize=20)

sns.barplot(x=mean.index, y=mean.LOAN)





plt.subplot(1, 3, 3)

soma = df.groupby('TIPO_PROPRIEDADE').sum()

plt.xticks(rotation=30);

plt.title("Quantidade de Inadiplentes", fontsize=20)

sns.barplot(x=soma.index, y=soma.BAD)



distplot('YOJ', 'Distribuição da variável YOJ - Total', df)

distplot('YOJ', 'Distribuição da variável YOJ - Maus pagadores', df[df['BAD'] == 1])

distplot('YOJ', 'Distribuição da variável YOJ - Bons pagadores', df[df['BAD'] == 0])
clage = df

clage['CLAGE']= clage['CLAGE'].round()
distplot('CLAGE', 'Distribuição da variável CLAGE - Total', clage)

distplot('CLAGE', 'Distribuição da variável CLAGE - Maus pagadores', clage[clage['BAD'] == 1])

distplot('CLAGE', 'Distribuição da variável CLAGE - Bons pagadores', clage[clage['BAD'] == 0])
sns.boxplot(clage['CLAGE'])
clage = clage[clage['CLAGE'] < 380]

df = df[df['CLAGE'] < 380]
distplot('CLAGE', 'Distribuição da variável CLAGE - Total', clage)

distplot('CLAGE', 'Distribuição da variável CLAGE - Maus pagadores', clage[clage['BAD'] == 1])

distplot('CLAGE', 'Distribuição da variável CLAGE - Bons pagadores', clage[clage['BAD'] == 0])
corr = df.corr()



fig, ax = plt.subplots(figsize=(10,8))



colors = sns.diverging_palette(200, 0, as_cmap=True)



sns.heatmap(corr, cmap=colors, annot=True, fmt=".2f")



plt.xticks(range(len(corr.columns)), corr.columns);



plt.yticks(range(len(corr.columns)), corr.columns)



plt.show()
# Criando dummys para todas as colunas categóricas

df = pd.get_dummies(df, columns=['TIPO_PROPRIEDADE', 'REASON', 'JOB'])
from sklearn.model_selection import GridSearchCV


# XGBoost



# Importar o modelo

from xgboost import XGBClassifier



# Instanciar o modelo

xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42)

hyper = {

            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

            "min_child_weight": [ 1, 3 ] ,

            "gamma": [0.1, 0.2 ],

            "colsample_bytree": [0.1, 0.3 ],

        }

metricas = ['accuracy', 'recall', 'f1']
xgb = XGBClassifier(n_estimators=200, n_jobs=4, random_state=42)



grid = GridSearchCV(xgb, param_grid=hyper, 

                         scoring=metricas, 

                         refit='f1', 

                         return_train_score=False)
# definindo colunas de entrada

feats = [c for c in df.columns if c not in ['BAD']]
grid.fit(df[feats], df['BAD'])
grid.best_estimator_
grid.best_score_
# Separando o dataframe



# Importando o train_test_split

from sklearn.model_selection import train_test_split



# Separando treino e teste

train, test = train_test_split(df, test_size=0.20, random_state=42)



# Não vamos mais usar o dataset de validação



train.shape, test.shape



# XGBoost



# Importar o modelo

from xgboost import XGBClassifier



# Instanciar o modelo

xgb = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.1, gamma=0.2, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.05, max_delta_step=0, max_depth=6,

              min_child_weight=3, monotone_constraints=None,

              n_estimators=200, n_jobs=4, num_parallel_tree=1,

              objective='binary:logistic', random_state=42, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

              validate_parameters=False, verbosity=None)

# Usar o cross validation

from sklearn.model_selection import cross_val_score



scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5)



# média de score é de 0.89

scores, scores.mean()
# Usando o XGB para treinamento e predição

xgb.fit(train[feats], train['BAD'])



# Fazendo predições

preds = xgb.predict(test[feats])
test['BAD'].value_counts()
plt.title("Tabela cruzada entre os valores preditos e os valores reais", fontsize=20)



ax = sns.heatmap(pd.crosstab(test['BAD'], preds),  annot=True, fmt="d", cmap="YlGnBu")

plt.xlabel('Predito')

plt.ylabel('Original')

ax.set_xticklabels(['BOM','MAU' ])

ax.set_yticklabels(['BOM','MAU' ])
# Medir o desempenho do modelo

from sklearn.metrics import f1_score



f1_score(test['BAD'], preds)