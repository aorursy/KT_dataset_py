# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction/notebooks
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
def eda(dfA, all=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nIs Null: {dfA.isnull().sum().sum()}')

    print(f'{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if all:  # here you put yours prefered analysis that detail more your dataset



        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))



        # print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead:\n{dfA.head()}')

        print(f'\nSamples:\n{dfA.sample(2)}')

        print(f'\nTail:\n{dfA.tail()}')
eda(train)
train.columns = train.columns.str.lower()
train.head()
eda(test)
test.columns = test.columns.str.lower()

test.head()
import seaborn as sns

import matplotlib.pyplot as plt
gender = train.gender.unique()

gender
m = train[train.gender == gender[0]]['gender'].shape[0]

f = train[train.gender == gender[1]]['gender'].shape[0]
plt.pie([m,f], labels=gender, autopct='%1.1f%%')
age = train[['id', 'age']].groupby('age').count()

age
fig, ax1 = plt.subplots( sharey=True, figsize=(15,5))

sns.barplot(x=age.index, y=age.id.values, ax=ax1).set_title('Age')
def sepColumns(dataset):

    num = []

    cat = []

    for i in dataset.columns:

        if dataset[i].dtype == 'object':

            cat.append(i)

        else:

            num.append(i)

    return num, cat
num, cat = sepColumns(train)

train[cat]
vuCat = dict()

for c in cat:

    v = train[c].unique().tolist()

    vuCat[c] = v

print(vuCat)
for vc in vuCat:

#     print(vc, vuCat[vc])

    newCol = f'{vc}_N'

    train[newCol] = train[vc].apply(lambda x: vuCat[vc].index(x))
train.head()
numTrain, catTrain = sepColumns(train)

train[numTrain]
test.head()
numTest, catTest = sepColumns(test)

test[catTest]
vuCatTest = dict()

for c in catTest:

    v = test[c].unique().tolist()

    vuCatTest[c] = v

print(vuCatTest)
for vc in vuCatTest:

    newCol = f'{vc}_N'

    test[newCol] = test[vc].apply(lambda x: vuCatTest[vc].index(x))
test
numTest, catTest = sepColumns(test)

test[numTest]
def correlation(df, varT, xpoint=-0.5, showGraph=True):

    corr = df.corr()

    print(f'\nFeatures correlation:\n'

          f'Target: {varT}\n'

          f'Reference.: {xpoint}\n'

          f'\nMain features:')

    corrs = corr[varT]

    features = []

    for i in range(0, len(corrs)):

        if corrs[i] > xpoint and corrs.index[i] != varT:

            print(corrs.index[i], f'{corrs[i]:.2f}')

            features.append(corrs.index[i])

    if showGraph:

        fig, ax1 = plt.subplots( sharey=True, figsize=(15,10))

        sns.heatmap(corr,

                    annot=True, fmt='.2f', vmin=-1, vmax=1, linewidth=0.01,

                    linecolor='black', cmap='RdBu_r', ax=ax1

                    )

        plt.title('Correlations between features w/ target')

        plt.show()

    return features
varTarget = 'response'

varFeatures = correlation(train[numTrain], varTarget, 0.01)
# ML Algoritmos

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.dummy import DummyRegressor



# ML selecao de dados de treino e teste

from sklearn.model_selection import train_test_split

# calcular o menor erro medio absoluto entre 2 dados apresentados

from sklearn.metrics import mean_absolute_error
# I used this to choose what the Regressor fit better with data



# nrs = np.random.randint(1,43)

# nest = np.random.randint(1,43)

# regressors = [

#         LogisticRegression(random_state=nrs),

#         DecisionTreeRegressor(random_state=nrs),

#         RandomForestRegressor(n_estimators=nest, random_state=nrs),

#         SVR(C=1.0, epsilon=0.2),

#         LinearRegression(),

#         GradientBoostingRegressor(n_estimators=nest, random_state=nrs),

#         PoissonRegressor(),

#         DummyRegressor(strategy="mean"),

#         GaussianNB(),

#         AdaBoostRegressor(n_estimators=nest, random_state=nrs)

#     ]



# X = train[varFeatures]

# y = train[varTarget]

# Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=42)



# reg = []

# mae = []

# sco = []



# for regressor in regressors:

#     modelo = regressor

#     modelo.fit(Xtreino, np.array(ytreino))

#     sco.append(modelo.score(Xtreino, ytreino))

#     previsao = modelo.predict(Xteste)

#     mae.append(round(mean_absolute_error(yteste, previsao), 2))

#     reg.append(regressor)



# meuMae = pd.DataFrame(columns=['Regressor', 'mae', 'score'])

# meuMae['Regressor'] = reg

# meuMae['mae'] = mae

# meuMae['score'] = sco

# meuMae = meuMae.sort_values(by='score', ascending=False)



# print('Best score: ', meuMae["Regressor"].values[0])
Xtreino = train[varFeatures]

ytreino = train[varTarget]

Xteste = test[varFeatures]



modelo = LogisticRegression(random_state=44)  #meuMae["Regressor"].values[0]

modelo.fit(Xtreino, np.array(ytreino))

score = modelo.score(Xtreino, ytreino)

predict = modelo.predict(Xteste)



print(f'Score: {score:.2f}')