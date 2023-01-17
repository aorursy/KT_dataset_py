# https://www.kaggle.com/rsrishav/youtube-trending-video-dataset



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
data = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('csv'):

            dft = pd.DataFrame(pd.read_csv(os.path.join(dirname, filename), header=0))

            dft['country'] = filename[:2]

            data.append(dft)



df = pd.concat(data, axis=0, ignore_index=True)
country = df.country.unique().tolist()

df['countryId'] = df.country.apply(lambda x: country.index(x))
df.head(2)
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

        

        #print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead:\n{dfA.head()}')

        print(f'\nSamples:\n{dfA.sample(2)}')

        print(f'\nTail:\n{dfA.tail()}')
eda(df)
# Is Null: 912

# description          0.014809

len(df[df.description.isna()])
df.description = df.description.fillna('no-discription')
eda(df)
import seaborn as sns

import matplotlib.pyplot as plt
g = df[['countryId', 'country']].groupby('country').count()

a = df[['likes', 'dislikes', 'view_count', 'country']].groupby('country').sum()
g
sns.barplot(x=g.index, y=g.countryId).set_title('Number videos WW')
a
sns.barplot(x=a.index, y=a.view_count).set_title('Sum of Views by country')
sns.barplot(x=a.index, y=a.likes).set_title('Sum of likes by country')
sns.barplot(x=a.index, y=a.dislikes).set_title('Sum of dislikes by country')
br = df.query("country == 'BR' or country == 'US'")
def sepColumns(dataset):

    num = []

    cat = []

    for i in dataset.columns:

        if dataset[i].dtype == 'object':

            cat.append(i)

        else:

            num.append(i)

    return num, cat
num, categ = sepColumns(br)
br[num].describe()
br[['title', 'dislikes']].groupby('title').sum().sort_values(by='dislikes', ascending=False).head()
br[['title', 'likes']].groupby('title').sum().sort_values(by='likes', ascending=False).head()
br[['title', 'view_count']].groupby('title').sum().sort_values(by='view_count', ascending=False).head()
dashbr=br[['title', 'view_count', 'likes', 'dislikes']].groupby('title').sum().sort_values(by='view_count', ascending=False).head()

dashbr
br.sample()
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

        sns.heatmap(corr,

                    annot=True, fmt='.2f', vmin=-1, vmax=1, linewidth=0.01,

                    linecolor='black', cmap='RdBu_r'

                    )

        plt.title('Correlations between features w/ target')

        plt.show()

    return features
varTarget = 'likes'
varFeatures = correlation(br, varTarget, 0.5)
def removeOutliers(out, varTarget):

    print('\nOutliers\nRemoving ...', end='')

    cidgrp = out[varTarget]

    print('..', end='')

    # quantiles

    qtl1 = cidgrp.quantile(.25)  

    qtl3 = cidgrp.quantile(.75)

    print('..', end='')

    # calculating iqr

    iqr = qtl3 - qtl1

    print('..', end='')



    # creating limits

    baixo = qtl1 - 1.5 * iqr

    alto = qtl3 + 1.5 * iqr

    print('..', end='')



    # removing outliers

    novodf = pd.DataFrame()

    print('..', end='')



    limites = out[varTarget].between(left=baixo, right=alto, inclusive=True)

    novodf = pd.concat([novodf, out[limites]])



    print('.....Done')



    return novodf
noOut = removeOutliers(br, varTarget)
# Two subplots

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,5))

sns.boxplot(x=br[varTarget], ax=ax1).set_title('Original')

sns.boxplot(x=noOut[varTarget], ax=ax2).set_title('Original No outliers')
print(br[varTarget].describe())

sns.barplot(x=br[varTarget].describe().index[1:], y=br[varTarget].describe().values[1:])
print(noOut[varTarget].describe())

sns.barplot(x=noOut[varTarget].describe().index[1:], y=noOut[varTarget].describe().values[1:])
varFeatures = correlation(noOut, varTarget, 0.5)
# ML Algoritmos

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.dummy import DummyRegressor



# ML selecao de dados de treino e teste

from sklearn.model_selection import train_test_split

# calcular o menor erro medio absoluto entre 2 dados apresentados

from sklearn.metrics import mean_absolute_error
regressors = [

        DecisionTreeRegressor(),

        RandomForestRegressor(),

#         SVR(),

#         LinearRegression(),

#         GradientBoostingRegressor(),

#         PoissonRegressor(),

#         DummyRegressor(),

#         LogisticRegression(),

#         GaussianNB()

    ]
X = noOut[varFeatures]

y = noOut[varTarget]

Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=42)
reg = []

mae = []

sco = []

for regressor in regressors:

    modelo = regressor

    modelo.fit(Xtreino, np.array(ytreino))

    sco.append(modelo.score(Xtreino, ytreino))

    previsao = modelo.predict(Xteste)

    mae.append(round(mean_absolute_error(yteste, previsao), 2))

    reg.append(regressor)
meuMae = pd.DataFrame(columns=['Regressor', 'mae', 'score'])

meuMae['Regressor'] = reg

meuMae['mae'] = mae

meuMae['score'] = sco
meuMae = meuMae.sort_values(by='score', ascending=False)

meuMae
meuMae["Regressor"].values[0]
model = meuMae["Regressor"].values[0]

x = noOut['view_count']

y = noOut[varTarget]

model.fit(np.array(x).reshape(-1, 1), y)
# what is the prediction to 1mi views?

valFeatures = [1000000]

predict = float(model.predict([valFeatures]))
print(f'Summary:\n'

          f'Regs analyzed: {len(noOut)}\n'

          f'ML applied: {meuMae["Regressor"].values[0]}\n'

          f'Features analyzed:')



print(f' - {varFeatures[0]}: {valFeatures[0]}')



print(f"Predicted likes: {predict:.0f} ")
noOut[noOut.view_count > 1000000][['view_count', 'likes']].describe()
go = noOut[['countryId', 'country']].groupby('country').count()

ao = noOut[['likes', 'dislikes', 'view_count', 'country']].groupby('country').sum()
sns.barplot(x=go.index, y=go.countryId).set_title('Number videos WW')
sns.barplot(x=ao.index, y=ao.view_count).set_title('Sum of Views by country')
sns.barplot(x=ao.index, y=ao.likes).set_title('Sum of likes by country')
sns.barplot(x=ao.index, y=ao.dislikes).set_title('Sum of dislikes by country')