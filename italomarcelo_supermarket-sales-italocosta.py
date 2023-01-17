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
df = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
def edaFromData(dfA, allEDA=False, desc='Exploratory Data Analysis'):

    print('Explorando os dados')

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nIs Null:\n{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if allEDA:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))

        

        #print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead dos dados:\n{dfA.head()}')

        print(f'\nSamples dos dados:\n{dfA.sample(2)}')

        print(f'\nTail dos dados:\n{dfA.tail()}')
edaFromData(df)
df.head(3)
import seaborn as sns

import matplotlib.pyplot as plt

from tabulate import tabulate
def correlation(dfA, varT, minValue=0.5, showGraphic=True, title='Correlation between variables'):

    corr = dfA.corr()

    print(f'\nAnalysing features:\n'

          f'Target: {varT}\n'

          f'minValue de ref.: {minValue}\n'

          f'\nMain Features:')

    corrs = corr[varT]

    features = []

    for i in range(0, len(corrs)):

        if corrs[i] > minValue and corrs.index[i] != varT:

            print(corrs.index[i], f'{corrs[i]:.2f}')

            features.append(corrs.index[i])

    if showGraphic:

        fig = plt.subplots(figsize=(15,8))

        sns.heatmap(corr,

                    annot=True, fmt='.2f', vmin=-1, vmax=1, linewidth=0.01,

                    linecolor='black', cmap='RdBu_r'

                    )

        plt.title(title)

        plt.show()

    

    return features
varTarget = 'Quantity'

varFeatures = correlation(dfA=df, varT=varTarget)
def sepColumns(dataset):

    num = []

    cat = []

    for i in dataset.columns:

        if dataset[i].dtype == 'object':

            cat.append(i)

        else:

            num.append(i)

    return num, cat
num, cat = sepColumns(df)

num, cat
for x in cat:

    df[x] = df[x].str.lower()

df[cat].sample(3)
branch = df.Branch.unique().tolist()

city = df.City.unique().tolist()

customerType = df['Customer type'].unique().tolist()

gender = df.Gender.unique().tolist()

productLine = df['Product line'].unique().tolist()

payment = df.Payment.unique().tolist()
df['year'] = pd.DatetimeIndex(df['Date']).year

df['month'] = pd.DatetimeIndex(df['Date']).month
df['branchNum'] = df['Branch'].apply(lambda x: branch.index(x))

df['cityNum'] = df['City'].apply(lambda x: city.index(x))

df['customerTypeNum'] = df['Customer type'].apply(lambda x: customerType.index(x))

df['genderNum'] = df['Gender'].apply(lambda x: gender.index(x))

df['productLineNum'] = df['Product line'].apply(lambda x: productLine.index(x))

df['paymentNum'] = df['Payment'].apply(lambda x: payment.index(x))

df.sample(3)
num, cat = sepColumns(df)
varFeatures = correlation(dfA=df[num], varT=varTarget, minValue=0.5, showGraphic=True)
df[varFeatures].describe()
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
noOut = removeOutliers(df, varTarget)

# Is there outlier?

sns.set(style="whitegrid")

# Two subplots

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,5))

sns.boxplot(x=df[varTarget], ax=ax1).set_title('Original')

sns.boxplot(x=noOut[varTarget], ax=ax2).set_title('Original No outliers')
print(df[varTarget].describe())

sns.barplot(x=df[varTarget].describe().index[1:], y=df[varTarget].describe().values[1:])
print(noOut[varTarget].describe())

sns.barplot(x=noOut[varTarget].describe().index[1:], y=noOut[varTarget].describe().values[1:])
varTarget = 'Total'

varFeatures = correlation(noOut, varT=varTarget, minValue=0.1, showGraphic=True)
# ML Algorithms sklearn

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

        SVR(),

        LinearRegression(),

        GradientBoostingRegressor(),

        PoissonRegressor(),

        DummyRegressor(),

        LogisticRegression(),

        GaussianNB()

    ]
X = noOut[varFeatures]

y = noOut[varTarget]

Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=123)
reg = []

mae = []

sco = []

for regressor in regressors:

    modelo = RandomForestRegressor()

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
f'Best Regressor: {meuMae["Regressor"].values[0]}'
noOut.sample(3)
varFeatures = ['genderNum', 'customerTypeNum']

valFeatures = [gender.index('male'), customerType.index('member')]

varFeatures, valFeatures
model = meuMae["Regressor"].values[0]

x = noOut[varFeatures]

y = noOut[varTarget]

model.fit(x, y)
predict = float(model.predict([valFeatures]))
print(f'Summary:\n'

          f'Regs analyzed: {len(noOut)}\n'

          f'ML applied: {meuMae["Regressor"].values[0]}\n'

          f'Features analyzed:')



for i in range(0, len(varFeatures)):

    print(f' - {varFeatures[i]}: {valFeatures[i]}')



print(f"Predicted value: US${predict:.2f} ")
noOut.query(f'genderNum == {valFeatures[0]} and customerTypeNum == {valFeatures[1]}')[['genderNum', 'customerTypeNum', 'Total']].describe()