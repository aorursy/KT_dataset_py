import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from tabulate import tabulate

import os

files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        files.append(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/top-cars-sales-20182020-fictitious/topCarsSales.csv')
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
newdf = pd.DataFrame()

newdf['id']=df['id']

newdf['date']=pd.to_datetime(df['date'])

newdf['car']=df['car']

newdf['colorCar'] = df['car_color']

newdf['colorSeat'] = df['seat_color']

newdf['price']=df['value (US$ mi)']

newdf['priceOff'] = df['value off (US$ mi)']

newdf['discount']=df['discount (%)']

newdf['total']=df['total (US$ mi)']

newdf['salesperson']=df['salesperson']

newdf['city']=df['city']

newdf['country']=df['country']

newdf.head(3)
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

        plt.subplots()

        sns.heatmap(corr,

                    annot=True, fmt='.2f', vmin=-1, vmax=1, linewidth=0.01,

                    linecolor='black', cmap='RdBu_r'

                    )

        plt.title(title)

        plt.show()

    

    return features
varTarget = 'total'

varFeatures = correlation(dfA=newdf, varT=varTarget, minValue=0.1, showGraphic=True)


newdf['year']=pd.DatetimeIndex(newdf['date']).year

newdf['month']=pd.DatetimeIndex(newdf['date']).month

newdf.sample(10)
def sepColumns(dataset):

    num = []

    cat = []

    for i in dataset.columns:

        if dataset[i].dtype == 'object':

            cat.append(i)

        else:

            num.append(i)

    return num, cat
num, cat = sepColumns(newdf)

num, cat
# lower case for all

for x in cat:

    newdf[x] = newdf[x].str.lower()
newdf.country.unique().tolist()

newdf.sample(10)
car = newdf.car.unique().tolist()

seller = newdf.salesperson.unique().tolist()

city = newdf.city.unique().tolist()

country = newdf.country.unique().tolist()

colorcar = newdf.colorCar.unique().tolist()

colorseat = newdf.colorSeat.unique().tolist()

car, city, country, seller, colorcar, colorseat
# replace city newyouk to new york

newdf['city'] = newdf['city'].apply(lambda x: 'new york' if x == 'newyouk' else x)

city = newdf.city.unique().tolist()

city
newdf['carNum'] = newdf['car'].apply(lambda x: car.index(x))

newdf['salespersonNum'] = newdf['salesperson'].apply(lambda x: seller.index(x))

newdf['cityNum'] = newdf['city'].apply(lambda x: city.index(x))

newdf['countryNum'] = newdf['country'].apply(lambda x: country.index(x))

newdf['colorCarNum'] = newdf['colorCar'].apply(lambda x: colorcar.index(x))

newdf['colorSeatNum'] = newdf['colorSeat'].apply(lambda x: colorseat.index(x))
newdf.sample(10)
sns.pairplot(newdf[num])
varTarget = 'total'

varFeatures = correlation(dfA=newdf, varT=varTarget, minValue=0.1, showGraphic=True)
varFeatures = ['year','carNum']

newdf[['year','price', 'total']].describe()
sns.set(style="ticks")

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(15,5))

col = 0

for var in varFeatures:

    data = newdf.pivot_table(index=var, values=varTarget, aggfunc="mean")

    sns.barplot(x=data.index, y=data[varTarget], ax=ax[col]).set_title(f'Feature {var}')

    col += 1

plt.show()
# source image https://publiclab.org/notes/mimiss/06-18-2019/creating-a-boxplot-to-identify-outliers-using-codap

from IPython.display import Image

Image('https://publiclab.org/system/images/photos/000/032/980/original/Screen_Shot_2019-06-18_at_10.27.45_AM.png')
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
noOut = removeOutliers(newdf, varTarget)

# Is there outlier?

sns.set(style="whitegrid")

# Two subplots

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,5))

sns.boxplot(x=newdf[varTarget], ax=ax1).set_title('Original')

sns.boxplot(x=noOut[varTarget], ax=ax2).set_title('Original No outliers')





# yep... but..
noOut = removeOutliers(noOut, varTarget)

# Is there outlier?

sns.set(style="whitegrid")

# Two subplots

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,5))

sns.boxplot(x=newdf[varTarget], ax=ax1).set_title('Original')

sns.boxplot(x=noOut[varTarget], ax=ax2).set_title('Original No outliers')
len(newdf.car.unique()), newdf.car.unique()
len(noOut.car.unique()), noOut.car.unique()
newdf[newdf.car == 'bugatti la voiture noire']['total'].describe()
newdf[newdf.car == 'rolls-royce sweptail']['total'].describe()
newdf.total.describe(), noOut.total.describe()
sns.barplot(x=newdf.total.describe().index[1:], y=newdf.total.describe().values[1:])
sns.barplot(x=noOut.total.describe().index[1:], y=noOut.total.describe().values[1:])
# subplots varT x varFeatures

# year

sns.set(style="ticks")



fig, ax = plt.subplots(ncols=2, figsize=(15,6))

for varf in varFeatures:

    sns.scatterplot(x=varTarget, y='year', data=newdf, ax=ax[0]).set_title('Original')

    sns.scatterplot(x=varTarget, y='year', data=noOut, ax=ax[1]).set_title('Original no Outliers')
# subplots varT x varFeatures

# discount

sns.set(style="ticks")



fig, ax = plt.subplots(ncols=2, figsize=(15,6))

for varf in varFeatures:

    sns.scatterplot(x=varTarget, y='discount', data=newdf, ax=ax[0]).set_title('Original')

    sns.scatterplot(x=varTarget, y='discount', data=noOut, ax=ax[1]).set_title('Original no Outliers')
varFeatures = correlation(noOut, varT=varTarget, minValue=0.02, showGraphic=True)
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
# regressors list

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
varFeatures
X = noOut[varFeatures[2:]]

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
car, varFeatures[2:], f'Best Regressor: {meuMae["Regressor"].values[0]}'
varFeatures = ['year', 'carNum']

valFeatures = [2020, car.index('mclaren p1 lm')]

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



print(f"Predicted value: US${predict:.2f} mi")

# cars with the same setup in the dataset 

noOut.query(f'year == 2020 and carNum == 1')[['carNum', 'year', 'total']].describe()