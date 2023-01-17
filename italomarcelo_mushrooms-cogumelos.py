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
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')



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
eda(df)
df.head()
import seaborn as sns

import matplotlib.pyplot as plt
uniques = df.apply(pd.Series.nunique)
uniques
fig, ax1 = plt.subplots( sharey=True, figsize=(15,5))

sns.barplot(x=uniques.index, y=uniques.values, ax=ax1).set_title('Uniques by Column')

plt.xticks(rotation=80)
mushClass = df['class'].unique()
mushClass
ed = df[df['class'] == mushClass[1]]['class'].shape[0]

ined = df[df['class'] == mushClass[0]]['class'].shape[0]
plt.pie([ed,ined], labels=['Edible', 'Poisonous'], autopct='%1.1f%%')


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.dummy import DummyRegressor



# ML train & test data selection

from sklearn.model_selection import train_test_split

# mae metric

from sklearn.metrics import mean_absolute_error
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
vuCat = []

for c in cat:

    v = df[c].unique().tolist()

    x = {c: v}

    vuCat.append(v)

print(vuCat)
pos=0

for vc in cat:

    col = vc.replace('-','_')

    newCol = f'{col}_N'

    df[newCol] = df[vc].apply(lambda x: vuCat[pos].index(x))

    pos += 1

    print(newCol)
df
num, _ = sepColumns(df)

newDf = df[num]
newDf
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
varTarget = 'class_N'

varFeatures = correlation(newDf, varTarget, 0.1)
X = newDf[varFeatures]

y = newDf[varTarget]

Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=42)
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
df[varFeatures].sample(5)
for vf in varFeatures:

    var = vf.replace('_N','')

    var = var.replace('_','-')

    print(var, cat.index(var), vuCat[cat.index(var)])
# valFeatures = cap-surface == 'f'  <==> cap_surface_N == 2

#               stalk-shape == 'e'  <==> stalk_shape_N == 0

#               ring-number == 't'  <==> ring_number_N == 1

varFeaturesP = ['cap_surface_N', 'stalk_shape_N', 'ring_number_N']

valFeaturesP = [2, 0, 1]
model = meuMae["Regressor"].values[0]

x = newDf[varFeaturesP]

y = newDf[varTarget]

model.fit(x, y)

predict = model.predict([valFeaturesP])
print(f'Summary:\n'

          f'Regs analyzed: {len(newDf)}\n'

          f'ML applied: {meuMae["Regressor"].values[0]}\n'

          f'Features analyzed:')

for p in range(0, len(varFeaturesP)):

    print(f' - {varFeaturesP[p]}: {valFeaturesP[p]}')



print(f"Predicted mushroom class: {vuCat[0][predict[0]]} ")
view = []

view = varFeatures[:]

view.append('class')

view
df.query('cap_surface_N == 2 and stalk_shape_N == 0 and ring_number_N == 1')[view].sample(15)