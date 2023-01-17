from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

df1 = pd.read_csv('/kaggle/input/X_data.csv', delimiter=',')
def analiseData(dfAnalisado, describe=False, desc='EDA', variables=[0]):

    print(desc)

    if describe:

        if variables[0]:

            dfAnalisado = dfAnalisado[variables]

        print(f'\nShape:\n{dfAnalisado.shape}')

        print(f'\nHead:\n{dfAnalisado.head()}')

        print(f'\nSample:\n{dfAnalisado.sample(5)}')

        print(f'\nTail:\n{dfAnalisado.tail()}')

        print(f'\nDescribe:\n{dfAnalisado.describe()}')

        print(f'\nIs Null:\n{dfAnalisado.isnull().sum()}')

        print(dfAnalisado.info())

         

    else:

        print(f'\nSamples: \n',dfAnalisado.sample(10))

    print(len(dfAnalisado), ' rows')
analiseData(df1, desc='EDA only 10 samples')
analiseData(df1, describe=True, desc='Retinopath EDA complete')
analiseData(df1, describe=True, desc='Retinopath EDA only variables', 

            variables=['Systolic_BP','Diastolic_BP'])
idealbp = df1[(df1.Systolic_BP.between(89.5,120.5)) & (df1.Diastolic_BP.between(59.5, 80.5))]

hbp = df1[(df1.Systolic_BP > 139.5) & (df1.Diastolic_BP > 89.5)]

lbp = df1[(df1.Systolic_BP < 89.5) & (df1.Diastolic_BP < 59.6)]
print('BP ideal: ', len(idealbp), 'registers')

print('BP high: ', len(hbp), 'registers')

print('BP low: ', len(lbp))
df1.sample(5)
analiseData(df1, describe=True, desc='Retinopath EDA only variables', 

            variables=['Cholesterol'])
plotScatterMatrix(df1, 12, 10)
plotScatterMatrix(idealbp, 12, 10)
df2 = pd.read_csv('/kaggle/input/y_data.csv', delimiter=',')
analiseData(df2, describe=True)
from tabulate import tabulate

import seaborn as sns, matplotlib.pyplot as plt

import warnings

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

warnings.filterwarnings('ignore')
# choosing best model

def melhorModelo(mlData, mlAlgoritmo, varFeatures, varTarget, exibe=False, exibeGrafico=False):



    print(f'\n\nAnalizing Regressors ML')



    X = mlData[varFeatures]

    y = mlData[varTarget]



    Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=123)



    # applying regressors

    reg = []

    mae = []

    sco = []



    for regressor in mlAlgoritmo:

        modelo = regressor



        # training model

        try:

            modelo.fit(Xtreino, ytreino)

            sco.append(modelo.score(Xtreino, ytreino))

            previsao = modelo.predict(Xteste)

            mae.append(round(mean_absolute_error(yteste, previsao), 2))

            reg.append(regressor)

        except:

            pass



    meuMae = pd.DataFrame(columns=['Regressor', 'mae', 'score'])

    meuMae['Regressor'] = reg

    meuMae['mae'] = mae

    meuMae['score'] = sco

    meuMae = meuMae.sort_values(by='mae', ascending=True)

    if exibe:

        print(tabulate(meuMae, headers='keys', tablefmt='psql'))



    if exibeGrafico:

        try:

            resultado = meuMae.values[0][0].predict(Xteste)

            ax1 = plt.subplot(322)

            ax1.set_title('Distribution Prices')

            sns.distplot(resultado)

            ax2 = plt.subplot(321)

            ax2.set_title('Results - Prices')

            sns.boxplot(resultado)

        



            ax3 = plt.subplot(312)

            ax3.set_title('Performance - Regressors ML')

            g = sns.barplot(x=meuMae['Regressor'], y=meuMae['mae'])

            g.set_xticklabels(ax3.get_xticklabels(), rotation=30)



            ax4 = plt.subplot(312)

            ax4.set_title('Performance - Score - Regressors')

            g = sns.barplot(x=meuMae['Regressor'], y=meuMae['score'])

            g.set_xticklabels(ax4.get_xticklabels(), rotation=30)

            plt.show()

        except:

            pass



    return meuMae

# regressor used

def mlAlgoritmos():

    regressores = [

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

    return regressores
df = df1.copy()

df['y'] = df2.y

df.sample(5)
def viewGraph(dataset):

    g = sns.FacetGrid(dataset, col="y")

    g.map(plt.scatter, "Age", "Cholesterol", alpha=.7)

    g.add_legend();

    g = sns.FacetGrid(dataset, hue="y", palette='Set1', height=5)

    g.map(plt.scatter, "Age", "Cholesterol", s=50, alpha=.7, linewidth=.5, edgecolor="white")

    g.add_legend();
viewGraph(df)
def correlacao(df, varT, xpoint=-0.5, showGraph=True):

    corr = df.corr()

    print(f'\nFeatures correlation:\n'

          f'Target: {varT}\n'

          f'Reference.: {xpoint}\n'

          f'\nMain features:')

    if showGraph:

        sns.heatmap(corr,

                    annot=True, fmt='.2f', vmin=-1, vmax=1, linewidth=0.01,

                    linecolor='black', cmap='RdBu_r'

                    )

        plt.title('Correlations between features w/ target')

        plt.show()



    corrs = corr[varT]

    features = []

    for i in range(0, len(corrs)):

        if corrs[i] > xpoint and corrs.index[i] != varT:

            print(corrs.index[i], f'{corrs[i]:.2f}')

            features.append(corrs.index[i])

    return features
varT = 'y'

varF = correlacao(df, varT, xpoint=0.3, showGraph=True)
def removeOutliers(out, varTarget):

    print('\nRemovendo Outliers')

    cidgrp = out[varTarget]



    # criando quantis

    qtl1 = cidgrp.quantile(.25)  # exiba o valor da variavel

    qtl3 = cidgrp.quantile(.75)



    # calculando a diferenca entre os dois quantis, conhecido como interquartile range

    iqr = qtl3 - qtl1

    # print(qtl1, qtl3, iqr)



    # gerando os limites

    baixo = qtl1 - 1.5 * iqr

    alto = qtl3 + 1.5 * iqr



    # remover os outliers

    novodf = pd.DataFrame()



    limites = out[varTarget].between(left=baixo, right=alto, inclusive=True)

    novodf = pd.concat([novodf, out[limites]])



    print('Done')



    return novodf.copy()
df = removeOutliers(df, varT)
viewGraph(df)
bestML = melhorModelo(mlData=df, mlAlgoritmo=mlAlgoritmos(),

                               varTarget=varT, varFeatures=varF,

                               exibe=True, exibeGrafico=False)
bestML['Regressor'][0]
def previsao(dfEscolhida, mlAlgoritmo, varFeatures, valueFeatures, varTarget, desc=''):



    x = dfEscolhida[varFeatures]

    y = dfEscolhida[varTarget]



    modelo = mlAlgoritmo

    modelo.fit(x, y)



    previsao = float(modelo.predict([valueFeatures]))

    cond = ''



    print(f'Summary:\n'

          f'Regs analyzed: {len(dfEscolhida)}\n'

          f'ML applied: {mlAlgoritmo}\n'

          f'Features analyzed:')

    for i in range(0, len(varFeatures)):

        print(f' - {varFeatures[i]}: {valueFeatures[i]}')

        cond += f" and `{varFeatures[i]}` == {valueFeatures[i]}"

        

    if previsao:

        descPrev = 'Positive'

    else:

        descPrev = 'Negative'

    

    print(f"Predicted value: {previsao} [{descPrev}] \n")



   
df.sample(1)

idealbp
# example creating condition

# features name

varFeaturesFilters = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol']

# features values 

valueFeaturesFilters = [49.93829, 109.893662, 90.019716, 106.02519]
previsao(dfEscolhida=df, mlAlgoritmo=bestML['Regressor'][0],

         varFeatures=varFeaturesFilters,

         valueFeatures=valueFeaturesFilters, varTarget=varT,

         desc='\nPredicting price')
varFeaturesFilters = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol']

# features values 

valueFeaturesFilters = [54, 90, 75, 108]
previsao(dfEscolhida=df, mlAlgoritmo=bestML['Regressor'][0],

         varFeatures=varFeaturesFilters,

         valueFeatures=valueFeaturesFilters, varTarget=varT,

         desc='\nPredicting price')
varFeaturesFilters = ['Age', 'Cholesterol']

# features values 

valueFeaturesFilters = [35, 108]
previsao(dfEscolhida=df, mlAlgoritmo=bestML['Regressor'][0],

         varFeatures=varFeaturesFilters,

         valueFeatures=valueFeaturesFilters, varTarget=varT,

         desc='\nPredicting price')