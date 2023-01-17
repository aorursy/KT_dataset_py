import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
brands = []
df = pd.DataFrame()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        b = filename.split('.')
        #if filename[:7] != 'unclean': # all brands
        if b[0] == 'audi': # only audy brand
            brands.append(b[0])
            x = pd.read_csv(os.path.join(dirname, filename))
            x['brand'] = b[0]
            df = df.append(x)
            print(os.path.join(dirname, filename))
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
import os

warnings.filterwarnings('ignore')
def eda(dataset, title='EDA'):
    print(f'=={title}==')
    print('INFO \n')
    print(tabulate(dataset.info(), headers='keys', tablefmt='psql'))
    print('\nHEAD \n', tabulate(dataset.head(), headers='keys', tablefmt='psql'))
    print('\nTAIL \n', tabulate(dataset.tail(), headers='keys', tablefmt='psql'))
    print('\nDESCRIBE \n', tabulate(dataset.describe(), headers='keys', tablefmt='psql'))
    print('\n5 SAMPLES \n', tabulate(dataset.sample(5), headers='keys', tablefmt='psql'))
    print('\nNULLS QTY \n', dataset.isnull().sum())
    print('\nSHAPE \n', tabulate([dataset.shape], headers=['rows', 'cols'], tablefmt='psql'))
eda(df)
df = df.fillna(0)
df.isnull().sum()
models = df['model'].unique()
transmissions = df['transmission'].unique()
fuelTypes = df['fuelType'].unique()
def txtToNum(txt, mylist):
    return list(mylist).index(txt)

df['brandN'] = df['brand'].apply(lambda field: txtToNum(field, brands))
df['modelN'] = df['model'].apply(lambda field: txtToNum(field, models))
df['transmissionN'] = df['transmission'].apply(lambda field: txtToNum(field, transmissions))
df['fuelTypeN'] = df['fuelType'].apply(lambda field: txtToNum(field, fuelTypes))
df.sample(10)
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
varT = 'price'
varF = correlacao(df, varT, xpoint=0.1, showGraph=True)
print(f'Target: {varT}\n'
      f'Features: {list(varF)}')
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
# choosing best model
def melhorModelo(mlData, mlAlgoritmo, varFeatures, varTarget, exibe=False, exibeGrafico=False):
    # vamos selecionar os dados de treino e testes

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

bestML = melhorModelo(mlData=df, mlAlgoritmo=mlAlgoritmos(),
                               varTarget=varT, varFeatures=varF,
                               exibe=True, exibeGrafico=False)
bestML['Regressor'][0] # 0=Decision Tree, 1=Random Forester .... 8=GaussianNB
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
    print(f'Condition: {cond[5:]}')
    print(f"Predicted value: ${previsao:.2f} \n"
          f"Avg value: ${dfEscolhida.query(cond[5:])[varTarget].mean():.2f}\n\n")

    variaveis = varFeatures
    variaveis.append(varTarget)
    print(
        tabulate(
            dfEscolhida.query(cond[5:])[variaveis].head(10).sort_values(by=varTarget, ascending=False),
            headers='keys', tablefmt='psql'
        ), f"\n{dfEscolhida.query(cond[5:])[varTarget].shape}")

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

    # print(novodf[['city','rooms', 'rent amount (R$)']])

    return novodf.copy()
print('Features trained:')
print(f'modelN: ')
print(tabulate([range(0, len(brands)), brands], tablefmt='psql'))
print(f'transmissionN: ')
print(tabulate([range(0, len(transmissions)), transmissions], tablefmt='psql'))
print(f'fuelTypeN: ')
print(tabulate([range(0, len(fuelTypes)), fuelTypes], tablefmt='psql'))

# audi Q3 2017
df.query("brandN == 0 and modelN == 4 and year == 2017 and transmissionN == 1").sample(10)
# example creating condition
# features name
varFeaturesFilters = ['brandN', 'modelN', 'transmissionN', 'year'] 
# features values 
valueFeaturesFilters = [0, 4,  1, 2017]
pricesNoOutliers = removeOutliers(df, varTarget=varT)
previsao(dfEscolhida=pricesNoOutliers, mlAlgoritmo=bestML['Regressor'][1],
         varFeatures=varFeaturesFilters,
         valueFeatures=valueFeaturesFilters, varTarget=varT,
         desc='\nPredicting price')