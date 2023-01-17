import numpy as np

import pandas as pd

import statsmodels.api as sm



# Análise descritiva

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as md

import dateutil



# Treinamento e Teste

import statistics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import KFold
arquivo = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv", sep=',', skip_blank_lines=True, decimal=",")

arquivo.dropna(how="all", inplace=True)

arquivo.head(5)
feriados = pd.read_csv("../input/feriados-sao-paulo-2015/feriados_sao_paulo_2015.csv", sep=",")

feriados['Dia'] = pd.to_datetime(feriados['Dia'])

feriados
# Final de semana de texto para 0 ou 1

arquivo['Final de Semana'] = arquivo['Final de Semana'].astype(int)



# Unidade de milhar removida do consumo e transformação para float

arquivo['Consumo de cerveja (litros)'] = arquivo['Consumo de cerveja (litros)'].str.replace('.', '')

arquivo['Consumo de cerveja (litros)'] = arquivo['Consumo de cerveja (litros)'].astype(float)



# Data de string para datetime

arquivo['Data'] = pd.to_datetime(arquivo['Data'])



arquivo
# Criação da coluna de dia da semana

arquivo['Dia da Semana'] = arquivo['Data'].dt.dayofweek

arquivo['Dia da Semana'] = arquivo['Dia da Semana'].map({0:'Seg',1:'Ter',2:'Qua',3:'Qui',4:'Sex',5:'Sab',6:'Dom'})



# Identificação das observações que foram em feriados

arquivo['Feriado'] = [dia in list(feriados['Dia']) for dia in arquivo['Data']]



# Definição de dia útil ou não

arquivo['Dia util'] = ~(arquivo['Final de Semana'] | arquivo['Feriado'])



# Conversão de bool para inteiro para as novas colunas: feriado e dia útil

arquivo['Feriado'] = arquivo['Feriado'].astype(int)

arquivo['Dia util'] = arquivo['Dia util'].astype(int)



arquivo.head(5)
arquivo.shape[0]
def encontraModa(df):

    contagem = df.value_counts()

    valores_repeticoes = dict(contagem[contagem == max(contagem)])

    conector = ''

    moda = ''

    # Converte {38: 2, 49: 2} em "38 (x2), 49 (x2)"

    for valor in valores_repeticoes:

        moda += conector + "{:8.2f}".format(valor)

        conector = ', '

    moda += ' (x' + str(valores_repeticoes[valor]) +')'

    return moda



def analisa(coluna):

    df = arquivo[coluna]

    media = df.mean()

    dp = np.std(df)

    analise = {

        'Coluna': coluna,

        'Média': "{:8.4f}".format(media),

        'Moda': encontraModa(df),

        'Mediana': "{:8.4f}".format(df.median()),

        'Min': "{:8.4f}".format(df.min()),

        'Max': "{:8.4f}".format(df.max()),

        'Desvio padrão': "{:8.4f}".format(dp),

        'Variância': "{:8.4f}".format(dp ** 2),

        'Coeficiente de Var.': "{:8.4f}".format(dp / media)

    }

    return analise



colunas = ['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Consumo de cerveja (litros)']

analises = []

for coluna in colunas:

    analises.append(analisa(coluna))
pd.DataFrame(analises)
fontsize = 16



arquivo['Data'] = arquivo['Data'].astype(str)



fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14,4))

datas = [dateutil.parser.parse(i) for i in arquivo['Data']]

sns.distplot(arquivo['Temperatura Media (C)'],color='y', ax=ax1)

sns.distplot(arquivo['Temperatura Minima (C)'],color='b', ax=ax2)

sns.distplot(arquivo['Temperatura Maxima (C)'],color='r', ax=ax3)

plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(9,4))

datas = [dateutil.parser.parse(i) for i in arquivo['Data']]

sns.distplot(arquivo['Precipitacao (mm)'],color='c', ax=ax1)

sns.distplot(arquivo['Consumo de cerveja (litros)'],color='orange', ax=ax2)

plt.show()



# voltar a conversão para os passos posteriores

arquivo['Data'] = pd.to_datetime(arquivo['Data'])
order = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex',  'Sab', 'Dom']

ax = sns.boxplot(x = 'Dia da Semana', y = 'Consumo de cerveja (litros)', data = arquivo, order=order)

ax.set_xticklabels(

    ax.get_xticklabels(),

    fontdict={'fontsize': fontsize}

);
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.lmplot(x='Temperatura Maxima (C)', y='Consumo de cerveja (litros)', hue='Final de Semana',

           data=arquivo, aspect=1.5, height=6);



plt.grid(ls='-.', lw=.5);
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.lmplot(x='Precipitacao (mm)', y='Consumo de cerveja (litros)', hue='Final de Semana',

           data=arquivo, aspect=1.5, height=6);



plt.grid(ls='-.', lw=.5);

xlim=plt.xlim(-5,100)
corr = arquivo.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True,

    annot=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    fontdict={'fontsize': fontsize + 1},

    horizontalalignment='right'

);

ax.set_yticklabels(

    ax.get_yticklabels(),

#     rotation=45,

    fontdict={'fontsize': fontsize},

#     horizontalalignment='right'

);
corr = arquivo.corr()

abs(corr['Consumo de cerveja (litros)']).sort_values(ascending=False)
variaveis = ['Temperatura Maxima (C)', 'Final de Semana', 'Precipitacao (mm)']

dependente = 'Consumo de cerveja (litros)'



y = arquivo[dependente]



for i in range(1, len(variaveis)+1):



    X = arquivo[variaveis[0:i]]

    X = sm.add_constant(X)



    model = sm.OLS(y, X).fit()

    print(model.summary())

    print('\n\n\n----------------\n\n\n')
R = 300



# Variável independente

X = arquivo[["Temperatura Maxima (C)","Final de Semana","Precipitacao (mm)"]]



# Variável dependente

y = arquivo["Consumo de cerveja (litros)"]



linearRegressor = LinearRegression()



mae = []

mse = []

rmse = []

result = []

model = []



for i in range(1,R):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5)

    

    #Regressão Linear

    model.append(linearRegressor.fit(X_train, y_train))

    y_pred = linearRegressor.predict(X_test)

    

    mae.append(metrics.mean_absolute_error(y_pred, y_test))

    mse.append(metrics.mean_squared_error(y_pred, y_test))

    rmse.append(np.sqrt(metrics.mean_squared_error(y_pred, y_test)))

    

print("MAE medio: ", np.mean(mae), " MAE desv pad: ", np.sqrt(np.var(mae)))

print("MSE medio: ", np.mean(mse), " MSE desv pad: ", np.sqrt(np.var(mse)))

print("RMSE medio: ", np.mean(rmse), " RMSE desv pad: ", np.sqrt(np.var(rmse)))

print("Intercepto ou Coeficiente Linear: ", linearRegressor.intercept_)

print("Coeficiente Angular (slope):", linearRegressor.coef_)



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
# Variável independente

X = arquivo[["Temperatura Maxima (C)","Final de Semana","Precipitacao (mm)"]]



# Variável dependente

Y = arquivo["Consumo de cerveja (litros)"]



linearRegressor = LinearRegression()



kf = KFold(n_splits=7)

kf.get_n_splits(X)



mae = []

mse = []

rmse = []

model = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]



    #Regressão Linear

    model.append(linearRegressor.fit(X_train, y_train))

    y_pred = linearRegressor.predict(X_test)



    mae.append(metrics.mean_absolute_error(y_pred, y_test))

    mse.append(metrics.mean_squared_error(y_pred, y_test))

    rmse.append(np.sqrt(metrics.mean_squared_error(y_pred, y_test)))



print("MAE médio: ", np.mean(mae), " MAE Desvio padrão: ", np.sqrt(np.var(mae)))

print("MSE médio: ", np.mean(mse), " MSE Desvio padrão: ", np.sqrt(np.var(mse)))

print("RMSE médio: ", np.mean(rmse), " RMSE Desvio padrão: ", np.sqrt(np.var(rmse)))

print("Intercepto ou Coeficiente Linear: ", linearRegressor.intercept_)

print("Coeficiente Angular (slope):", linearRegressor.coef_)