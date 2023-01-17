# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor
#Pegando arquivos .csv 

df = pd.read_csv('../input/dataset_treino.csv',sep=',', encoding='ISO-8859-1')

dfTest = pd.read_csv('../input/dataset_teste.csv',sep=',', encoding='ISO-8859-1')

dfLojas = pd.read_csv('../input/lojas.csv',sep=',', encoding='ISO-8859-1')
#Ajustando as categorias de StateHoliday

df['StateHoliday'] = df['StateHoliday'].astype(str)
#Abordagem 1: Eliminando as linhas NaN para a Store com valores de Open nulos(622)

#dfTest = dfTest[pd.notnull(dfTest['Open'])]



#Abordagem2: Considerar que a loja está aberta, já que aos domingos nessa situação, a loja está fechada!

#dfTest['Open'] = dfTest['Open'].fillna(1)



#Abordagem3: Considerar que a loja está fechada por algum motivo

dfTest['Open'] = dfTest['Open'].fillna(0)
#Alterar tipo coluna Open no df e dfTest

df['Open'] = df.Open.astype('int64')

dfTest['Open'] = dfTest.Open.astype('int64')
#Incluindo colunas de mês e dia nos dataframes

df['Year'] = pd.to_datetime(df['Date']).apply(lambda x: x.year)

df['Month'] = pd.to_datetime(df['Date']).apply(lambda x: x.month)

df['Day'] = pd.to_datetime(df['Date']).apply(lambda x: x.day)

dfTest['Year'] = pd.to_datetime(dfTest['Date']).apply(lambda x: x.year)

dfTest['Month'] = pd.to_datetime(dfTest['Date']).apply(lambda x: x.month)

dfTest['Day'] = pd.to_datetime(dfTest['Date']).apply(lambda x: x.day)
#Tratamento para novas variaveis relacionadas a data(fim de semana, quartil, dias desde o inicio)...

menor_dia_df = df['Date'].min()

df['DaysSinceBeginning'] = pd.to_datetime(df['Date']).apply(lambda x: (x-pd.to_datetime(menor_dia_df)).days)

df['Weekend'] = df['DayOfWeek']>4

df['Quarter'] = pd.to_datetime(df['Date']).apply(lambda x: int((x.month-1)/3)+1)

menor_dia_dfTest = dfTest['Date'].min()

dfTest['DaysSinceBeginning'] = pd.to_datetime(dfTest['Date']).apply(lambda x: (x-pd.to_datetime(menor_dia_dfTest)).days)

dfTest['Weekend'] = dfTest['DayOfWeek']>4

dfTest['Quarter'] = pd.to_datetime(dfTest['Date']).apply(lambda x: int((x.month-1)/3)+1)
#Retirar Store, date e Customers do Dataframe (e Id do Dataframe de teste)

dfAjustado = df.drop(['Store', 'Date', 'Customers'], axis=1)

dfAjustadoTest = dfTest.drop(['Store', 'Date'], axis=1)
#Transformando coluna 'Weekend' em binaria

df['Weekend'] = df['Weekend'].astype(int)

dfTest['Weekend'] = dfTest['Weekend'].astype(int)
#Pegando os dados categóricos do Dataframe e transformando em dummies(one-hot encoded)

dfAjustado = pd.get_dummies(dfAjustado, columns=['DayOfWeek', 'StateHoliday','Quarter'])

dfAjustadoTest = pd.get_dummies(dfAjustadoTest, columns=['DayOfWeek', 'StateHoliday','Quarter'])
#Colocando os Dados na mesma escala 



colsDFSel = ['Open', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day',

       'DaysSinceBeginning', 'Weekend', 'DayOfWeek_1', 'DayOfWeek_2',

       'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6',

       'DayOfWeek_7', 'StateHoliday_0', 'StateHoliday_a', 'Quarter_3', 'Sales']



             

colsDFSELTest = ['Open', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day',

       'DaysSinceBeginning', 'Weekend', 'DayOfWeek_1', 'DayOfWeek_2',

       'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6',

       'DayOfWeek_7', 'StateHoliday_0', 'StateHoliday_a', 'Quarter_3']



colTrain = colsDFSel

dfMLTrain = dfAjustado[colTrain]

arrayTrain = dfMLTrain.values



#Fazendo split nos dados de treino

XTrain = arrayTrain[:,0:18]

YTrain = arrayTrain[:,18]



#Criando escala

scaler = MinMaxScaler(feature_range = (0, 1))

rescaledXTrain = scaler.fit_transform(XTrain)



#Dados na escala

print(rescaledXTrain[0:5,:])





#================ TESTE ==========================

colTest = colsDFSELTest

dfMLTest = dfAjustadoTest[colTest]

arrayTest = dfMLTest.values

XTEST = arrayTest[:,0:18]

rescaledXTest = scaler.fit_transform(XTEST)

print(rescaledXTest[0:5,:])

#==================================================
#Dividindo os dados de treino com train test split

X_train, X_test, y_train, y_test = train_test_split(rescaledXTrain, YTrain, test_size=0.33, random_state=42)
#RMSPE

def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def resultado_rsmpe(y, yhat):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe
#Modelo Gradient Boosting Regressor

modelGBR = GradientBoostingRegressor(n_estimators=100, loss='ls', random_state=42)

modelGBR.fit(rescaledXTrain, YTrain)

print(round(modelGBR.score(rescaledXTrain, YTrain) * 100, 2))



# Predictions

YPredGBR = modelGBR.predict(rescaledXTest)
YPredGBR_2 = modelGBR.predict(X_train)

res3 = resultado_rsmpe(y_train, YPredGBR_2)

print (res3)
#Gerando o resultado para o Submission File

resultado = np.around(YPredGBR, decimals=2)
#Criando Submission file

submission = pd.DataFrame({

        "ID": dfAjustadoTest['Id'],

        "Sales": resultado

    })



submission.to_csv('submission_GBR.csv', index=False)