# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_covid19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', usecols=[1,3,5,6,7])



df_covid19.head()
df_covid19.info()
df_covid19.shape
df_covid19_brazil = df_covid19[df_covid19['Country/Region'] == 'Brazil']

df_covid19_brazil.reset_index()



df_covid19_brazil.head(20)
#identificando os infectados por data



df_covid19_brazil_infecteds = df_covid19_brazil[['ObservationDate', 'Infected']]

df_covid19_brazil_infecteds['ObservationDate'] = pd.to_datetime(df_covid19_brazil_infecteds['ObservationDate'])

df_covid19_brazil_infecteds.set_index('ObservationDate', inplace=True)

display(df_covid19_brazil_infecteds.head())

display(df_covid19_brazil_infecteds.shape)
plt.figure(figsize=(20,10))

plt.title('Covid no Brasil')

plt.plot(df_covid19_brazil ['ObservationDate'], df_covid19_brazil['Infected'])
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
df_covid19_brazil_infecteds = pd.concat([df_covid19_brazil_infecteds['Infected'],

                                         df_covid19_brazil_infecteds['Infected'].shift(1)], axis=1, keys=['Infected', 'Infected_shifted'])

df_covid19_brazil_infecteds.head(10)
df_covid19_brazil_infecteds2 = df_covid19_brazil_infecteds[1:]



df_covid19_brazil_infecteds2.Mse_baseline = np.sqrt(mean_squared_error(df_covid19_brazil_infecteds2.Infected, df_covid19_brazil_infecteds2.Infected_shifted))

df_covid19_brazil_infecteds2.r2_baseline = r2_score(df_covid19_brazil_infecteds2.Infected, df_covid19_brazil_infecteds2.Infected_shifted)



print(df_covid19_brazil_infecteds2.r2_baseline)

print(df_covid19_brazil_infecteds2.Mse_baseline)
plt.figure(figsize=(10,10))

plt.title('Infected / Infected_shifted')

plt.grid()

plt.plot(df_covid19_brazil_infecteds2.Infected, color='Blue', linewidth=2)

plt.plot(df_covid19_brazil_infecteds2.Infected_shifted, color='orange', linewidth=3, linestyle='--')

plt.legend(['Real', 'Predito'])

plt.show()
X = np.reshape([i for i in range(0, len(df_covid19_brazil_infecteds2['Infected']))],(-1, 1))

y = df_covid19_brazil_infecteds2['Infected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression(n_jobs=-1)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)



print(r2_score(y_test, lr_pred))

print(np.sqrt(mean_squared_error(y_test, lr_pred)))

plt.figure(figsize=(10,10))



plt.title('Predição de séries temporais utilizando Regressão linear')

plt.grid()

plt.plot(X, y, linewidth=2)

plt.plot(X, lr.predict(X), linestyle='--', linewidth=3, color='orange')

plt.xlabel('Dias após o primeiro caso')

plt.ylabel('Infectados')

plt.legend(['Real','Predito'])

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf 

# identificar o parâmetro q

plot_acf(df_covid19_brazil_infecteds2.Infected)

plt.show()
# identificar o parâmetro p

plot_pacf(df_covid19_brazil_infecteds2.Infected)

plt.show()
train_size = int(len(df_covid19_brazil_infecteds2) * 0.70)

df_covid19_brazil_infecteds2_train = df_covid19_brazil_infecteds2.Infected[:train_size].values.reshape(-1,1)

df_covid19_brazil_infecteds2_test = df_covid19_brazil_infecteds2.Infected[train_size:].values.reshape(-1,1)

print('Treino: {}'.format(df_covid19_brazil_infecteds2_train.shape[0]))

print('Teste: {}'.format(df_covid19_brazil_infecteds2_test.shape[0]))
from statsmodels.tsa.arima_model import ARIMA
df_teste_arima = pd.DataFrame([])

rmse = pd.Series([])

ordem = pd.Series([])

r2 = pd.Series([])

aic = pd.Series([])

predicoes = pd.DataFrame([])

c = 0



for i in range(0, 4):

    for j in range(0, 4):

        for k in range(0, 2):

            try:

                

                # instancia o modelo

                df_covid19_brazil_infecteds_model = ARIMA(df_covid19_brazil_infecteds2_train, order=(i,k,j))

                # ajustar o modelo

                df_covid19_brazil_infecteds_model_fit = df_covid19_brazil_infecteds_model.fit()

                # Calcula o AIC

                aic[c] = df_covid19_brazil_infecteds_model_fit.aic

                # realiza a predição

                df_covid19_brazil_infecteds_model_fit_forecast = df_covid19_brazil_infecteds_model_fit.forecast(steps=(len(df_covid19_brazil_infecteds2)-train_size))[0]

                # salva a ordem que está sendo utilizada no ARIMA

                ordem[c] = '({}, {}, {})'.format(i,j,k)

                # Salva o r2

                r2[c] = r2_score(df_covid19_brazil_infecteds2_test, df_covid19_brazil_infecteds_model_fit_forecast)

                # salva o RMSE

                rmse[c] = np.sqrt(mean_squared_error(df_covid19_brazil_infecteds2_test, df_covid19_brazil_infecteds_model_fit_forecast))



                # Salva as prediçoes deste ARIMA

                predicoes.insert(c, ordem[c] , df_covid19_brazil_infecteds_model_fit_forecast)

                

                c += 1

            except:

                continue

                

df_teste_arima = pd.concat([ordem, r2, rmse, aic], axis=1, keys=['Order', 'R2_score', 'RSME', 'AIC']).sort_values(by=['R2_score','RSME', 'AIC'], ascending=False)

df_teste_arima
plt.figure(figsize=(20,15))

if len(df_teste_arima) % 2 == 0:

    for c in range(len(df_teste_arima)):

        plt.subplot(int(len(df_teste_arima)/4),4,c+1)

        plt.title('Arima Ordem: {}'.format(predicoes.columns[c]))

        plt.grid()

        plt.plot(df_covid19_brazil_infecteds2_train, linewidth=2)

        plt.plot([None for i in df_covid19_brazil_infecteds2_train] + [j for j in df_covid19_brazil_infecteds2_test], linewidth=2)

        plt.plot([None for i in df_covid19_brazil_infecteds2_train] + [j for j in predicoes[predicoes.columns[c]]], linestyle='--', color='red', linewidth=3)

        plt.legend(['Treino', 'Teste', 'Predito'])

else: 

    for c in range(len(df_teste_arima)):

        plt.subplot(np.ceil(int(len(df_teste_arima)/3))+1,3,c+1)

        plt.title('Arima Ordem: {}'.format(predicoes.columns[c]))

        plt.grid()

        plt.plot(df_covid19_brazil_infecteds2_train, linewidth=2)

        plt.plot([None for i in df_covid19_brazil_infecteds2_train] + [j for j in df_covid19_brazil_infecteds2_test], linewidth=2)

        plt.plot([None for i in df_covid19_brazil_infecteds2_train] + [j for j in predicoes[predicoes.columns[c]]], linestyle='--', color='red', linewidth=3)

        plt.legend(['Treino', 'Teste', 'Predito'])

plt.tight_layout() 

plt.show()
df_covid19_brazil_infecteds2.plot(kind='kde')