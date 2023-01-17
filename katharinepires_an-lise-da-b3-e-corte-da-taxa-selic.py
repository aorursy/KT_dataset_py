import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        b3 = os.path.join(dirname, filename)
b3 = pd.read_csv(b3)

b3.loc[:, "datetime"] = b3.datetime.map(pd.Timestamp)

selic = pd.read_csv('../input/ibovespa-stocks/selic.csv')

selic.loc[:, "datetime"] = selic.datetime.map(pd.Timestamp)
b3
selic
import seaborn as sns

import matplotlib.pyplot as plt
date = selic.iloc[3248: , :]

date
plt.figure(figsize=(30,10))

plt.plot(date['datetime'], date['selic'])

plt.xlabel('Data')

plt.ylabel('Taxa Selic')

plt.show()
data = b3.iloc[1753290: , :]

data
bradesco = data[data['ticker'] == 'BBDC3']

itau = data[data['ticker'] == 'ITUB3']

santander = data[data['ticker'] == 'SANB3']

inter = data[data['ticker'] == 'BIDI3']

bb = data[data['ticker'] == 'BBAS3']

bco_nordeste = data[data['ticker'] == 'BNBR3']
plt.figure(figsize=(30,10))

plt.plot(bradesco['datetime'], bradesco['close'])

plt.xlabel('Mês 2020')

plt.ylabel('Cotação')

plt.title('Cotação do Banco Bradesco (BBDC3)')

plt.show()
date = selic.iloc[6400: , :]

plt.figure(figsize=(30,10))

plt.plot(date['datetime'], date['selic'])

plt.xlabel('Mês 2020')

plt.ylabel('SELIC')

plt.title('Variação da Taxa SELIC')

plt.show()
plt.figure(figsize=(30,10))

plt.plot(bb['datetime'], bb['close'])

plt.xlabel('Mês 2020')

plt.ylabel('Cotação')

plt.title('Cotação do Banco do Brasil (BBAS3)')

plt.show()
plt.figure(figsize=(30,10))

plt.plot(bco_nordeste['datetime'], bco_nordeste['close'])

plt.xlabel('Mês 2020')

plt.ylabel('Cotação')

plt.title('Cotação do Banco do Nordeste (BNBR3)')

plt.show()
plt.figure(figsize=(30,10))

plt.plot(itau['datetime'], itau['close'])

plt.xlabel('Mês 2020')

plt.ylabel('Cotação')

plt.title('Cotação do Itaú Unibanco (ITUB3)')

plt.show()
plt.figure(figsize=(30,10))

plt.plot(inter['datetime'], inter['close'])

plt.xlabel('Mês 2020')

plt.ylabel('Cotação')

plt.title('Cotação do Banco Inter (BIDI3)')

plt.show()