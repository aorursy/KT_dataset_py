

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chardet

import unicodedata

import pandas





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import seaborn as sns
df = pandas.read_csv("../input/compras/TopCompra.csv", encoding='latin1', quotechar='"', delimiter=',') 

df.head()
dfPrecoTamanho = df[['Cidade', 'Bairro','Estado', 'Tipo', 'Valor', 'Area', 'Quartos', 'Banheiros', 'Garagem']]

dfPrecoTamanho.head()
sns.pairplot(data=dfPrecoTamanho, kind="reg")
from sklearn import linear_model

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()


X = np.array(dfPrecoTamanho['Quartos']).reshape(-1, 1)

Y = le.fit_transform(dfPrecoTamanho['Valor'])

regressao.fit(X, Y)
tamanho = 1250

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))
tamanhos = [900,1100,1150,2100,2510,3100]

for i in tamanhos:

    j = regressao.predict(np.array(i).reshape(-1, 1))

    print('Tamanho: ',i,' Valor: ',j,'\n')

    
dfPrecoTamanhoQuartos = df[['price','bedrooms','sqft_living']]

dfPrecoTamanhoQuartos.head()
X = np.array(dfPrecoTamanhoQuartos[['sqft_living','bedrooms']])

Y = le.fit_transform(dfPrecoTamanhoQuartos['price'])

regressao.fit(X, Y)
quartos = 3

tamanho = 1150

print(regressao.coef_)

print('Tamanho: ',tamanho,'Quartos: ',quartos,' Valor: ',regressao.predict(np.array([[tamanho,quartos]])))

    