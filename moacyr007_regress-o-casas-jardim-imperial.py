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

import seaborn as sns
df = pd.read_csv("../input/casas-jardim-imperial-cuiaba/limpo - Pgina1.csv")

df.head()
dfPrecoTamanho = df[['preco','tamanhom2']]

dfPrecoTamanho.head()
sns.pairplot(data=dfPrecoTamanho, kind="reg")
from sklearn import linear_model

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()
X = np.array(dfPrecoTamanho['tamanhom2']).reshape(-1, 1)

y = le.fit_transform(dfPrecoTamanho['preco'])

regressao.fit(X, y)
tamanho = 1250

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))
tamanhos = [900,1100,1150,2100,2510,3100]

for i in tamanhos:

    j = regressao.predict(np.array(i).reshape(-1, 1))

    print('Tamanho: ',i,' Valor: ',j,'\n')

    
dfPrecoTamanhoQuartos = df[['preco','banheiros','tamanhom2']]

dfPrecoTamanhoQuartos.head()
X = np.array(dfPrecoTamanhoQuartos[['tamanhom2','banheiros']])

y = le.fit_transform(dfPrecoTamanhoQuartos['preco'])

regressao.fit(X, y)
quartos = 2

tamanho = 1150

print('Tamanho: ',tamanho,'Quartos: ',quartos,' Valor: ',regressao.predict(np.array([[tamanho,quartos]])))

    