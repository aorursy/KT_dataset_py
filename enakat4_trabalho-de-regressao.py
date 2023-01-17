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

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/trab003.csv") 

df.head() # visualizar a base de dados
df.describe()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,linewidths=0.01 ,cmap='coolwarm', cbar=True)

plt.show() # faz a correlação entre todas as variáveis, quando cada variável muda a relação entre eles varia tb
features =['valor','area construido','quarto']

dfPreco = df[features]

dfPreco.head()
sns.pairplot(data=dfPreco, kind="reg")
X = df[features[1:]]

y = df.valor.values

X.head()
from sklearn.model_selection import train_test_split # faz o split dos dados de teste e treino dando uma misturada

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

print(len(x_train),len(x_test))
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

#scaler = preprocessing.MaxAbsScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train),columns=features[1:]) # fit_transform cria um modelo de normalização

x_test = pd.DataFrame(scaler.transform(x_test),columns=features[1:])

print(x_train[:5],'\n',x_test[:5])
from sklearn.linear_model import LinearRegression

lr=LinearRegression()



lr.fit(x_train,y_train)



print("Valor real de y_test[23]:" + str(y_test[5]) + " -> Valor predito:" + str(lr.predict(x_test.iloc[[5],:])))

print("Valor real de y_test[20]:"+ str(y_test[3]) + " -> Valor predito: " + str(lr.predict(x_test.iloc[[3],:])))



from sklearn.metrics import r2_score

y_head_lr=lr.predict(x_test)

print("r_square score (teste):",r2_score(y_test,y_head_lr))



y_head_lr_train=lr.predict(x_train)

print("r_square score (treino):",r2_score(y_train,y_head_lr_train))
features[1:]
parametros = [[133,3]] # predizendo o valor com relação a área construida e o numero de quartos



parametros = pd.DataFrame(scaler.transform(parametros),columns=features[1:])

print("Valor do imóvel: $ %.2f" % lr.predict(parametros.values))