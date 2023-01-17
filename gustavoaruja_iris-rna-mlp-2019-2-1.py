# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importando o dataset

from sklearn import datasets as dt

iris = dt.load_iris()

iris.keys()

x= iris.data

y= iris.target
df = pd.DataFrame(x,columns=iris.feature_names)
df['target'] = y

df.head() # data frame é uma matriz do panda
#importando gráfico

import matplotlib.pyplot as plt

cores = df['target']

cores = cores.astype('category')

cores = cores.cat.codes

plt.figure(figsize=(6,3))

plt.scatter(df['sepal length (cm)'],

            df['petal width (cm)'],

            c = cores, marker='*')

plt.title('dispersão')

plt.xlabel('sepal length (cm)')

plt.ylabel('petal width(cm)')

plt.show()            
#pré processamento

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x = ss.fit_transform(x)
#treino de teste

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(x_train, y_train)

y_predict = modelo.predict(x_test)
#score

score = modelo.score(x,y)

score