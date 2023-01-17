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
data = pd.read_csv('../input/Minha segurana.csv')

data.columns = ['DATA',

                   'IDADE',

                   'CONTAS_ATIVAS',

                   'TROCA_SENHA',

                   'EMAILS',

                   'HABITO_TROCAR']

data.dropna(axis = 0, how = 'any', inplace = True)

data.index = pd.RangeIndex(len(data.index))
data.head()

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["IDADE"] = labelencoder.fit_transform(data["IDADE"].fillna('0'))

data["CONTAS_ATIVAS"] = labelencoder.fit_transform(data["CONTAS_ATIVAS"].fillna('0'))

data["TROCA_SENHA"] = labelencoder.fit_transform(data["TROCA_SENHA"].fillna('0'))

data["EMAILS"] = labelencoder.fit_transform(data["EMAILS"].fillna('0'))

data["HABITO_TROCAR"] = labelencoder.fit_transform(data["HABITO_TROCAR"].fillna('0'))

data.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='HABITO_TROCAR')

g.map(plt.hist, 'IDADE', bins=5)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='HABITO_TROCAR')

g.map(plt.hist, 'CONTAS_ATIVAS', bins=5)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='HABITO_TROCAR')

g.map(plt.hist, 'TROCA_SENHA', bins=5)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='HABITO_TROCAR')

g.map(plt.hist, 'EMAILS', bins=5)
data = data.drop(['DATA'],axis=1,)

data.head()
list(data.columns.values)
data.isna().sum()
classe = data['HABITO_TROCAR']

atributos = data.drop('HABITO_TROCAR', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))