# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data = pd.read_csv('../input/jogos_2016_2019.csv')



# Any results you write to the current directory are saved as output.
data.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'ID_PARTIDA', bins=20)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'CASA_ID', bins=20)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'PLACAR_CASA', bins=20)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'PLACAR_FORA', bins=20)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'PLACAR_CASA', bins=20)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'PLACAR_FORA', bins=20)
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='VENCEDOR')

g.map(plt.hist, 'FORA_ID', bins=20)
data = data.drop(['ID',

           'TIME_CASA',

           'TIME_FORA',

           'STATUS',],axis=1,)

data.head()
list(data.columns.values)
data=data.dropna(subset=['PLACAR_CASA'])
data.isna().sum()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["VENCEDOR"] = labelencoder.fit_transform(data["VENCEDOR"].fillna('0'))

data.head()
classe = data['VENCEDOR']

atributos = data.drop('VENCEDOR', axis=1)

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