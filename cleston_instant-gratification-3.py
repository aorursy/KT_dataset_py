# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/instant-gratification/train.csv')
df0 = df[df['target'] == 0].drop(['id', 'target'], axis=1).reset_index(drop=True)

df1 = df[df['target'] == 1].drop(['id', 'target'], axis=1).reset_index(drop=True)
temp = df.drop(['id'], axis=1).reset_index(drop=True)

dfcor = temp.corr(method ='pearson')

dfcor
corlist = dfcor.apply( lambda x: (x.index[abs(x) > 0.008]).tolist() , axis=0)
principais = corlist['target']
treino, teste  = train_test_split(df[principais], test_size=0.3)
print(teste.shape, treino.shape)
clf = tree.DecisionTreeClassifier()

clf.fit(treino.drop(['target'], axis=1), treino['target'])
res = clf.predict(teste.drop(['target'], axis=1))
(res == teste['target']).sum()/teste.shape[0]
treino, teste  = train_test_split(df.drop(['id'], axis=1), test_size=0.3)
clf = tree.DecisionTreeClassifier()

clf.fit(treino.drop(['target'], axis=1), treino['target'])
res = clf.predict(teste.drop(['target'], axis=1))
(res == teste['target']).sum()/teste.shape[0]