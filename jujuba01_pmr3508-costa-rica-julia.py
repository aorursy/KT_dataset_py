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
import pandas as pd
import sklearn
pop = pd.read_csv("../input/juliadados/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="NaN")
pop.shape

pop
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
npop = pop.dropna(axis=1)
npop.head()
pop.v2a1 = pop.v2a1.interpolate()
ipop = pop.dropna(axis=1)
ipop.shape
xpop = ipop.iloc[0:9557,0:138]
ypop = ipop.Target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5).fit(xpop,ypop)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, xpop, ypop, cv=10)
scores

import matplotlib.pyplot as plt
pop["Target"].value_counts().plot(kind="bar")

teste = pd.read_csv("../input/juliadados/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="NaN")
teste.head()
teste.v2a1 = teste.v2a1.interpolate()
teste = teste.dropna(axis=1)
teste.shape
yteste = knn.predict(teste)
