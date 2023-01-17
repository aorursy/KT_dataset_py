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
import numpy as np
adult = pd.read_csv('../input/data-tarefa1/train_data.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult
adult.columns = ['id',
        "age", "workclass", "fnlwgt", "education", "education.num", "martial status",
        "occupation", "relationship", "race", "sex", "capital gain", "capital loss",
        "hours per week", "country", "target"]
adult.head()
adult["education.num"].value_counts().plot(kind="bar")
adult = adult.dropna()
Yadult = adult.target
Xadult = adult[["age","education.num","capital gain", "capital loss", "hours per week"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='brute', leaf_size=20, metric='minkowski',
           metric_params=None, n_jobs=4, n_neighbors=35, p=2,
           weights='distance')
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=7)
scores
knn.fit(Xadult,Yadult)
Ypred = knn.predict(Xadult)
from sklearn.metrics import accuracy_score
accuracy_score(Yadult,Ypred)
teste = pd.read_csv('../input/data-tarefa1/test_data.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
teste.columns = ['id',
        "age", "workclass", "fnlwgt", "education", "education.num", "martial status",
        "occupation", "relationship", "race", "sex", "capital gain", "capital loss",
        "hours per week", "country"]
Xteste = teste[["age","education.num","capital gain", "capital loss", "hours per week"]]
Ytestepred = knn.predict(Xteste)
