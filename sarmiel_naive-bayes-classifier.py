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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dados = pd.DataFrame(pd.read_csv("../input/train.csv"))

dados['target'].value_counts().plot(kind='bar')
xtreino = dados.iloc[:, 2:].values.astype('float64')

ytreino = dados.iloc[:, 1].values



dados_teste = pd.read_csv('../input/test.csv')

label_teste = pd.read_csv('../input/sample_submission.csv')



xteste = dados_teste.iloc[:, 1:].values.astype('float64')

yteste = label_teste.iloc[:, 1].values

from sklearn.naive_bayes import GaussianNB

NB_clf = GaussianNB()

NB_clf.fit(xtreino, ytreino)

predicao = NB_clf.predict(xteste)
from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(yteste, predicao))

accuracy_score(yteste, predicao)
ret = confusion_matrix(yteste, predicao)

plt.bar(x=['0 True','0 False', '1 True', '1 False'], height=[ret[0][0], ret[0][1], ret[1][0], ret[1][1]])