# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
teste = pd.read_csv("../input/desafio-worcap-2020/teste.csv")
treino = pd.read_csv("../input/desafio-worcap-2020/treino.csv")

df_teste = pd.DataFrame(data=teste)
df_treino = pd.DataFrame(data=treino)


#df_teste.head()
df_treino.head()
# df_treino_pred = df_treino.iloc[:,9:27]
# df_treino_pred.describe() 
df_treino.head()
pontos_attr = df_treino.iloc[:,[2,3,4,5,8,9]]
pontos_attr.head()

pontos_label = df_treino['label']
pontos_label.head()


# Hinoki_correlation = df_treino.corr()['pred_minus_obs_H_b3']
# columns = list()

# for i, corr in enumerate(Hinoki_correlation):
#     if corr >= 0.5 or corr < -0.5:
#         columns.append(df_treino.columns[i])
#         print(df_treino.columns[i], '=', corr)
        
# new_df_treino = df_treino[columns]
# new_df_treino.head()

knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(pontos_attr, pontos_label)
pontos_teste = df_teste.iloc[:,[2,3,4,5,8,9]]
pontos_teste_id = df_teste.iloc[:,0]
pontos_teste_id.head()
ids = pontos_teste_id.to_xarray()

#pontos_teste.head()

predicoes = knn.predict(pontos_teste)
#labels = predicoes.to_xarray()

# predicoes_df = pd.DataFrame(data = predicoes,columns=['label'])
d = {'id':ids,'label':predicoes}
predicoes_df = pd.DataFrame(data = d)

predicoes_df.head()
predicoes_df.to_csv("./submission_Arthur04.csv",index=False)

