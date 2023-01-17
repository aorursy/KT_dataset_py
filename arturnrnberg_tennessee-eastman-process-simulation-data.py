%load_ext autoreload

%autoreload 2



%matplotlib inline
!pip install pyreadr==v0.3.3



import pyreadr

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
%%time



# troque aqui pela localização do dataset na sua máquina

PATH = '/kaggle/input/tennessee-eastman-process-simulation-dataset/'



train_normal_path = PATH+'TEP_FaultFree_Training.RData'

train_faulty_path = PATH+'TEP_Faulty_Training.RData'



test_normal_path = PATH+'TEP_FaultFree_Testing.RData'

test_faulty_path = PATH+'TEP_Faulty_Testing.RData'



train_normal_complete = pyreadr.read_r(train_normal_path)['fault_free_training']

#train_faulty_complete = pyreadr.read_r(train_fault_path)['faulty_training']



#test_normal_complete = pyreadr.read_r(test_normal_path)['fault_free_testing']

test_faulty_complete = pyreadr.read_r(test_faulty_path)['faulty_testing']
def draw_graph(fault_number=1, simulation=1):

    

    df_train = train_normal_complete[train_normal_complete.simulationRun==simulation].iloc[:,3:]



    df_test = test_faulty_complete[(test_faulty_complete.simulationRun==simulation)&

                               (test_faulty_complete.faultNumber==fault_number)].iloc[:,3:]

    

    fig, ax = plt.subplots(13, 4, figsize=(30, 70))

    

    for i in range(df_train.shape[1]):

        

        x = df_train.iloc[:, i]

        x_test = df_test.iloc[:, i]

        

        mean = x.mean()

        std = x.std(ddof=1)

        

        limite_superior = mean + 3*std

        limite_inferior = mean - 3*std

        

        x_test.plot(ax=ax.ravel()[i])

        

        ax.ravel()[i].legend()

        

        ax.ravel()[i].axhline(mean, c='k')

        ax.ravel()[i].axhline(limite_superior, ls='--', c='r')

        ax.ravel()[i].axhline(limite_inferior, ls='--', c='r')

        

        ax.ravel()[i].axvline(df_test.index[0] + 160, c='g')
draw_graph(fault_number=3, simulation=1)
draw_graph(fault_number=6, simulation=2)
draw_graph(fault_number=14, simulation=1)
from sklearn.datasets import load_iris

from sklearn.decomposition import PCA
data = load_iris().data

target = load_iris().target



print(load_iris().DESCR)
pd.DataFrame(data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
pca = PCA(n_components = 4)

pca.fit(data)



print('componentes:')

print(pca.components_)



print('\nvariâncias explicadas:')

print(pca.explained_variance_)
pca = PCA(n_components = 2)

pca.fit(data)



print('componentes:')

print(pca.components_)



print('\nvariâncias explicadas:')

print(pca.explained_variance_)
# Dois vetores ortogonais em 4 dimensões podem ser representados em 2 dimensões.

T = pca.transform(data)
plt.scatter(T[:, 0], T[:, 1], alpha=0.8)

plt.axis('equal')

plt.xlabel('t1')

plt.ylabel('t2');
plt.scatter(T[:, 0], T[:, 1], c=target, cmap='Dark2_r', alpha=0.7)

plt.axis('equal')

plt.xlabel('t1')

plt.ylabel('t2');