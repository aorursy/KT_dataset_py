import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

treino = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')
treino = treino.drop(columns=["Id"])
treino.head(20)
treino.info()
treino.describe()
plt.title('Matriz de correlação')
sns.heatmap(treino.corr(), annot=True, linewidths=0.5)
treino_Y = treino['median_house_value']
treino = treino.drop(columns='median_house_value')
LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
LR.fit(treino, treino_Y)
LR.score(treino, treino_Y)
#demora demais

'''
parameters = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
clf_grid = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
clf_grid.fit(treino, treino_Y)
clf_grid.score(treino, treino_Y)
'''

clf = linear_model.Lasso(alpha=2500)
clf.fit(treino, treino_Y)
clf.score(treino, treino_Y)