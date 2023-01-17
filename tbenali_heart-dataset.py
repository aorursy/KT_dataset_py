import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV



import os

print(os.listdir("../input"))



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)

data = pd.read_csv('../input/heart.csv')

columns = data.columns

print(columns)



# Count Nan values in each column 

nan_values = pd.isna(data).sum()

if nan_values.sum() == 0:

    print("Aucune valeur manquante n'a été détectée.")    

data.hist()

data

    

correlation = data.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlation,cmap='hot', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(data.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data.columns)

ax.set_yticklabels(data.columns)

plt.show()

X = data.drop(columns = ['target'])

y = data[['target']].astype("category")

X = StandardScaler().fit_transform(X)





X = data[['cp','slope','restecg','thalach','exang','oldpeak','ca']]

X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)

logistic_regressor = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)

y_pred = logistic_regressor.predict(X_test)

matrice_confusion = confusion_matrix(y_test, y_pred)

taux = sum(np.diag(matrice_confusion))/sum(sum(matrice_confusion))

print(matrice_confusion)

print(taux)

print('\n')
mlp = MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(5, ), max_iter = 100)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
matrice_confusion = confusion_matrix(y_test, y_pred)

taux = sum(np.diag(matrice_confusion))/sum(sum(matrice_confusion))

print(matrice_confusion)

print(taux)
parameter_space = {

    'hidden_layer_sizes': [(5,), (10,), (7,)],

    'activation': ['tanh', 'relu'],

    'alpha': [0.0001, 0.001, 0.01],

}



clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

clf.fit(X_train, y_train)



print('Best parameters found:\n', clf.best_params_)



means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
mlp = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(10, ), activation = 'relu')

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

matrice_confusion = confusion_matrix(y_test, y_pred)

taux = sum(np.diag(matrice_confusion))/sum(sum(matrice_confusion))

print(matrice_confusion)

print(taux)