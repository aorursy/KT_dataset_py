import numpy as np
import pandas as pd
from math import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
def partitions(n, condition_to_insert = None):
    assert isinstance(n, int), 'n must be an integer'
    assert n > 0, 'n must be a natural number but zero.'

    if(condition_to_insert is None):
        condition_to_insert = lambda partition: True

    a = list(range(n+1))
    tuples = []

    for m in range(2, n+1):
        a[m] = 1

    m = 1
    a[0] = 0

    while(True):

        a[m] = n
        q = m - int(n == 1)

        while(True):
            partition = tuple(a[1:m+1])

            if(condition_to_insert(partition)):
                permutations = list(set(list(itertools.permutations(partition))))
                tuples += permutations

            if(a[q] != 2):
                break

            a[q] = 1
            q -= 1
            m += 1

        if(q == 0):
            break

        x = a[q] - 1
        a[q] = x
        n = m - q + 1
        m = q + 1

        while(n > x):

            a[m] = x
            m += 1
            n -= x

    return tuples
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
Y_id = df_test.id
df_train.drop(['id'],axis=1,inplace=True)
df_test.drop(['id'],axis=1,inplace=True)
plt.figure(figsize = (10,7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

corr = df_train.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,vmin=-1, vmax=1,linewidths=.5, cmap = "RdBu_r",annot=True)
plt.title('Heatmap da correlação de Pearson')
plt.show()
input_neurons_amount, output_neurons_amount = len(df_train.columns), 2
def hidden_neurons_amount(alpha):
    return alpha * sqrt(input_neurons_amount * output_neurons_amount)

def condition_to_insert(partition):
    return len(partition) <= 2
alpha = [0.5, 2, 3]
hidden_neurons_amounts = [ceil(hidden_neurons_amount(a)) for a in alpha]
hidden_layer_sizes = []

for n in hidden_neurons_amounts:
    tuples = partitions(n, condition_to_insert)
    hidden_layer_sizes += tuples
params = {
    'activation': ['identity', 'logistic','tanh','relu'],
    'hidden_layer_sizes': hidden_layer_sizes,
    'solver': ['lbfgs']
}
gs = GridSearchCV(MLPClassifier(), params, cv=3, scoring='f1_micro', return_train_score=False)
target = df_train.Classification
df_train.drop(['Classification'],axis=1,inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(df_train, target, test_size=0.2)
gs.fit(X_train, Y_train);
best_model = gs.best_estimator_
Y_pred = best_model.predict(X_test)
f1_score(Y_test,Y_pred,average='micro')
Y_pred = best_model.predict(df_test)
results = pd.DataFrame(data={'Classification' : Y_pred, 'id' : Y_id})
results.to_csv('result.csv', index=False)
pd.read_csv('result.csv')
