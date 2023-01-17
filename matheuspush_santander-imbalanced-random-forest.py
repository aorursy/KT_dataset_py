import numpy as np, pandas as pd

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from imblearn.ensemble import *

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import os

SEED = 13

np.random.seed(SEED)
# Load data

data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



y_data = data['target']

x_data = data.drop(columns=['ID_code', 'target'])



y_data = y_data.astype('int8')

x_data = x_data.astype('float16')



print("data loaded")
%%time



clf = BalancedRandomForestClassifier(n_estimators=500, 

                                     criterion='entropy', 

                                     n_jobs=-1)



print('fiting...')

clf.fit(x_data, y_data)



y_pred = clf.predict(x_data)

print('Score:', clf.score(x_data, y_data))

print('B Score:', metrics.balanced_accuracy_score(y_data, y_pred))

print('AUC Score:', metrics.roc_auc_score(y_data, y_pred))
# Condusion Matrix

cnf_matrix = metrics.confusion_matrix(y_data, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
# Save outputc file

id_code = test.pop('ID_code')

test = test.astype('float16')

targets = clf.predict(test)

output = pd.DataFrame({'ID_code': id_code.values, 'target': targets})

output.to_csv('output.csv', index=False)