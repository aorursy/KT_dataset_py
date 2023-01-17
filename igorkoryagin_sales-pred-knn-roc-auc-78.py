# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.tabular.transform import add_cyclic_datepart
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Metrics for models evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import pickle # for saving and loading processed datasets and hyperparameters
import gc

from sklearn.neighbors import KNeighborsClassifier
import optuna # for hyperparameter tuning

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/recsys-challenge-2015'
def load_saved_dataset(filename):
    try:
        with open('../input/recsys-preprocessed/{}.pickle'.format(filename), 'rb') as fin:
            X = pickle.load(fin)
        print('Dataset loaded')
    except FileNotFoundError:
        print('File with saved dataset not found')
    return X

def load_saved_parameters(filename):
    try:
        with open('../input/recsys-parameters/{}.pickle'.format(filename), 'rb') as fin:
            X = pickle.load(fin)
        print('Parameters loaded')
    except FileNotFoundError:
        print('File with saved parameters not found')
    return X
filename = 'Processed_recsys'
param_file = 'KNN_4_79auc'
df = load_saved_dataset(filename)
knn_model = load_saved_parameters(param_file)
df = df[:3000000]
y = df["buy"]
results = {}
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=1, test_size=0.2, shuffle=False)
del df

gc.collect()
print('Training KNN ')
#Fit the model
knn_model.fit(X_train, y_train)

#Compute accuracy on the training set
train_accuracy = knn_model.score(X_train, y_train)

#Compute accuracy on the test set
test_accuracy = knn_model.score(X_test, y_test)

y_pred_proba = knn_model.predict_proba(X_test)[:,1]
y_pred = np.where(y_pred_proba > 0,1,0)

accuracy = accuracy_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)

print('The accuracy of prediction is:', accuracy)
print('The ROC AUC of prediction is:', roc_auc)
print('The F1 Score of prediction is:', f1)
print('The Precision of prediction is:', prec)
print('The Recall of prediction is:', rec)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=15) ROC curve')
plt.show()