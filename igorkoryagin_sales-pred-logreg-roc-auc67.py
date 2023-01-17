import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from fastai.tabular.transform import add_cyclic_datepart
import matplotlib.pyplot as plt

# Metrics for models evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import pickle # for saving and loading processed datasets and hyperparameters
import gc

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import optuna # for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/recsys-challenge-2015'
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system names
        if name == "PIL":
            name = "Pillow"
        elif name == "sklearn":
            name = "scikit-learn"

        yield name
imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))
import sys
print(sys.version)
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
param_file = 'LogReg_68_params'
df = load_saved_dataset(filename)
parameters = load_saved_parameters(param_file)
y = df["buy"]
df.drop(['buy'], 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=1, test_size=0.2, shuffle=False)
del df

gc.collect()
parameters
log_reg = LogisticRegression(C = parameters['C'], class_weight = parameters['class_weight'], \
                             penalty = parameters['penalty'], solver=parameters['solver'])
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)

print('The accuracy of prediction is:', accuracy)
print('The ROC AUC of prediction is:', roc_auc)
print('The F1 Score of prediction is:', f1)
print('The Precision of prediction is:', prec)
print('The Recall of prediction is:', rec)
y_pred_proba = log_reg.predict_proba(X_test)
y_pred_proba[:,0]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Logistic Regression')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'Logistic Regression ROC curve')
plt.show()