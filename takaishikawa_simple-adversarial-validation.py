import numpy as np; np.random.random(42)

import pandas as pd

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

import warnings; warnings.filterwarnings("ignore")



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
plt.rcParams['figure.figsize'] = [10, 5]

plt.rcParams['font.size'] = 12
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
train.head(3)
print( train.target.value_counts() / train.shape[0] * 100 )
feature_names = train.columns[2:]
train["train_test"] = 1

test["train_test"] = 0
data = pd.concat(( train, test ))



np.random.seed(42)

data = data.iloc[ np.random.permutation(len( data )) ]

data.reset_index( drop = True, inplace = True )



x = data.drop( [ 'target', 'ID_code','train_test' ], axis = 1 )

y = data.train_test
train_examples = len(train)



x_train = x[:train_examples]

x_test = x[train_examples:]

y_train = y[:train_examples]

y_test = y[train_examples:]
x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = train_examples, random_state=42 )
clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", random_state=42)

clf.fit(x_train, y_train)

y_pred = clf.predict_proba(x_test)[:, 1]

roc_auc_score(y_test, y_pred)
clf = RandomForestClassifier(n_estimators=10, random_state=42)

clf.fit(x_train, y_train)

y_pred = clf.predict_proba(x_test)[:, 1]

print("AUC:",round(roc_auc_score(y_test, y_pred)*100,2),"%")
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(x_train, y_train)

y_pred = clf.predict_proba(x_test)[:, 1]

print("AUC:",round(roc_auc_score(y_test, y_pred)*100,2),"%")