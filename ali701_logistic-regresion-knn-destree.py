# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

test.head()
train.shape , test.shape
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

%matplotlib inline

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split,  cross_val_score
train['target'].hist();
y = train['target']

X = train.drop(['target','ID_code'],axis=1)

X_test = test.drop('ID_code', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=200)
from scipy.sparse import csr_matrix
X_train.shape, X_valid.shape
logit = LogisticRegression(n_jobs=-1, random_state=17)

logit.fit(X_train, y_train)
logit.predict_proba(X_test)[0,:]
logit.predict_proba(X_test)[:10,:]
logit.predict_proba(X_test)[:10, :] [: ,1]
def get_auc_lr_valid(X, y, C=1.0, ratio = 0.9, seed=17):

 

    logit = LogisticRegression( C=C, n_jobs=-1, random_state=seed)

    logit.fit(X_train, y_train)        

    valid_pred = logit.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y_valid, valid_pred)
get_auc_lr_valid(X_train, y_train)
test_pred = logit.predict_proba(X_test)[:, 1]
test_pred.shape
sub_df = pd.DataFrame({'ID_code':test.ID_code.values})

sub_df['target'] = test_pred

sub_df.to_csv('submission.csv', index=False)
sub_df.head(90)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
tree = DecisionTreeClassifier(random_state=17)
from sklearn.model_selection import GridSearchCV
first_tree = DecisionTreeClassifier(random_state=200)
tree_params = {'max_depth': np.arange(1,11), 'max_features' : range(4, 1)}
tree_params = {'max_depth': range(1,11),'max_features': range(1,10)}
tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)
tree_grid.best_score_, tree_grid.best_params_
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
knn_params = {'knn__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_params,

cv=5, n_jobs=-1,

verbose=True)
from sklearn.tree import export_graphviz
export_graphviz(tree_grid.best_estimator_, feature_names=X.columns, 

out_file='des_tree.dot', filled=True,)
!ls -l *.dot
!dot -Tpng des_tree.dot -o des_tree.png