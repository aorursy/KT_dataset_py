# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

data = pd.read_csv("/kaggle/input/bank-marketing/bank-additional-full.csv", delimiter=";")
data.head()
data_pos = data[data["y"]=='yes']

data_neg = data[data["y"]=='no']



from sklearn.utils import shuffle

balanced_data = shuffle(pd.concat([data_pos, data_neg.sample(len(data_pos))]))

small_balanced_data = shuffle(pd.concat([data_pos.sample(500), data_neg.sample(500)]))



# data = balanced_data

data = small_balanced_data
Y = (data["y"]=="yes")*1
data.info()
data.drop('y', axis=1, inplace = True)
from sklearn.preprocessing import LabelEncoder

categorical_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',

                      'day_of_week', 'poutcome']

for i in categorical_column:

    le = LabelEncoder()

    data[i] = le.fit_transform(data[i])

print(data.head())
# Dropping duration of call because it creates a heavy bias as pointed in original dataset.

data.drop('duration', inplace = True, axis=1)
data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, Y, train_size = 0.7, test_size = 0.3)
import xgboost as xgb

evals_result = dict()

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train, eval_set=evals_result)

Y_pred_proba = gbm.predict_proba(X_test)

Y_pred = gbm.predict(X_test)
from sklearn.metrics import roc_auc_score, accuracy_score



print('roc_auc_score:', roc_auc_score(Y_test, Y_pred_proba[:,1]))

print('accuracy_score:', accuracy_score(Y_test, Y_pred))
#Getting the ROC curve

from sklearn import metrics

import matplotlib.pyplot as plt

fpr, tpr, _ = metrics.roc_curve(Y_test,  Y_pred)

auc = metrics.roc_auc_score(Y_test, Y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic graph')

plt.show()
from xgboost import plot_importance

plot_importance(gbm)
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

threshold = 0.5

tn, fp, fn, tp = confusion_matrix(Y_test, (Y_pred>=threshold)*1).ravel()

(tn, fp, fn, tp)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from xgboost.sklearn import XGBClassifier



gkf = KFold(n_splits=5, shuffle=True, random_state=42).split(X=X_train, y=Y_train)

#max_depth=3, n_estimators=300, learning_rate=0.05

# A parameter grid for XGBoost

params = {

    'num_leaves': [15, 30], #number of leaves in the tree

    'max_depth': [3, 6], #The maximum depth of a tree

    'n_estimators': [300, 600], #the number of trees to be used in the forest

    'learning_rate': [0.01, 0.05], 

    'min_child_weight': [1], #the minimum sum of weights of all observations required in a child

    'gamma': [0.5], #A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.

    'subsample': [0.6], # the fraction of observations to be randomly samples for each tree.

    'colsample_bytree': [0.6], #the fraction of columns to be randomly samples for each tree.

}



my_estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1, seed=27)



gsearch1 = GridSearchCV(estimator = my_estimator, param_grid = params, scoring='roc_auc',n_jobs=4, cv=5)

gsearch1.fit(X_train,Y_train)

gsearch1.best_params_
new_params = gsearch1.best_params_

gbm_new = xgb.XGBClassifier(

    max_depth = gsearch1.best_params_['max_depth'], 

    n_estimators = gsearch1.best_params_['n_estimators'], 

    learning_rate = gsearch1.best_params_['learning_rate'],

    num_leaves = gsearch1.best_params_['num_leaves']

).fit(X_train, Y_train)

Y_pred_proba_new = gbm_new.predict_proba(X_test)

Y_pred_new = gbm_new.predict(X_test)
print('roc_auc_score:', roc_auc_score(Y_test, Y_pred_proba_new[:,1]))

print('accuracy_score:', accuracy_score(Y_test, Y_pred_new))
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=10)

# Train

model.fit(X_train, Y_train)

# Extract single tree

estimator = model.estimators_[5]
df_Y_train = pd.DataFrame(Y_train)

from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = X_train.columns,

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')