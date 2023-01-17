'''Import basic modules'''

import pandas as pd

import numpy as np

import shap

import string

from catboost import CatBoostClassifier, Pool, cv, CatBoostRegressor

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, ShuffleSplit

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression

import os



'''import visualization'''

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.style as style

style.use('fivethirtyeight')



'''Ignore deprecation and future, and user warnings.'''

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
X = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

print(X.shape, test.shape)
X.info() # Checking the datatypes of the Columns available
print(X['target'].unique())

gp = sns.countplot(X['target'])
sns.heatmap(X.corr(), linewidths = 0.1)
train_test = pd.concat([X, test])

train_test.head()
# Remove and Fill in certain features in the dataset

train_test.fillna(-1, axis = 1, inplace = True)

train_test.drop(['id'], axis = 1, inplace = True)
X = train_test.iloc[:X.shape[0], :]

test = train_test.iloc[X.shape[0]:, :]

test.drop(['target'], axis = 1, inplace = True)
X.shape # Checking the test set size
# Getting the Categorical features

cat_features = []

for i in X.columns:

    if X[i].dtype == 'object':

        cat_features.append(i)

        

print(len(cat_features), ' -- Number of Categorical Features')
X.columns # Names of all the columnar data
# Prepare for CatBoost

target = X['target']

train = X.drop(['target'], axis = 1)
def print_cv_summary(cv_data):

    cv_data.head(10)



    best_value = cv_data['test-Logloss-mean'].min()

    best_iter = cv_data['test-Logloss-mean'].values.argmin()



    print('Best validation Logloss score : {:.4f}±{:.4f} on step {}'.format(

        best_value,

        cv_data['test-Logloss-std'][best_iter],

        best_iter))
# Model

cv_dataset = Pool(data=train, label=target, cat_features=cat_features)

params = {"iterations": 100, "depth": 2, "loss_function": "Logloss", "verbose": False, 'custom_loss': 'AUC', 'learning_rate': 0.01}

scores = cv(cv_dataset, params, fold_count = 2, plot = "True")



print_cv_summary(scores)
# Initialize CatBoostClassifier

model = CatBoostRegressor(iterations = 500, learning_rate = 0.09, depth = 2)

model.fit(train, target, cat_features)
shap_values = model.get_feature_importance(Pool(train, label = target, cat_features = cat_features), type="ShapValues")

shap_values = shap_values[:,:-1]

shap.summary_plot(shap_values, train)
predict = model.predict(test)

print(pd.Series(predict).value_counts())
test_id = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', usecols=['id'])

submission = pd.DataFrame({'id': test_id['id'], 'target': predict})

submission.to_csv("submission.csv", index = False)

submission.head()