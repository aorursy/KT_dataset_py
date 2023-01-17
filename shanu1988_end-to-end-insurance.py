import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# data selecton module

from sklearn.model_selection import train_test_split



# Modelling modules

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

# Explicitly require this experimental feature

from sklearn.experimental import enable_hist_gradient_boosting

# Import normally hist from ensemble

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import VotingClassifier



# Model Evaluations

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve
train = pd.read_csv("../input/porto-seguro-safe-driver-prediction/train.csv")

test = pd.read_csv("../input/porto-seguro-safe-driver-prediction/test.csv")
# check the shape of datasets

print("The shape of Train data", train.shape)

print("The shape of Test data", test.shape)
# The head of data

train.head()
#Find out how many of each class there

train.target.value_counts()
train["target"].value_counts().plot(kind='bar', color=["salmon","lightblue"]);
train.info()
# let's check Null values if are there any

train.isnull().mean()
#ps_reg_03 vs target

plt.figure(figsize=(19, 6))



# Scatter with positive examples

plt.scatter(train.id[train.target==1],

           train.ps_reg_03[train.target==1],

           c='salmon')



# scatter with positive examples

plt.scatter(train.id[train.target==0],

           train.ps_reg_03[train.target==0],

           c="lightblue")



# let's add some helpfull information

plt.title("Car Insurance in function of ID and ps_reg_03")

plt.xlabel("ID")

plt.ylabel("ps_reg_03")

plt.legend(["Claim", "No Claim"]);
#Split train data into X & y



X = train.drop("target", axis=1)

y = train["target"]





# Split train data into X_train & validation sets so that we check our model results before prediction on test set

X_train, X_valid, y_train, y_valid = train_test_split(X,

                                                      y,

                                                      test_size=0.1)

                                                     

# Create X_test from test data this will use while prediction from trained model

X_test = test.copy()



# Check the shape of split datasets

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
## create function for Gini cofficent



def ginic(actual, pred):

    actual = np.asarray(actual) 

    n = len(actual)

    a_s = actual[np.argsort(pred)]

    a_c = a_s.cumsum()

    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0

    return giniSum / n

 

def gini_normalized(a, p):

    if p.ndim == 2:

        p = p[:,1] 

    return ginic(a, p) / ginic(a, a)

    



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score

# XGB modeling

# import xgboost as xgb

# params = {'eta': 0.025,

#           'max_depth':4,

#           'subsamples':0.9,

#           'colsample_bytree': 0.7,

#           'colsample_bylevel':0.7,

#           'min_chiled_weight':100,

#           'alpha':4,

#           'objective':'binary:logistic',

#           'eval_metric':'auc',

#           'seed':99,

#           'silent':True}



import xgboost as xgb

params = {'eta': 0.02543,

          'max_depth':9,

          'subsamples':0.5,

          'colsample_bytree': 0.9,

          'colsample_bylevel':0.3,

          'min_chiled_weight':300,

          'alpha':5,

          'objective':'binary:logistic',

          'eval_metric':'auc',

          'seed':99,

          'silent':True}
watchlist = [(xgb.DMatrix(X_train, y_train),'train'),(xgb.DMatrix(X_valid, y_valid),'valid')]

model = xgb.train(params, xgb.DMatrix(X_train, y_train), 5000, watchlist, feval=gini_xgb, maximize=True,

                 verbose_eval=100, early_stopping_rounds=70)
sub = pd.DataFrame()

sub['id'] = X_test.id

sub['target'] = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

sub.to_csv('submi.csv', index=False)

sub[:10]