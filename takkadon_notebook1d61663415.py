import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss, roc_curve, confusion_matrix, plot_roc_curve,mean_absolute_error,mean_squared_log_error

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier,LGBMRegressor



import eli5

import seaborn as sns
pd.set_option('display.max_rows', 120)

pd.set_option('display.max_columns', 120)
train = pd.read_csv('../input/exam-for-students20200923/train.csv')

test = pd.read_csv('../input/exam-for-students20200923/test.csv')
train = train.query('ConvertedSalary != 0')
y_train = train['ConvertedSalary']

X_train = train.drop(['ConvertedSalary'], axis=1)



X_test = test
train.sort_values('ConvertedSalary',ascending=False)
len(X_train)
X_train.isnull().sum()
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)
len(cats)
oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.fit_transform(X_test[cats])
X_train
X_train.fillna(X_train.median(), axis=0, inplace=True)

X_test.fillna(X_test.median(), axis=0, inplace=True)
X_train.isnull().sum()
y_train = np.log(y_train + 1)
random = [40,71,80]
scoress = 0
for i in random:

    scores = []

    y_pred_test = np.zeros(len(test))

    skf = KFold(n_splits=5, random_state=i, shuffle=True)



    for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]





        clf = LGBMRegressor(num_leaves = 31,

                            learning_rate=0.1,

                            min_child_samples=10,

                            n_estimators=1000)



        clf.fit(X_train_, y_train_, early_stopping_rounds=10, eval_metric='rmse', eval_set=[(X_val, y_val)])

        y_pred = clf.predict(X_val)

        score = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append(score)

        y_pred_test += np.exp(clf.predict(X_test))-1

        print('CV Score of Fold_%d is %f' % (i, score))

    print(score)

    scores = np.array(scores)

    y_pred_test /= 5

    scoress += y_pred_test
scoress = scoress/3
submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)
submission['ConvertedSalary'] = scoress
y_pred_test = y_pred_test.round()
submission.to_csv('submission.csv')
submission