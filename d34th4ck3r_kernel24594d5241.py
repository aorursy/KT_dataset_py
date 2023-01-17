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
data['age'].unique()
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
Y_train[Y_train==1]
# Categorical index

categorical_index = [1,2,3,4,5,6,7,8,9,13]

print('Categorical parametres: ' + str(X_train.columns[categorical_index].values))
import lightgbm as lgb

#Create Training Datasset

lgb_train = lgb.Dataset(data=X_train, label=Y_train,  free_raw_data=False, categorical_feature=categorical_index)

#Creat Evaluation Dataset 

lgb_eval = lgb.Dataset(data=X_test, label=Y_test, reference=lgb_train,  free_raw_data=False, categorical_feature=categorical_index)

# Determinate training parametres

params = {

    'task': 'train',

    'boosting_type': 'goss',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'verbose': -1

}

evals_result={}

gbm = lgb.train(

    params,

    lgb_train,

    valid_sets = lgb_eval,

    num_boost_round= 150,

    early_stopping_rounds= 25,

    evals_result=evals_result

)
Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)



from sklearn.metrics import roc_auc_score, accuracy_score



print('The Best iteration: ', gbm.best_iteration)

print('roc_auc_score:', roc_auc_score(Y_test, Y_pred))

print('accuracy_score:', accuracy_score(Y_test, ( Y_pred>= 0.5)*1))
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
#How did area under curve change as training progressed

ax = lgb.plot_metric(evals_result, metric='auc')

ax.set_title('Variation of the Curved Area According to Iteration')

ax.set_xlabel('İteration')

ax.set_ylabel('roc_auc_score')

ax.legend_.remove()
#Plotting importance of variables

ax = lgb.plot_importance(gbm, max_num_features=10)

ax.set_title('The values of Parametres')

ax.set_xlabel('Values')

ax.set_ylabel('Parametres')
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

threshold = 0.5

tn, fp, fn, tp = confusion_matrix(Y_test, (Y_pred>=threshold)*1).ravel()

(tn, fp, fn, tp)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

gkf = KFold(n_splits=5, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



param_grid = {

    'num_leaves': [30, 60],

    'max_depth': [-1, 30],

    'n_estimators': [100],

    'learning_rate': [0.1],

#   'reg_alpha': [0.1, 0.5],

#   'min_data_in_leaf': [30, 50, 100, 300, 400]

#   'lambda_l1': [0, 1, 1.5],

#   'lambda_l2': [0, 1]

    }



lgb_estimator = lgb.LGBMClassifier(

    boosting_type='goss',

    objective='binary', 

    num_boost_round=2000, 

    learning_rate=0.01, 

    metric='auc'

)



gsearch = GridSearchCV(

    estimator=lgb_estimator, 

    param_grid=param_grid, 

    cv=gkf

)

lgb_model = gsearch.fit(X=X_train, y=Y_train)
print(lgb_model.best_params_, lgb_model.best_score_)
new_params = lgb_model.best_params_

new_params['task'] = 'train'

new_params['boosting_type'] = 'goss'

new_params['objective'] = 'binary'

new_params['metric']='auc'

new_params

gbm_new = lgb.train(

    lgb_model.best_params_,

    lgb_train,

    valid_sets = lgb_eval,

    num_boost_round= 150,

    early_stopping_rounds= 25,

    evals_result=evals_result

)
new_Y_pred = gbm_new.predict(X_test, num_iteration=gbm.best_iteration)



from sklearn.metrics import roc_auc_score, accuracy_score



print('The Best iteration: ', gbm.best_iteration)

print('roc_auc_score:', roc_auc_score(Y_test, new_Y_pred))

print('accuracy_score:', accuracy_score(Y_test, ( new_Y_pred>= 0.5)*1))
import pandas as pd

import statsmodels.api as sm

import pylab as pl

import numpy as np

from sklearn.metrics import roc_curve, auc



fpr, tpr, thresholds = roc_curve(Y_test, new_Y_pred)

roc_auc = auc(fpr, tpr)

print("Area under the ROC curve : %f" % roc_auc)
####################################

# The optimal cut off would be where tpr is high and fpr is low

# tpr - (1-fpr) is zero or near to zero is the optimal cut off point

####################################

i = np.arange(len(tpr)) # index for df

roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})

print(roc.iloc[(roc.tf-0).abs().argsort()[:1]])



# Plot tpr vs 1-fpr

fig, ax = pl.subplots()

pl.plot(roc['tpr'])

pl.plot(roc['1-fpr'], color = 'red')

pl.xlabel('1-False Positive Rate')

pl.ylabel('True Positive Rate')

pl.title('Receiver operating characteristic')

ax.set_xticklabels([])
lgb.create_tree_digraph(gbm_new)