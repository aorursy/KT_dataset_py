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
import pandas as pd # load and manipulate data and for One-Hot Encoding

import numpy as np # calculate the mean and standard deviation

import xgboost as xgb # XGBoost stuff

from sklearn.model_selection import train_test_split # split  data into training and testing sets

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring during cross validation

from sklearn.model_selection import GridSearchCV # cross validation

from sklearn.metrics import confusion_matrix # creates a confusion matrix

from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
train = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/train.csv')

test = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/test.csv')

train.shape,test.shape
# look at the first five rows

train.head()
# what kind of data is each column?

train.dtypes
# check if any X_train columns contain any NAs

train.columns[train.isna().any()].tolist()
# check if any X_test columns contain any NAs

test.columns[test.isna().any()].tolist()
X = train.drop('churn', axis=1).copy() 

X.head()
y = train['churn'].copy()

y.head()
X.dtypes
X['state'].unique()
X['area_code'].unique()
X['international_plan'].unique()
X['voice_mail_plan'].unique()
# Create Binary Labels

X['international_plan'] = np.where(X['international_plan'].str.contains('yes'), 1, 0) # Note that 'no' is 1 to keep consistent with the data dictionary

X['voice_mail_plan'] = np.where(X['voice_mail_plan'].str.contains('yes'), 1, 0)



# Do the same for our test set

test['international_plan'] = np.where(test['international_plan'].str.contains('yes'), 1, 0) # Note that 'no' is 1 to keep consistent with the data dictionary

test['voice_mail_plan'] = np.where(test['voice_mail_plan'].str.contains('yes'), 1, 0)



X.head()
# Create Dummy Variables

X = pd.get_dummies(X, columns=['state', 'area_code'])



# Do the same for our test set

test = pd.get_dummies(test, columns=['state', 'area_code'])



X.head()
# check to make sure y only has two values

y.unique()
# change y to numeric 

# y = pd.Series(np.where(y == 'yes', 1, 0),y.index)
# check if we need to stratify

# sum(y)/len(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
# sum(y_train)/len(y_train)
# sum(y_test)/len(y_test)
# ## NOTE: When data are imbalanced, the XGBoost manual says...

# ## If you care only about the overall performance metric (AUC) of your prediction

# ##     * Balance the positive and negative weights via scale_pos_weight

# ##     * Use AUC for evaluation

# ## ALSO NOTE: I ran GridSearchCV sequentially on subsets of parameter options, rather than all at once

# ## in order to optimize parameters in a short period of time.



## ROUND 1

param_grid = {

     'max_depth': [3, 4, 5],

     'learning_rate': [0.1, 0.01, 0.05],

     'gamma': [0, 0.25, 1.0],

     'reg_lambda': [0, 1.0, 10.0],

     'scale_pos_weight': [1, 3, 5] # NOTE: XGBoost recommends sum(negative instances) / sum(positive instances)

}

# Output: {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 10.0, 'scale_pos_weight': 1}

# Because learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...



# ## ROUND 2

param_grid = {

     'max_depth': [4],

     'learning_rate': [0.1, 0.5, 1],

     'gamma': [0.25],

     'reg_lambda': [10.0, 20, 100],

      'scale_pos_weight': [1]

}

# ## Output: {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 10.0, 'scale_pos_weight': 1}



## NOTE: To speed up cross validiation, and to further prevent overfitting.

## We are only using a random subset of the data (90%) and are only

## using a random subset of the features (columns) (50%) per tree.

optimal_params = GridSearchCV(

     estimator=xgb.XGBClassifier(objective='binary:logistic', 

                                 seed=42,

                                 subsample=0.9,

                                 colsample_bytree=0.5),

     param_grid=param_grid,

     scoring='roc_auc', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

     verbose=0, # NOTE: If you want to see what Grid Search is doing, set verbose=2

     n_jobs = 10,

     cv = 3

 )



optimal_params.fit(X_train, 

                    y_train, 

                    early_stopping_rounds=10,                

                    eval_metric='auc',

                    eval_set=[(X_test, y_test)],

                    verbose=False)

print(optimal_params.best_params_)
# Evaluate Optimized Model

clf_xgb = xgb.XGBClassifier(seed=42,

                        objective='binary:logistic',

                        gamma=0.25,

                        learn_rate=0.01,

                        max_depth=4,

                        reg_lambda=10,

                        scale_pos_weight=1,

                        subsample=0.9,

                        colsample_bytree=0.5)

clf_xgb.fit(X_train, 

            y_train, 

            verbose=True, 

            early_stopping_rounds=10,

            eval_metric='aucpr',

            eval_set=[(X_test, y_test)])
preds = clf_xgb.predict(X_test)



accuracy = (preds == y_test).sum().astype(float) / len(preds)*100



print("XGBoost's prediction accuracy with optimal hyperparameters is: %3.2f" % (accuracy))
plot_confusion_matrix(clf_xgb, 

                      X_test, 

                      y_test,

                      values_format='d',

                      display_labels=["Did not leave", "Left"])
predictor_cols = X_test.columns

# Use the model to make predictions

predicted = clf_xgb.predict(test[predictor_cols])

# We will look at the predicted prices to ensure we have something sensible.

print(predicted)
submission = pd.DataFrame({'id': test.id, 'churn': predicted})



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'churn.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)