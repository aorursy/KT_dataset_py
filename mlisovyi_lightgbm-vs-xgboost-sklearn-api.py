import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
train = pd.read_csv('../input/train.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title('Train')
age_mean = train['Age'].mean()
train.fillna(age_mean, inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title('Train')
# Add one-hot encoding for the Sex, as we know that women were more likely to survive than men
sex_cats_dummies    = pd.get_dummies(train['Sex'], prefix='Sex', drop_first=True) 

# Add dummies for Pclass as well, as it is a category, although stored as an int
pclass_cats_dummies = pd.get_dummies(train['Pclass'], prefix='Pclass', drop_first=True)

# Add one-hot-encoded features to the main dataset
train = pd.concat([train, sex_cats_dummies, pclass_cats_dummies], axis=1)

# Drop categorical columns. Feel free to extract more information from those before droping
train.drop(['Sex','Embarked','Name','Ticket', 'Pclass', 'PassengerId', 'Cabin'], axis=1, inplace=True)
train.head()
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True)
from sklearn.model_selection import train_test_split
# We use the 80/20 split here to match the 5-fold cross-validation in the next section
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.20, 
                                                    random_state=314)
# Added to suppress warnings from model fiting of type
# DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. 
import warnings
warnings.filterwarnings('ignore', 'The truth value of an empty array is ambiguous. .*')
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_validate
# Let's define several simple randomly-picked LighGBM and XGBoost models.
models = {'XGB_depth3':  xgb.XGBClassifier(max_depth=3, random_state=314, seed=314, silent=True),
          'XGB_depth5':  xgb.XGBClassifier(max_depth=5, random_state=314, seed=314, silent=True),
          'XGB_depth7':  xgb.XGBClassifier(max_depth=7, random_state=314, seed=314, silent=True),
          'XGB_depth7_regularised':  xgb.XGBClassifier(max_depth=7, colsample_bytree= 0.90, reg_lambda= 1, subsample= 0.80, random_state=314, seed=314, silent=True),
          'LGB_depthINFleaves7': lgb.LGBMClassifier(max_depth=-1, num_leaves=7, random_state=314),
          'LGB_depthINFleaves20': lgb.LGBMClassifier(max_depth=-1, num_leaves=20, random_state=314),
          'LGB_depth3':   lgb.LGBMClassifier(max_depth=3, random_state=314),
          'LGB_depth5':  lgb.LGBMClassifier(max_depth=5, random_state=314)}
# These will be pandas.DataFrame's to store the final performance values for plotting
acc_valid_summary = pd.DataFrame(index=sorted(models.keys()), columns=['raw'], dtype=np.float32)
acc_train_summary = pd.DataFrame(index=sorted(models.keys()), columns=['raw'], dtype=np.float32)
fit_time_summary  = pd.DataFrame(index=sorted(models.keys()), columns=['raw'], dtype=np.float32)


def benchmarkModels(models_dict, transform_name='raw'):
    '''
    The function is used to evaluate performance of each algorithm
    
    Parameters
    ----------
    models_dict: dictionary 
        A dictionary of models for evaluation. 
        The items are name string and sklearn-like classifier pairs
    trasform_name: string
        Not used in this example, but allows one to evaluate performance on different data transformations
    '''
    for clf_name, clf in models_dict.items():
        clf.fit(X_train, y_train)
        acc_valid_summary.loc[clf_name, transform_name] = accuracy_score(y_test, clf.predict(X_test))
        acc_train_summary.loc[clf_name, transform_name] = accuracy_score(y_train, clf.predict(X_train))
def plotPerformance(perf_valid, perf_test, perf_fit_time=None, suff=''):
    n_plots = 3 if isinstance(perf_fit_time, pd.DataFrame) else 2
    #create a figure with 2 or subplots
    fig, ax = plt.subplots(ncols=n_plots, figsize=(12,6))
    # increase the white space to fil it long Y axis labels
    fig.subplots_adjust(wspace=1.25)
    
    # The comparison of the two tells us about amount of overtraining
    # performance of the VALIDATION sample 
    sns.heatmap(perf_valid, cmap='Blues', annot=True, vmin=0.75, vmax=0.9, ax=ax[0])
    ax[0].set_title('Accuracy on VALIDATION sample ' + suff)
    # performance of the TRAINING sample 
    sns.heatmap(perf_test, cmap='Blues', annot=True, vmin=0.75, vmax=0.9, ax=ax[1])
    ax[1].set_title('Accuracy on TRAIN sample ' + suff)
    # Plot also trainign time, if provided
    if len(ax) > 2:
        sns.heatmap(perf_fit_time, cmap='Blues', annot=True, ax=ax[2])
        ax[2].set_title('Training time ' + suff)
benchmarkModels(models)
plotPerformance(acc_valid_summary, acc_train_summary, suff='(Train/Test split)')
acc_valid_cv_summary = pd.DataFrame(index=sorted(models.keys()), columns=['raw'], dtype=np.float32)
acc_train_cv_summary = pd.DataFrame(index=sorted(models.keys()), columns=['raw'], dtype=np.float32)
fit_time_cv_summary  = pd.DataFrame(index=sorted(models.keys()), columns=['raw'], dtype=np.float32)

def benchmarkModelsCV(models_dict, transform_name='raw'):
    for clf_name, clf in models_dict.items():
        # Let's run cross validation on the classifier
        # Note, that  we run it on the full train+test sample:
        # this increases statistics and we do not need the test sample
        score = cross_validate(clf,
                               X = train.drop('Survived', axis=1),
                               y = train['Survived'],
                               scoring=make_scorer(accuracy_score, greater_is_better=True),
                               cv=5, 
                               return_train_score=True)
        # save evaluated performance results
        acc_valid_cv_summary.loc[clf_name, transform_name] = score['test_score'].mean()
        acc_train_cv_summary.loc[clf_name, transform_name] = score['train_score'].mean()
        fit_time_cv_summary.loc[clf_name, transform_name] = score['fit_time'].mean()

benchmarkModelsCV(models)
plotPerformance(acc_valid_cv_summary, acc_train_cv_summary, perf_fit_time=fit_time_cv_summary, suff='(CV mean)')
