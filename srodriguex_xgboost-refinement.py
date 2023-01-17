# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).
decode("utf8"))

# Any results you write to the current directory are saved as output.
from itertools import combinations
from sklearn import cross_validation as cv
from sklearn import metrics
import xgboost as xgb
from numpy import array, array_equal 
from itertools import combinations
import pandas as pd
from numpy import array, array_equal
import numpy as np

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
def write_file(prediction, file_name='submission.csv'):
    global test_dataset
    submit = pd.DataFrame(index=test_dataset.index, data=prediction, columns=['TARGET'])
    submit.head()
    submit.to_csv(file_name, header=True)
    print('Results written to file "{}".'.format(file_name))

def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = array_equal(dataframe[compare[0]],dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features

def print_shapes():
    global train_dataset
    global test_dataset
    print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))


def write_results(prediction, file_name='submission.csv'):
    global test_dataset
    submit = pd.DataFrame(index=test_dataset.index, data=prediction, columns=['TARGET'])
    submit.head()
    submit.to_csv(file_name, header=True)
    print('Results written to file "{}".'.format(file_name))

def features_uncorrelated(threshold=0.8):
    correlated = array(correlation_matrix[
        correlation_matrix.abs() > threshold].index.tolist())
    return list(set(X.columns.tolist())- set(correlated[:,1]))

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Source: http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    
    #if not isinstance(dtrain, xgb.DMatrix):
    #    raise TypeError('Parameter dtrain: expected DMatrix type, got {} type.'.format(type(dtrain)))
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['TARGET'])

        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cv.cross_val_score(alg, dtrain[predictors], dtrain['TARGET'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['TARGET'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['TARGET'], dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

  
%%time

train_dataset = pd.read_csv('../input/train.csv', index_col='ID')
test_dataset = pd.read_csv('../input/test.csv', index_col='ID')

##
## Drop the constant features
constant_features_train = identify_constant_features(train_dataset)
print('There were {} constant features in TRAIN dataset.'.format(
        len(constant_features_train)))
train_dataset.drop(constant_features_train, axis=1, inplace=True)
test_dataset.drop(constant_features_train, axis=1, inplace=True)

##
## Drop equal features
equal_features_train = identify_equal_features(train_dataset)

print('There were {} pairs of equal features in TRAIN dataset.'.format(len(equal_features_train)))

features_to_drop = array(equal_features_train)[:,1]
train_dataset.drop(features_to_drop, axis=1, inplace=True)
test_dataset.drop(features_to_drop, axis=1, inplace=True)

##
## Clean

# The var3 is outlier
#var3_outlier = -999999
#train_dataset.var3.replace(var3_outlier, 0, inplace=True)
#test_dataset.var3.replace(var3_outlier, 0, inplace=True)

##
## Define the variables model.
y_name = 'TARGET'
feature_names = train_dataset.columns.tolist()
feature_names.remove(y_name)

X = train_dataset[feature_names]
y = train_dataset[y_name]

X_submit = test_dataset[feature_names]

train_dataset.shape, test_dataset.shape, X.shape, y.shape, X_submit.shape

%%time 
## Use a sklearn GBM
from sklearn import ensemble
gbm = ensemble.GradientBoostingClassifier(random_state=1)
modelfit(gbm, train_dataset, feature_names)

%%time 

## Use a sklearn GBM
from sklearn import ensemble
xgbmodel = xgb.XGBClassifier(seed=1)
modelfit(xgbmodel, train_dataset, feature_names)

# cv.cross_val_score(xgb.XGBClassifier(), X_train, y_train, cv=3, scoring=score_metric)

# Scores:  0.80735352,  0.83863037,  0.83129632
%%time

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())

%%time
params = {'objective':'binary:logistic' }
xgb.cv(params, 
       dtrain, 
       metrics='auc',
      nfolds=3)
cv.cro