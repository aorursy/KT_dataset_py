# THIS WAS MY FIRST SCRIPT AND IT HAS SOME FLAWS. 
# IT'S STILL HERE BECAUSE I GOT FOND OF IT.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%pylab inline
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from itertools import combinations
from numpy import array,array_equal

from sklearn import cross_validation as cv
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model 
from sklearn import naive_bayes 

import xgboost as xgb


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]
).decode("utf8"))


# Any results you write to the current directory are saved as output.
def print_shapes():
    print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))
train_dataset = pd.read_csv('../input/train.csv', index_col='ID')
test_dataset = pd.read_csv('../input/test.csv', index_col='ID')

print_shapes()
# How many nulls are there in the datasets?
nulls_train = (train_dataset.isnull().sum()==1).sum()
nulls_test = (test_dataset.isnull().sum()==1).sum()
print('There are {} nulls in TRAIN and {} nulls in TEST dataset.'.format(nulls_train, nulls_test))
# Remove constant features

def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

constant_features_train = set(identify_constant_features(train_dataset))

print('There were {} constant features in TRAIN dataset.'.format(
        len(constant_features_train)))

# Drop the constant features
train_dataset.drop(constant_features_train, inplace=True, axis=1)


print_shapes()
# Remove equals features

def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = array_equal(dataframe[compare[0]],dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features

equal_features_train = identify_equal_features(train_dataset)

print('There were {} pairs of equal features in TRAIN dataset.'.format(len(equal_features_train)))

# Remove the second feature of each pair.

features_to_drop = array(equal_features_train)[:,1] 
train_dataset.drop(features_to_drop, axis=1, inplace=True)

print_shapes()
# Define the variables model.

y_name = 'TARGET'
feature_names = train_dataset.columns.tolist()
feature_names.remove(y_name)

X = train_dataset[feature_names] # data
y = train_dataset[y_name] # class

# Save the features selected for later use.
pd.Series(feature_names).to_csv('features_selected_step1.csv', index=False)
print('Features selected\n{}'.format(feature_names))
   
    
# Proportion of classes
y.value_counts()/len(y)

skf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)



#score_metric = 'roc_auc'
#scores = {}

#def score_model(model):
#    return cv.cross_val_score(model, X, y, cv=skf, scoring=score_metric)

# time: 10s
#scores['tree'] = score_model(tree.DecisionTreeClassifier()) 

# time: 9s
#scores['extra_tree'] = score_model(ensemble.ExtraTreesClassifier())

# time: 7s
#scores['forest'] = score_model(ensemble.RandomForestClassifier())

# time: 33s
#scores['ada_boost'] = score_model(ensemble.AdaBoostClassifier())

# time: 1min
#scores['bagging'] = score_model(ensemble.BaggingClassifier())

# time: 2min30s
#scores['grad_boost'] = score_model(ensemble.GradientBoostingClassifier())

# time: 49s
#scores['ridge'] = score_model(linear_model.RidgeClassifier())

# time: 4s
#scores['passive'] = score_model(linear_model.PassiveAggressiveClassifier())

# time: 4s
#scores['sgd'] = score_model(linear_model.SGDClassifier())

# time: 3s
#scores['gaussian'] = score_model(naive_bayes.GaussianNB())

# time: 4min
#scores['xgboost'] = score_model(xgb.XGBClassifier())


# Print the scores
#model_scores = pd.DataFrame(scores).mean()
#model_scores.sort_values(ascending=False)
#model_scores.to_csv('model_scores.csv', index=False)
#print('Model scores\n{}'.format(model_scores))

# xboost classifier
X_fit, X_eval, y_fit, y_eval= cv.train_test_split(X, y, test_size=0.3)
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=570, learning_rate=0.02, nthread=4, subsample=0.68, colsample_bytree=0.7, seed=1235)
# fitting
clf.fit(X, y, early_stopping_rounds=500, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', metrics.roc_auc_score(y, clf.predict_proba(X)[:,1]))

X_test = test_dataset[feature_names]
# predicting
y_pred= clf.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":X_test.index, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)
