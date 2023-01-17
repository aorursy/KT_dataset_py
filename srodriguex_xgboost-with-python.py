# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from itertools import combinations
from sklearn import cross_validation as cv
import xgboost as xgb
from numpy import array, array_equal # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).
decode("utf8"))

# Any results you write to the current directory are saved as output.

def write_file(prediction, file_name='to_submit.csv'):
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
    print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))
    
    
def write_results(prediction, file_name='submission.csv'):
    submit = pd.DataFrame(index=test_dataset.index, data=prediction, columns=['TARGET'])
    submit.head()
    submit.to_csv(file_name, header=True)
    print('Results written to file "{}".'.format(file_name))
train_dataset = pd.read_csv('../input/train.csv', index_col='ID')
test_dataset = pd.read_csv('../input/test.csv', index_col='ID')
# Drop the constant features

constant_features_train = set(identify_constant_features(train_dataset))
print('There were {} constant features in TRAIN dataset.'.format(
        len(constant_features_train)))
train_dataset.drop(constant_features_train, inplace=True, axis=1)

print_shapes()
# Drop equal features

equal_features_train = identify_equal_features(train_dataset)

print('There were {} pairs of equal features in TRAIN dataset.'.format(len(equal_features_train)))

features_to_drop = array(equal_features_train)[:,1] 
train_dataset.drop(features_to_drop, axis=1, inplace=True)

print_shapes()
# Define the variables model.

y_name = 'TARGET'
feature_names = train_dataset.columns.tolist()
feature_names.remove(y_name)

X = train_dataset[feature_names]
y = train_dataset[y_name]

X_test = test_dataset[feature_names]
# Save the features selected for later use.
pd.Series(feature_names).to_csv('features_selected_step1.csv', index=False)
%%time
# Score the selected model

skf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = cv.cross_val_score(xgb.XGBClassifier(), X, y, cv=skf, scoring=score_metric)
scores
%%time
# Train model

model = xgb.XGBClassifier()
model.fit(X,y)
# Predict
yhat = model.predict_proba(X_test)[:,1]
write_results(yhat)