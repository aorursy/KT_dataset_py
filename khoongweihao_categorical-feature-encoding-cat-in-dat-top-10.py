import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat/train.csv', index_col='id')

test = pd.read_csv('../input/cat-in-the-dat/test.csv', index_col='id')
train = train.drop('bin_0', axis=1)

test = test.drop('bin_0', axis=1)
map_bin_3_4 = {'T': 1, 'F': 0, 'Y': 1, 'N': 0}



train['bin_3'] = train['bin_3'].map(map_bin_3_4)

test['bin_3'] = test['bin_3'].map(map_bin_3_4)



train['bin_4'] = train['bin_4'].map(map_bin_3_4)

test['bin_4'] = test['bin_4'].map(map_bin_3_4)
train['ord_5_1'] = train['ord_5'].str[0]

train['ord_5_2'] = train['ord_5'].str[1]

train = train.drop('ord_5', axis=1)



test['ord_5_1'] = test['ord_5'].str[0]

test['ord_5_2'] = test['ord_5'].str[1]

test = test.drop('ord_5', axis=1)
train = train.drop('ord_5_2', axis=1)

test = test.drop('ord_5_2', axis=1)
ord_1_values = ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']



map_ord_1 = lambda x: ord_1_values.index(x)



train['ord_1'] = train['ord_1'].apply(map_ord_1)

test['ord_1'] = test['ord_1'].apply(map_ord_1)
ord_2_values = ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']



map_ord_2 = lambda x: ord_2_values.index(x)



train['ord_2'] = train['ord_2'].apply(map_ord_2)

test['ord_2'] = test['ord_2'].apply(map_ord_2)
import string





map_to_ascii_index = lambda x: string.ascii_letters.index(x)



# 'ord_5_2' dropped!

for column in ['ord_3', 'ord_4', 'ord_5_1']:

    train[column] = train[column].apply(map_to_ascii_index)

    test[column] = test[column].apply(map_to_ascii_index)
# columns_to_test = list(test.columns)



columns_to_test = ['nom_7', 'nom_8', 'nom_9']



replace_xor = lambda x: 'xor' if x in xor_values else x



for column in columns_to_test:

    xor_values = set(train[column].unique()) ^ set(test[column].unique())

    if xor_values:

        print('Column', column, 'has', len(xor_values), 'XOR values')

        train[column] = train[column].apply(replace_xor)

        test[column] = test[column].apply(replace_xor)

    else:

        print('Column', column, 'has no XOR values')
def get_infrequent_values(data, column, threshold):

    value_counts = data[column].value_counts()

    return list(value_counts[value_counts < threshold].index)
for column in train.columns:

    n = len(get_infrequent_values(train, column, 3))

    if n > 0:

        print('Column', column, 'has', n, 'unique infrequent value(s)')
thresholds = {

    # 'nom_7': 3,

    # 'nom_8': 3,

    'nom_9': 3

}



for column in thresholds.keys():

    values_to_replace = get_infrequent_values(train, column, thresholds[column])

    

    replace = lambda x: 'value' if x in values_to_replace else x

    

    train[column] = train[column].transform(replace)

    test[column] = test[column].transform(replace)
for column in thresholds.keys():

    print(column, ':', train[column].value_counts()['value'], 'values replaced in train dataset')
train.head().T
all_features = test.columns



selected_features_for_thermometer = ['ord_1', 'ord_4', 'day', 'month', 'ord_5_1']



selected_features_for_both = ['ord_2', 'ord_3']
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()

del test
from sklearn.base import BaseEstimator, TransformerMixin





# This implementation supports numeric features only

class ThermometerEncoder(BaseEstimator, TransformerMixin):

    

    def __init__(self, cols=None, drop_invariant=True):

        self.cols = cols

        self.drop_invariant = drop_invariant

    

    def get_params(self, deep=True):

        return {'cols': self.cols, 'drop_invariant': self.drop_invariant}

    

    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)

        return self

    

    def fit(self, X, y=None):

        self.bars = {}

        for c in self.cols:

            k = np.arange(X[c].max() + 1)

            self.bars[c] = (k[:-1] < k.reshape(-1, 1)).astype(int)

        return self

    

    def transform(self, X, y=None):

        out = pd.DataFrame(index=X.index)

        for c in self.cols:

            out = out.join(self.transform_one(X, c))

        

        if self.drop_invariant:

            columns_to_drop = []

            for c in out.columns:

                if len(out[c].unique()) == 1:

                    columns_to_drop.append(c)

            out = out.drop(columns_to_drop, axis=1)

        

        return out

    

    def transform_one(self, X, c):

        bars = self.bars[c]

        out = pd.DataFrame(index=X.index, data=bars[X[c]])

        out.columns = [c + '_' + str(k) for k in range(bars.shape[1])]

        return out
train_part = len(x_train)

traintest = pd.concat([x_train, x_test])
ohe_features = set(all_features) - set(selected_features_for_thermometer)
ohe_traintest = pd.get_dummies(traintest[ohe_features], columns=ohe_features, drop_first=True, sparse=True).sparse.to_coo().tocsr()
thermometer = ThermometerEncoder(selected_features_for_thermometer + selected_features_for_both)

thermometer_traintest = thermometer.fit_transform(traintest)
import scipy





final_traintest = scipy.sparse.hstack([ohe_traintest, thermometer_traintest]).tocsr()
final_x_train = final_traintest[:train_part]

final_x_test  = final_traintest[train_part:]
del x_train

del x_test
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

from cross_validation_framework import *
logit_param_grid = {

    'C': [0.100, 0.150, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150]

}



logit_grid = GridSearchCV(LogisticRegression(solver='lbfgs'), logit_param_grid,

                          scoring='roc_auc', cv=5, n_jobs=-1, verbose=0)

logit_grid.fit(final_x_train, y_train)



best_C = logit_grid.best_params_['C']

# best_C = C = 0.12345



print('Best C:', best_C)
logit = LogisticRegression(C=best_C, solver='lbfgs', max_iter=10000)

cv = KFold(n_splits=10, random_state=42)

_, trained_estimators = fit(ScikitLearnPredictProbaEstimator(logit), roc_auc_score, final_x_train, y_train, cv)

y = predict(trained_estimators, final_x_test)
logit = LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000)

logit.fit(final_x_train, y_train)

y_full_train = logit.predict_proba(final_x_test)[:, 1]
submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')

submission['target'] = y

submission.to_csv('logit.csv')
submission.head()
submission['target'] = y_full_train

submission.to_csv('logit-full-train.csv')
submission.head()