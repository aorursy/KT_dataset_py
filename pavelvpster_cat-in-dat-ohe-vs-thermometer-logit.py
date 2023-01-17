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
day_map = {

    1: 3,

    2: 2,

    3: 1,

    4: 0,

    5: 1,

    6: 2,

    7: 3

}



train['day'] = train['day'].map(day_map)

test['day'] = test['day'].map(day_map)
features_to_drop_most_frequent_values = ['nom_1', 'nom_3', "day"]



most_frequent_values = [train[column].value_counts().index[0] for column in features_to_drop_most_frequent_values]

most_frequent_values
train.head().T
thermometer_features = ['ord_1', 'ord_2', 'ord_3', 'ord_4', 'month', 'ord_5_1']



ohe_2_features = ['nom_1', 'nom_3', 'day']

ohe_3_features = ['nom_8']

ohe_1_features = set(test.columns) - set(ohe_2_features) - set(ohe_3_features) - set(thermometer_features)
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
from sklearn.preprocessing import OneHotEncoder
ohe_1 = OneHotEncoder(dtype='uint8', handle_unknown="ignore")

ohe_1_traintest = ohe_1.fit_transform(traintest[ohe_1_features])
ohe_2 = OneHotEncoder(drop=most_frequent_values, dtype='uint8')

ohe_2_traintest = ohe_2.fit_transform(traintest[ohe_2_features])
ohe_3 = OneHotEncoder(drop='first', dtype='uint8')

ohe_3_traintest = ohe_3.fit_transform(traintest[ohe_3_features])
import scipy





ohe_traintest = scipy.sparse.hstack([ohe_1_traintest, ohe_2_traintest, ohe_3_traintest]).tocsr()



del ohe_1_traintest

del ohe_2_traintest

del ohe_3_traintest
thermometer = ThermometerEncoder(thermometer_features)

thermometer_traintest = thermometer.fit_transform(traintest)
import scipy





final_traintest = scipy.sparse.hstack([ohe_traintest, thermometer_traintest]).tocsr()
final_x_train = final_traintest[:train_part]

final_x_test  = final_traintest[train_part:]
del x_train

del x_test
final_x_train.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
logit_param_grid = {

    'C': [0.100, 0.150, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150]

}



logit_grid = GridSearchCV(LogisticRegression(solver='lbfgs'), logit_param_grid,

                          scoring='roc_auc', cv=5, n_jobs=-1, verbose=0)

# logit_grid.fit(final_x_train, y_train)



# best_C = logit_grid.best_params_['C']

best_C = 0.14



print('Best C:', best_C)
logit = LogisticRegression(C=best_C, solver="lbfgs", max_iter=5000)

logit.fit(final_x_train, y_train)

y_full_train = logit.predict_proba(final_x_test)[:, 1]
submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')
submission['target'] = y_full_train

submission.to_csv('logit-full-train.csv')
submission.head()