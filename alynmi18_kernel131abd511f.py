# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.impute import SimpleImputer

import category_encoders as ce

from sklearn.preprocessing import OrdinalEncoder

from fancyimpute import KNN 

import xgboost as xgb

from sklearn.metrics import roc_auc_score
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
train_y = train['target']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
train.info()
# how many missing values?

num_of_missing = train.isnull().sum()

print(num_of_missing)
# reduce features from one hot encoding

# nom_3: 'Russia', nan, 'Canada', 'Finland', 'Costa Rica', 'China', 'India'

# train.nom_3 = train.nom_3.replace({'Russia': 'Eurasia', 'China': 'Asia', 'India': 'Asia', 'Finland': 'Europe', 'Canada': 'North America', 'Costa Rica':'North America', 'nan': np.nan})

# test.nom_3 = test.nom_3.replace({'Russia': 'Eurasia', 'China': 'Asia', 'India': 'Asia', 'Finland': 'Europe', 'Canada': 'North America', 'Costa Rica':'North America', 'nan': np.nan})
# reduce features from one hot encoding

# nom_4: 'Bassoon', 'Theremin', nan, 'Oboe', 'Piano'

# train.nom_4 = train.nom_4.replace({'Bassoon': 'Woodwind', 'Oboe': 'Woodwind', 'Piano': 'Strings', 'Theremin':'Electronic', 'nan': np.nan})

# test.nom_4 = test.nom_4.replace({'Bassoon': 'Woodwind', 'Oboe': 'Woodwind', 'Piano': 'Strings', 'Theremin':'Electronic', 'nan': np.nan})
# one hot encode: 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'

train = pd.get_dummies(train, prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])

test = pd.get_dummies(test, prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])
# binary encoder: 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

ce_bin = ce.BinaryEncoder(cols=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])

train = ce_bin.fit_transform(train)

test = ce_bin.fit_transform(test)
# encode T/F and Y/N cols: 'bin_3', 'bin_4'

train.bin_3 = train.bin_3.replace({'T': 1, 'F': 0})

train.bin_4 = train.bin_4.replace({'Y': 1, 'N': 0})

test.bin_3 = test.bin_3.replace({'T': 1, 'F': 0})

test.bin_4 = test.bin_4.replace({'Y': 1, 'N': 0})
# ordinal encoding: 'ord_1', 'ord_2', 'ord_3'

ord_1 = 'Grandmaster', 'Master', 'Expert', 'Novice', 'Contributor'

train.ord_1 = train.ord_1.replace({'Grandmaster': 5, 'Master': 4, 'Expert': 3, 'Novice': 2, 'Contributor': 1, 'nan': np.nan})

test.ord_1 = test.ord_1.replace({'Grandmaster': 5, 'Master': 4, 'Expert': 3, 'Novice': 2, 'Contributor': 1, 'nan': np.nan})
# ord_2 = 'Lava Hot', 'Boiling Hot', 'Hot', 'Warm', 'Cold', 'Freezing'

#col_two_list = X_train.ord_2.unique()

train.ord_2 = train.ord_2.replace({'Lava Hot': 6, 'Boiling Hot': 5, 'Hot': 4, 'Warm': 3, 'Cold': 2, 'Freezing': 1, 'nan': np.nan})

test.ord_2 = test.ord_2.replace({'Lava Hot': 6, 'Boiling Hot': 5, 'Hot': 4, 'Warm': 3, 'Cold': 2, 'Freezing': 1, 'nan': np.nan})
# ord_3 : ['c' 'e' 'm' 'd' 'b' 'o' 'i' 'n' 'f' 'k' 'l' nan 'h' 'a' 'g' 'j'] = 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'

#col_three_list = X_train.ord_3.unique()

train.ord_3 = train.ord_3.replace({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 

                                       'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,

                                       'm': 13, 'n': 14, 'o': 15})

test.ord_3 = test.ord_3.replace({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 

                                       'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,

                                       'm': 13, 'n': 14, 'o': 15})
# map ord 4 to numerical

train['ord_4'].fillna('ZNaN', inplace=True)

col_four_list = train.ord_4.unique()

col_four_list.sort()

dict_ord4 = {col_four_list[i] :i for i in range(0, len(col_four_list))}

train.ord_4 = train.ord_4.replace(dict_ord4)

test.ord_4 = test.ord_4.replace(dict_ord4)
# replace 'ZNan' with NaN

train.ord_4 = train.ord_4.replace({26: np.nan})

test.ord_4 = test.ord_4.replace({26: np.nan})
# map ord 5 to numerical

train['ord_5'].fillna('zzzNaN', inplace=True)

col_five_list = train.ord_5.unique()

col_five_list.sort()

dict_ord5 = {col_five_list[i] : i for i in range(0, len(col_five_list))}

train.ord_5 = train.ord_5.replace(dict_ord5)

test.ord_5 = test.ord_5.replace(dict_ord5)
# replace 'zzzNaN' with NaN

train.ord_5 = train.ord_5.replace({190: np.nan})

test.ord_5 = test.ord_5.replace({190: np.nan})
# imp_mean = IterativeImputer(random_state=0)

# train_imputed = imp_mean.fit_transform(train)
simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

train_imputed = simple_imputer.fit_transform(train)
train_imputed = pd.DataFrame(train_imputed, index=train.index, columns=train.columns)

train_imputed.head(5)
# test_imputed = simple_imputer.fit_transform(test)

# test_imputed = pd.DataFrame(test, index=test.index, columns=train.columns)
train_imputed['ord_1'] = pd.to_numeric(train_imputed['ord_1'])

train_imputed['ord_2'] = pd.to_numeric(train_imputed['ord_2'])

train_imputed['ord_4'] = pd.to_numeric(train_imputed['ord_4'])

train_imputed['ord_5'] = pd.to_numeric(train_imputed['ord_5'])
xg_reg = xgb.XGBRegressor(objective = 'binary:logistic', 

                          verbosity = '1', 

                          eta = 0.7, 

                          max_depth = 3)

xg_reg.fit(train_imputed,train_y)

test['ord_1'] = pd.to_numeric(test['ord_1'])

test['ord_2'] = pd.to_numeric(test['ord_2'])

test['ord_4'] = pd.to_numeric(test['ord_4'])

test['ord_5'] = pd.to_numeric(test['ord_5'])
preds = xg_reg.predict(test)
#print(roc_auc_score(test,preds))
submission = pd.DataFrame({'id': test_id, 'target': preds})

submission.to_csv('submission.csv', index=False)