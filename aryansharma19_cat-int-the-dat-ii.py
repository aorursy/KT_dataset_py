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
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

train.drop(['id'], axis = 1, inplace = True)

train.head()
y = train.iloc[:, -1]

train.drop(['target'], axis = 1, inplace = True)
train.isnull().sum()
binary_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

binary = train[binary_cols]

for col in binary_cols:

    print('---------------------' + col + '----------------------------\n')

    print(binary[col].value_counts())
print('------------------------ TARGET COMPOSITION-----------------------\n')

print(f'ZERO : {y.value_counts()[0]*100/len(y)} %,       ONE : {y.value_counts()[1]*100/len(y)} %')
import seaborn as sns

sns.set_style('white')



sns.heatmap(binary.isnull(), yticklabels=False)
binary['bin_0'].fillna(0.0, inplace = True)

binary['bin_1'].fillna(0.0, inplace = True)

binary['bin_2'].fillna(0.0, inplace = True)

binary['bin_3'].fillna('F', inplace = True)

binary['bin_4'].fillna('N', inplace = True)





sns.heatmap(binary.isnull(), yticklabels=False)
binary['bin_3'].value_counts()
binary['bin_3'] = binary['bin_3'].apply(lambda x: 1 if x == 'T' else 0)

binary['bin_4'] = binary['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)



binary.bin_3.value_counts()
binary['bin_4'].value_counts()
nominal_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

nominal = train[nominal_cols]

for col in nominal_cols:

    print('---------------------' + col + '----------------------------\n')

    print(nominal[col].value_counts())
sns.heatmap(nominal.isnull(), yticklabels=False)
nominal['nom_0'].mode()[0]
def imputer_for_nom(data):

    data.fillna(data.mode()[0], inplace = True)



for col in nominal_cols:

    imputer_for_nom(nominal[col])

    

nominal.isnull().sum()
sns.heatmap(nominal.isnull(), yticklabels=False)
from sklearn import base

from sklearn.model_selection import KFold

class KFoldTargetEncoderTrain(base.BaseEstimator,

                               base.TransformerMixin):

    def __init__(self,colnames,targetName,

                  n_fold=5, verbosity=True,

                  discardOriginal_col=False):

        self.colnames = colnames

        self.targetName = targetName

        self.n_fold = n_fold

        self.verbosity = verbosity

        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):

        return self

    def transform(self,X):

        assert(type(self.targetName) == str)

        assert(type(self.colnames) == str)

        print(self.colnames)

        print(X.columns)

        assert(self.colnames in X.columns)

        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()

        kf = KFold(n_splits = self.n_fold,

                   shuffle = False, random_state=2019)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'

        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):

            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())

            X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:

            encoded_feature = X[col_mean_name].values

            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,np.corrcoef(X[self.targetName].values,encoded_feature)[0][1]))

        if self.discardOriginal_col:

            X = X.drop(self.targetName, axis=1)

        return X
nominal_sub_data = pd.concat([nominal, y], axis=1)

cols = nominal_sub_data.columns

data = []

print(cols)

for col in cols[:-1]:

    nom = nominal_sub_data.loc[:,[col, 'target']]

    targetc = KFoldTargetEncoderTrain(nom.columns[0],'target',n_fold=5)

    new_train = targetc.fit_transform(nom)

    data.append(new_train)

print(data)

target_encoded_nominal_data = []

for i in data:

    target_encoded_nominal_data.append([i.columns[-1], i.iloc[:, -1]])

columns = [i[0] for i in target_encoded_nominal_data]

x = pd.Series(target_encoded_nominal_data[0][1], name = columns[0])

for i in range(1, len(target_encoded_nominal_data)):

    temp = pd.Series(target_encoded_nominal_data[i][1], name = columns[i])

    x = pd.concat([x, temp], axis = 1)

x.head()
x.info()
ordinal_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

ordinal = train[ordinal_cols]

ordinal.head()
sns.heatmap(ordinal.isnull())
ordinal.isnull().sum()
ordinal.ord_0.value_counts()
ordinal.apply(lambda x: x.fillna(x.mode()[0], inplace = True))

ordinal.isnull().sum()
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

for col in ordinal_cols:

    ordinal[col] = encoder.fit_transform(ordinal[col])

ordinal.head()
date_feat = train[['day', 'month']]

date_feat.head()
date_feat.apply(lambda x: x.fillna(x.mode()[0], inplace = True))

date_feat.isnull().sum()
preprocessed_train = pd.concat([binary, x, ordinal, date_feat], axis = 1)

preprocessed_train.head()
from imblearn.over_sampling import SMOTE

smote = SMOTE()



TEMP = smote.fit_sample(preprocessed_train, y)
cols = preprocessed_train.columns

preprocessed_train = pd.DataFrame(TEMP[0], columns = cols)

y = pd.Series(TEMP[1], name = 'target')

preprocessed_train.shape, y.shape
from sklearn.model_selection import KFold, cross_val_score

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score



kfold = KFold(n_splits = 10, shuffle = True, random_state = 101)

for train_idx, test_idx in kfold.split(preprocessed_train, y):

    x_train, x_test = preprocessed_train.iloc[train_idx], preprocessed_train.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] 



xgb = XGBClassifier()

xgb.fit(x_train, y_train)

xgb_pred = xgb.predict_proba(x_test)[::, 1]

roc_auc_score(y_test, xgb_pred)
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

test_id = test['id']

test.drop(['id'], axis = 1, inplace = True)

test.head()
def binary_imputer(binary):

    binary['bin_0'].fillna(0.0, inplace = True)

    binary['bin_1'].fillna(0.0, inplace = True)

    binary['bin_2'].fillna(0.0, inplace = True)

    binary['bin_3'].fillna('F', inplace = True)

    binary['bin_4'].fillna('N', inplace = True)

    binary['bin_3'] = binary['bin_3'].apply(lambda x: 1 if x == 'T' else 0)

    binary['bin_4'] = binary['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)

    

    

binary_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

binary_test = test[binary_cols]

binary_imputer(binary_test)
nominal_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

nominal_test = test[nominal_cols]



nominal_test.info()



def imputer_for_nom(data):

    data.fillna(data.mode()[0], inplace = True)



for col in nominal_cols:

    imputer_for_nom(nominal_test[col])

print(nominal_test.columns)



class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

    def __init__(self,train,colNames,encodedName):

        self.train = train

        self.colNames = colNames

        self.encodedName = encodedName

        

    def fit(self, X, y=None):

        return self



    def transform(self,X):

        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 

        dd = {}

        for index, row in mean.iterrows():

            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]

        X = X.replace({self.encodedName: dd})

        return X

    

test_nominal_data = [] 

for i in range(len(data)):

    test_targetc = KFoldTargetEncoderTest(data[i],data[i].columns[0],data[i].columns[-1])

    test_sub = pd.DataFrame(nominal_test.iloc[:, i], columns = [nominal_test.columns[i]])

    print('----------------------------------------------------------------------------------')

    test_sub.info()

    print('----------------------------------------------------------------------------------')

    temp = test_targetc.fit_transform(test_sub)

    test_nominal_data.append(temp)

    

print(test_nominal_data)
nominal_test_data = pd.Series(test_nominal_data[0].iloc[:, -1], name = test_nominal_data[0].columns[-1], dtype='float64')

for i in range(1, len(test_nominal_data)):

    nominal_test_data = pd.concat([nominal_test_data, pd.Series(test_nominal_data[i].iloc[:, -1], name = test_nominal_data[i].columns[-1], dtype='float64')], axis = 1)

nominal_test_data.head()
ordinal_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

ordinal_test = test[ordinal_cols]

ordinal_test.apply(lambda x: x.fillna(x.mode()[0], inplace = True))



from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

for col in ordinal_cols:

    ordinal_test[col] = encoder.fit_transform(ordinal_test[col])

ordinal_test.head()
date_feat_test = test[['day', 'month']]

date_feat_test.apply(lambda x: x.fillna(x.mode()[0], inplace = True))
preprocessed_test = pd.concat([binary_test, nominal_test_data, ordinal_test, date_feat_test], axis = 1)

preprocessed_test.head()
preprocessed_test.nom_6_Kfold_Target_Enc.replace('a885aacec', 0.208084, inplace = True)
preprocessed_test.info()
predictions = xgb.predict_proba(preprocessed_test)[::, 1]
submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

submission.target = predictions

submission.head()
submission.to_csv('submission.csv', index = False)