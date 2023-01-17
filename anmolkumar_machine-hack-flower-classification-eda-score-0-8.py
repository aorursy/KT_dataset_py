from sklearn.utils import all_estimators

estimators = all_estimators()

for name, class_ in estimators:
    if hasattr(class_, 'predict_proba'):
        print(name)
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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, StratifiedShuffleSplit, KFold

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = [20, 6]
plt.style.use('seaborn-darkgrid')
train = pd.read_csv('/kaggle/input/machinehack-flower-class-recognition/train.csv')
test = pd.read_csv('/kaggle/input/machinehack-flower-class-recognition/test.csv')
submission = pd.read_csv('/kaggle/input/machinehack-flower-class-recognition/sample_submission.csv')
train.columns = train.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
test.columns = test.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print('Train Data shape: ', train.shape, 'Test Data shape: ', test.shape)

train.head(10)
cat_cols = train.columns[~(train.columns.isin(['class']))].tolist()
test.head()
train.nunique()
test.nunique()
train.isnull().sum()
i = 1
for column in train.columns[~(train.columns.isin(['area_code', 'region_code', 'height', 'diameter']))].tolist():
    plt.figure(figsize = (80, 10))
    plt.subplot(3, 3, i)
    sns.barplot(x = train[column].value_counts().index, y = train[column].value_counts())
    i += 1
    plt.show()
# Unique areas in train and test set

train_area = train[['area_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique areas in train data: ', len(train_area))
test_prod = test[['area_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique areas in test data: ', len(test_prod))

# Unique pairs of area and locality in train and test set

train_area_locality = train[['area_code', 'locality_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique locality - area pairs in train data: ', len(train_area_locality))

test_area_locality = test[['area_code', 'locality_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique locality - area pairs in test data: ', len(test_area_locality))

# Unique pairs of region and locality in train and test set

train_region_locality = train[['region_code', 'locality_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique locality - region pairs in train data: ', len(train_region_locality))

test_region_locality = test[['region_code', 'locality_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique locality - region pairs in test data: ', len(test_region_locality))

# Unique pairs of region and area in train and test set

train_area_region = train[['area_code', 'region_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique region - area pairs in train data: ', len(train_area_region))

test_area_region = test[['area_code', 'region_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique region - area pairs in test data: ', len(test_area_region))

# Unique pairs of area, ocality and region in train and test set

train_area_region_loc = train[['area_code', 'region_code', 'locality_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique area - region - locality pairs in train data: ', len(train_area_region_loc))

test_area_region_loc = test[['area_code', 'region_code', 'locality_code']].drop_duplicates(subset = None, keep = 'first', inplace = False)
print('Unique area - region - locality pairs in test data: ', len(test_area_region_loc))
train['type'] = 'train'
test['type'] = 'test'

master = pd.concat([train, test])
master.head()
master['diameter'] = master['diameter'].apply(lambda x: 0.5 if x == 0 else x)
master['height'] = master['height'].apply(lambda x: 0.1 if x == 0 else x)
i = 1
for column in master.columns[~(master.columns.isin(['area_code', 'region_code', 'height', 'diameter', 'type']))].tolist():
    plt.figure(figsize = (80, 10))
    plt.subplot(3, 3, i)
    sns.barplot(x = master[column].value_counts().index, y = master[column].value_counts())
    i += 1
    plt.show()
# grouping by frequency of species
species_fq = master.groupby('species').size()/len(master)
master.loc[:, "{}_freq".format('species')] = master['species'].map(species_fq)

# grouping by frequency of area, region, locality
area_fq = master.groupby('area_code').size()/len(master)
reg_fq = master.groupby('region_code').size()/len(master)
loc_fq = master.groupby('locality_code').size()/len(master)

# grouping by frequency of pairs of area, region, locality
reg_area_fq = (master.groupby(['region_code', 'area_code']).size()/len(master)).reset_index(name = 'reg_area_freq')
reg_loc_fq = (master.groupby(['region_code', 'locality_code']).size()/len(master)).reset_index(name = 'reg_loc_freq')
loc_area_fq = (master.groupby(['locality_code', 'area_code']).size()/len(master)).reset_index(name = 'loc_area_freq')

# grouping by frequency of triplets area, region, locality
reg_area_loc_fq = (master.groupby(['locality_code', 'area_code', 'region_code']).size()/len(master)).reset_index(name = 'reg_area_loc_freq')

# grouping by frequency of pairs of species to each of area, region, locality
spec_area_fq = (master.groupby(['species', 'area_code']).size()/len(master)).reset_index(name = 'spec_area_freq')
spec_reg_fq = (master.groupby(['species', 'region_code']).size()/len(master)).reset_index(name = 'spec_reg_freq')
spec_loc_fq = (master.groupby(['species', 'locality_code']).size()/len(master)).reset_index(name = 'spec_loc_freq')

# grouping by frequency of species & pairs of area, region, locality
spec_area_reg_fq = (master.groupby(['species', 'area_code', 'region_code']).size()/len(master)).reset_index(name = 'spec_area_reg_freq')
spec_area_loc_fq = (master.groupby(['species', 'area_code', 'locality_code']).size()/len(master)).reset_index(name = 'spec_area_loc_freq')
spec_loc_reg_fq = (master.groupby(['species', 'locality_code', 'region_code']).size()/len(master)).reset_index(name = 'spec_loc_reg_freq')

# grouping by frequency of species & triplets of area, region, locality
spec_area_loc_reg_fq = (master.groupby(['species', 'locality_code', 'region_code', 'area_code']).size()/len(master)).reset_index(name = 'spec_area_loc_reg_freq')

#Merging to main master
master.loc[:, "{}_freq".format('area')] = master['area_code'].map(area_fq)
master.loc[:, "{}_freq".format('reg')] = master['region_code'].map(reg_fq)
master.loc[:, "{}_freq".format('loc')] = master['locality_code'].map(loc_fq)

master = master.merge(reg_area_fq, on = ['region_code', 'area_code'], how = 'left')
master = master.merge(reg_loc_fq, on = ['region_code', 'locality_code'], how = 'left')
master = master.merge(loc_area_fq, on = ['locality_code', 'area_code'], how = 'left')

master = master.merge(reg_area_loc_fq, on = ['locality_code', 'area_code', 'region_code'], how = 'left')

master = master.merge(spec_area_fq, on = ['species', 'area_code'], how = 'left')
master = master.merge(spec_reg_fq, on = ['species', 'region_code'], how = 'left')
master = master.merge(spec_loc_fq, on = ['species', 'locality_code'], how = 'left')

master = master.merge(spec_area_reg_fq, on = ['species', 'area_code', 'region_code'], how = 'left')
master = master.merge(spec_area_loc_fq, on = ['species', 'area_code', 'locality_code'], how = 'left')
master = master.merge(spec_loc_reg_fq, on = ['species', 'locality_code', 'region_code'], how = 'left')

master = master.merge(spec_area_loc_reg_fq, on = ['species', 'locality_code', 'region_code', 'area_code'], how = 'left')

master.head()
ss = StandardScaler()
master[['height', 'diameter']] = ss.fit_transform(master[['height', 'diameter']])

master = master.drop(['area_code', 'region_code', 'locality_code', 'species'], axis = 1)
master.head()
train_data = master.loc[(master['type'] == 'train')]
test_data = master.loc[(master['type'] == 'test')]

train_data = train_data.sort_values(by = ['class'])

train_data = train_data.drop(['type'], axis = 1)
test_data = test_data.drop(['type', 'class'], axis = 1)
# Partitioning the features and the target

#X = train[train.columns[~(train.columns.isin(['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7']))].tolist()]

X = train_data[train_data.columns[~(train_data.columns.isin(['class']))].tolist()]
y = train_data[['class']]
pca = PCA()
pca.fit_transform(X)
pca.get_covariance()
explained_variance = pca.explained_variance_ratio_
explained_variance
with plt.style.context('seaborn-darkgrid'):
    plt.figure(figsize=(10, 8))

    plt.bar(range(17), explained_variance, alpha = 0.5, align = 'center', label = 'individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc = 'best')
    plt.tight_layout()
X = X.values
y = y.values
kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()
for train, test in kfold.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    
    model = XGBClassifier(random_state = 22, max_depth = 5, n_estimators = 200, objective = 'reg:squaredlogerror')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average = 'weighted')
    scores.append(score)
    print('Validation F1Score:', score)
print("Average Validation F1Score: ", sum(scores)/len(scores))
yPreds = model.predict(test_data.values)
yPred_Probs = model.predict_proba(test_data.values)
pd.set_option('display.float_format', lambda x: '%.20f' % x)

submission = pd.DataFrame(yPred_Probs)
submission.columns = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7']
submission.to_csv('flower_class_3.csv', index = False)
submission.head()