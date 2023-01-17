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
import os

import pandas as pd

import numpy as np

import scipy as sp

import seaborn as sns

import math as m

from scipy import stats

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/avhranalytics/train_jqd04QH.csv')

test = pd.read_csv('/kaggle/input/avhranalytics/test_KaymcHn.csv')
train.head()
train.shape,test.shape
train.apply(lambda x: (len(x.unique()))) 
test.apply(lambda x: (len(x.unique()))) 
combine = train.append(test,sort=False)

combine.shape
combine.isnull().sum()
train['target'].value_counts(normalize=True)
sns.countplot(train['target'])
plt.figure(figsize=(24, 6))

plt.subplot(121)

sns.countplot(combine['company_size'],order = combine['company_size'].value_counts(dropna=False).index)

plt.subplot(122)

sns.countplot(combine['company_type'],order = combine['company_type'].value_counts(dropna=False).index)
combine['company_size'].fillna('unknown', inplace=True)
combine['company_type'].fillna('unknown', inplace=True)
plt.figure(figsize=(20, 6))

plt.subplot(121)

sns.countplot(combine['gender'],order = combine['gender'].value_counts(dropna=False).index)

plt.subplot(122)

sns.countplot(combine['relevent_experience'],order = combine['relevent_experience'].value_counts(dropna=False).index)
combine['gender'].fillna('Male', inplace=True)
combine["gender"] = combine["gender"].map({'Male':2,  'Female':1, 'Other':0})
plt.figure(figsize=(22, 6))

plt.subplot(121)

sns.countplot(combine['last_new_job'],order = combine['last_new_job'].value_counts(dropna=False).index)

plt.subplot(122)

sns.countplot(combine['experience'],order = combine['experience'].value_counts(dropna=False).index)
combine['last_new_job'].fillna('1', inplace=True) #using Mode Option for fill Nan Values as there are very less null values
combine['last_new_job'].replace('>4','6', inplace=True)

combine['last_new_job'].replace('never','0' ,inplace=True)

combine['last_new_job']=combine['last_new_job'].astype(int)
combine['experience'].fillna('>20', inplace=True) #using Mode Option for fill Nan Values as there are very less null values
combine['experience'].replace('>20','25', inplace=True)

combine['experience'].replace('<1','0' ,inplace=True)

combine['experience']=combine['last_new_job'].astype(int)
plt.figure(figsize=(22, 6))

plt.subplot(131)

sns.countplot(combine['education_level'],order = combine['education_level'].value_counts(dropna=False).index)

plt.subplot(132)

sns.countplot(combine['enrolled_university'],order = combine['enrolled_university'].value_counts(dropna=False).index)

plt.subplot(133)

sns.countplot(combine['major_discipline'],order = combine['major_discipline'].value_counts(dropna=False).index)
combine['education_level'].value_counts(dropna=False)
combine['education_level'].fillna(train['education_level'].mode()[0], inplace=True) #Missing Value is less than 10% of Mode Value Itself
plt.figure(figsize=(22, 6))

city_tier_counts = (combine.groupby(['target'])['education_level'].value_counts(dropna=False,normalize=True).rename('percentage').mul(100).reset_index().sort_values('target'))

sns.barplot(x="education_level", y="percentage", hue="target", data=city_tier_counts)
combine["education_level"] = combine["education_level"].map({'Graduate':0, 'Masters':1, 'High School':2, 'Phd':3, 'Primary School':4})
combine.enrolled_university.value_counts(dropna=False)
combine['enrolled_university'].fillna(train['enrolled_university'].mode()[0], inplace=True) #Missing Value is less than 3% of Mode Value Itself
plt.figure(figsize=(22, 6))

city_tier_counts = (combine.groupby(['target'])['enrolled_university'].value_counts(dropna=False,normalize=True).rename('percentage').mul(100).reset_index().sort_values('target'))

sns.barplot(x="enrolled_university", y="percentage", hue="target", data=city_tier_counts)
combine["enrolled_university"] = combine["enrolled_university"].map({'no_enrollment':1, 'Full time course':4, 'Part time course':2})
combine.major_discipline.value_counts(dropna=False)
combine['major_discipline'].fillna(train['major_discipline'].mode()[0], inplace=True)
plt.figure(figsize=(20, 6))

city_tier_counts = (combine.groupby(['target'])['major_discipline'].value_counts(dropna=False,normalize=True).rename('percentage').mul(100).reset_index().sort_values('target'))

sns.barplot(x="major_discipline", y="percentage", hue="target", data=city_tier_counts)
combine.isnull().sum()
cat_col = combine.dtypes.loc[combine.dtypes=='object'].index

categorical_variables=cat_col.tolist()

categorical_variables
from sklearn import metrics, preprocessing, model_selection

for col in categorical_variables:

    print(col)

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(combine[col].values.astype('str')))

    combine[col] = lbl.transform(list(combine[col].values.astype('str')))
display(combine.columns),train.shape
train_features = combine.drop(['enrollee_id', 'target'], axis = 1)[:18359]

target = combine['target'][:18359]

test_features = combine.drop(['enrollee_id','target'], axis = 1)[18359:]
train_features.shape,target.shape,test_features.shape
from sklearn import metrics, preprocessing, model_selection

import lightgbm as lgb
train_X=train_features

train_y=target

test_X=test_features
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, dep=8, seed=0, data_leaf=200):

    params = {}

    params["objective"] = "binary"

    params['metric'] = 'auc'

    params["max_depth"] = dep

    params["num_leaves"] = 31

    params["min_data_in_leaf"] = data_leaf

    params["learning_rate"] = 0.01

    params["bagging_fraction"] = 0.9

    params["feature_fraction"] = 0.5

    params["feature_fraction_seed"] = seed

    params["bagging_freq"] = 1

    params["bagging_seed"] = seed

    params["lambda_l2"] =5

    params["lambda_l1"] = 5

    params["verbosity"] = -1

    num_rounds = 25000



    plst = list(params.items())

    lgtrain = lgb.Dataset(train_X, label=train_y)



    if test_y is not None:

        lgtest = lgb.Dataset(test_X, label=test_y)

        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=200, verbose_eval=500)

    else:

        lgtest = lgb.DMatrix(test_X)

        model = lgb.train(params, lgtrain, num_rounds)



    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)



    loss = 0

    if test_y is not None:

        loss = metrics.roc_auc_score(test_y, pred_test_y)

        print(loss)

        return model, loss, pred_test_y, pred_test_y2

    else:

        return model, loss, pred_test_y, pred_test_y2
print("Building model..")

cv_scores = []

pred_test_full = 0

pred_train = np.zeros(train_X.shape[0])

n_splits = 5

#kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=7988)

gkf = model_selection.GroupKFold(n_splits=n_splits)

model_name = "lgb"

for dev_index, val_index in gkf.split(train_X, combine['target'][:18359].values, combine['enrollee_id'][:18359].values):

    dev_X, val_X = train_X.iloc[dev_index,:], train_X.iloc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val = 0

    pred_test = 0

    n_models = 0.



    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=8, seed=2019)

    pred_val += pred_v

    pred_test += pred_t

    n_models += 1

    

    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=7, data_leaf=100, seed=9873)

    pred_val += pred_v

    pred_test += pred_t

    n_models += 1

    

    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=9, data_leaf=150, seed=4568)

    pred_val += pred_v

    pred_test += pred_t

    n_models += 1

    

    pred_val /= n_models

    pred_test /= n_models

    

    loss = metrics.roc_auc_score(val_y, pred_val)

        

    pred_train[val_index] = pred_val

    pred_test_full += pred_test / n_splits

    cv_scores.append(loss)

#     break

print(np.mean(cv_scores))
fig, ax = plt.subplots(figsize=(10,10))

lgb.plot_importance(model, max_num_features=100, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
sample = pd.read_csv('/kaggle/input/avhranalytics/sample_submission_sxfcbdx.csv')

sample["target"] = pred_test_full

sample.to_csv("Solution.csv", index=False)