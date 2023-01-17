import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,gc

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import eli5

import base64

from IPython.display import HTML

from tqdm import tqdm



from eli5.sklearn import PermutationImportance

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



# train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

# test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')



sample = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
print(train_transaction.shape,test_transaction.shape)

# print(train_id.shape, test_id.shape)

print(sample.shape)
train_transaction.head(10)
train_transaction.info(max_cols=400)
columns_to_drop = []

num_of_rows = train_transaction.shape[0]

for i in train_transaction.columns:

    count_of_null_values = train_transaction[i].isna().sum()

    if (count_of_null_values >= num_of_rows/2):

        columns_to_drop.append(i)

del num_of_rows        
# Dropping columns with more than 50% missing values.

train_transaction.drop(columns_to_drop, axis=1, inplace=True)

test_transaction.drop(columns_to_drop, axis=1, inplace=True)



print("No of columns dropped {}".format(len(columns_to_drop)))

del columns_to_drop

gc.collect()
object_columns = train_transaction.select_dtypes(include=object).columns

print("Number of categorical columns: {}".format(len(object_columns)))
for i in object_columns:

    print("Column Name : {}".format(i))

    print("-------------> No of missing values: {}".format(train_transaction[i].isna().sum()))

    print("-------------> Unique values: {}".format(train_transaction[i].unique()))
fig,ax = plt.subplots(3,3,figsize=(18,18))

for k,i in enumerate(object_columns):

    plt.subplot(3,3,k+1)

    if(i != 'P_emaildomain'):

        train_transaction[i].value_counts().plot(kind='bar')

    else:

        prob = train_transaction[i].value_counts(normalize=True)

        threshold = 0.02

        mask = prob > threshold

        tail_prob = prob.loc[~mask].sum()

        prob = prob.loc[mask]

        prob['other'] = tail_prob

        prob.plot(kind='bar')

    plt.title(i)
fig,ax = plt.subplots(3,3,figsize=(18,18))

for k,i in enumerate(object_columns):

    plt.subplot(3,3,k+1)

    if(i != 'P_emaildomain'):

        test_transaction[i].value_counts().plot(kind='bar')

    else:

        prob = test_transaction[i].value_counts(normalize=True)

        threshold = 0.02

        mask = prob > threshold

        tail_prob = prob.loc[~mask].sum()

        prob = prob.loc[mask]

        prob['other'] = tail_prob

        prob.plot(kind='bar')

    plt.title(i)
for i in object_columns:

    train_transaction[i].fillna(train_transaction[i].mode()[0], inplace=True)

    test_transaction[i].fillna(test_transaction[i].mode()[0], inplace=True)
# Categorical Features:

# ProductCD

# card1 - card6

# addr1, addr2

# Pemaildomain Remaildomain

# M1 - M9



# We handles few M* features above, others were dropped because of >50% missing values.

cat_num_features = ['addr1','addr2', 'card1', 'card2', 'card3', 'card5']

for i in cat_num_features:

    print("Column Name : {}".format(i))

    print("-------------> No of missing values: {}".format(train_transaction[i].isna().sum()))

    print("Mode value {} occurred in {} transactions \n".format(train_transaction[i].mode()[0], train_transaction[i].value_counts().values[0]))
# Filling the missing values with mode.

for i in cat_num_features:

    train_transaction[i].fillna(train_transaction[i].mode()[0], inplace=True)

    test_transaction[i].fillna(test_transaction[i].mode()[0], inplace=True)

del cat_num_features

gc.collect()
all_numeric_columns = train_transaction.select_dtypes(include=np.number).columns

numeric_missing = []

for i in all_numeric_columns:

    missing = train_transaction[i].isna().sum()

    if(missing>0):

        numeric_missing.append(i)

del all_numeric_columns        

print(len(numeric_missing))
train_transaction[numeric_missing].describe()
for k,i in enumerate(numeric_missing):

    print(k)

    print("Column {} has {} missing values".format(i, train_transaction[i].isna().sum()))

    print("Mode value {} occurred in {} transactions".format(train_transaction[i].mode()[0], train_transaction[i].value_counts().values[0]))

    print("Median value {} \n".format(train_transaction[i].median(), train_transaction[i].value_counts().values[0]))
# Filling the missing values with median.

for i in numeric_missing:

    train_transaction[i].fillna(train_transaction[i].median(), inplace=True)

    test_transaction[i].fillna(test_transaction[i].median(), inplace=True)

print(train_transaction.isna().any().sum(), test_transaction.isna().any().sum())   

del numeric_missing

gc.collect()
# object_columns

for f in object_columns:

    lbl = LabelEncoder()

    lbl.fit(list(train_transaction[f].values) + list(test_transaction[f].values))

    train_transaction[f] = lbl.transform(list(train_transaction[f].values))

    test_transaction[f] = lbl.transform(list(test_transaction[f].values))    
train_transaction[object_columns].head()
len(train_transaction.select_dtypes(exclude=np.number).sum())

del object_columns

gc.collect()
train_transaction.head()
train_transaction.describe()
# Let's plot the histogram of isFraud column.

train_transaction['isFraud'].plot(kind='hist')
X_train,X_val,y_train,y_val = train_test_split(train_transaction.drop(['isFraud'],axis=1), train_transaction['isFraud'], test_size=0.2)
# Score= .7427

params = {

    'objective': 'binary',

    'n_estimators':300,

    'learning_rate': 0.1,

    'subsample':0.8

}

# Score= .7306

params1 = {

    'objective': 'binary',

    'n_estimators': 200,

    'learning_rate': 0.1,

}

#Score= .7446

params2 = {

    'objective': 'binary',

    'n_estimators':300,

    'learning_rate': 0.1,

}

# Score=.774

params3 = {

    'objective': 'binary',

    'n_estimators':600,

    'learning_rate': 0.1

}

#Score= .7666

params4 = {

    'objective': 'binary',

    'n_estimators':500,

    'learning_rate': 0.1

}

#Score= .7711

params5 = {

    'objective': 'binary',

    'n_estimators':500,

    'learning_rate': 0.1,

    'num_leaves' : 50,

    'max_depth' : 7,

    'subsample' : 0.9,

    'colsample_bytree' : 0.9

}

#Score=.78109

params6 = {

    'objective': 'binary',

    'n_estimators':600,

    'learning_rate': 0.1,

    'num_leaves' : 50,

    'max_depth' : 7,

    'subsample' : 0.9,

    'colsample_bytree' : 0.9

}

#Score=.7863

params7 = {

    'objective': 'binary',

    'n_estimators':700,

    'learning_rate': 0.1,

    'num_leaves' : 50,

    'max_depth' : 7,

    'subsample' : 0.9,

    'colsample_bytree' : 0.9

}
clf = LGBMClassifier(**params7, random_state=108)

clf.fit(X_train,y_train)
preds = clf.predict(X_val)

roc_auc_score(y_val, preds)
predictions = clf.predict_proba(test_transaction)

sample['isFraud'] = predictions[:,1]

sample.to_csv('submission.csv', index=False)
# def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

#     csv = df.to_csv(index=False)

#     b64 = base64.b64encode(csv.encode())

#     payload = b64.decode()

#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

#     html = html.format(payload=payload,title=title,filename=filename)

#     return HTML(html)

# create_download_link(sample)