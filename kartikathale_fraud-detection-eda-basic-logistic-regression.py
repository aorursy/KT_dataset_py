# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn import preprocessing

import warnings

warnings.simplefilter('ignore')

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.listdir('../input/ieee-fraud-detection')
train_iden_df = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_trans_df = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

test_trans_df = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

test_iden_df = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
# shape, this gives the dimensions of the dataset

print('Dimensions of the Train Identity set:',train_iden_df.shape)

print('Dimensions of the Train transaction set:',train_trans_df.shape)

print('Dimensions of the Test transaction set:',test_trans_df.shape)

print('Dimensions of the Test Identity set:',test_iden_df.shape)
train_iden_df.head()
train_trans_df.head()
# Checkign if all Transactions IDs from transaction dataset are in identity dataset

print(np.sum(train_trans_df['TransactionID'].isin(train_iden_df['TransactionID'].unique())))

print(np.sum(test_trans_df['TransactionID'].isin(test_iden_df['TransactionID'].unique())))
# Merge both the transaction and identity by left

train_df=pd.merge(train_trans_df,train_iden_df,how="left",on="TransactionID")

test_df=pd.merge(test_trans_df,test_iden_df,how="left",on="TransactionID")
# Print Shapes

print("Train Dataset shape: ", train_df.shape)

print("Test Dataset shape: ", test_df.shape)
train_df.head()
test_df.head()
# This Step is done to reduce memory by conversion

# convert it to the low memory to fit the RAM

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
# Using  above function to reduce memory usage for Train test df

train_df=reduce_mem_usage(train_df)

test_df=reduce_mem_usage(test_df)
# delete the 4 variables in order to reduce the memory issue

del train_iden_df

del train_trans_df

del test_trans_df

del test_iden_df
# Explore Categorical features

print('Training set:')

for col_name in train_df.columns:

    if train_df[col_name].dtypes == 'object' :

        unique_cat = len(train_df[col_name].unique())

        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))



print("****************")

    

print('Test set:')

for col_name in test_df.columns:

    if test_df[col_name].dtypes == 'object' :

        unique_cat = len(test_df[col_name].unique())

        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

train_df.describe().transpose()
cat_cols = [col for col in train_df.columns if train_df[col].dtype in ['object']]

train_df[cat_cols].describe()
# Visualize target variable.(isFraud)

sns.countplot(train_df['isFraud'])

print("Count Plot for Fraudulent vs legitimate transaction") ## Imbalanced Dataset

train_df['isFraud'].value_counts(normalize=True) * 100
perc_fraud = len(train_df[train_df['isFraud']==1].index)*100/train_df.shape[0]

print("Percentage of Fradaulent records in dataset {:.2f}".format(perc_fraud) )
# Check missing data - Many Columns have more than 50% NA/Null records

def missing_data(df) :

    count = df.isnull().sum()

    percent = (df.isnull().sum()) / (df.isnull().count()) * 100

    total = pd.concat([count, percent], axis=1, keys = ['Count', 'Percent'])

    types = []

    for col in df.columns :

        dtypes = str(df[col].dtype)

        types.append(dtypes)

    total['dtypes'] = types

    

    return np.transpose(total)



missing_data(train_df)
## WE will Drop columns with more 50% Null value.

null_percent = train_df.isnull().sum()/train_df.shape[0]*100



cols_to_drop = np.array(null_percent[null_percent > 50].index)



cols_to_drop
# Drop Columns (cols_to_drop) from train and test

train_df = train_df.drop(cols_to_drop, axis=1)

test_df = test_df.drop(cols_to_drop,axis=1)
print(train_df.shape)

print(test_df.shape)
### Fill NaNs



train_df = train_df.fillna(-999)

test_df = test_df.fillna(-999)
train_y = train_df['isFraud']

train_X = train_df.drop('isFraud', axis=1)
# Label Encoding for categorical variables.

for f in train_X.columns:

    if train_X[f].dtype=='object' or test_df[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_X[f].values) + list(test_df[f].values))

        train_X[f] = lbl.transform(list(train_X[f].values))

        test_df[f] = lbl.transform(list(test_df[f].values))
## Create Base Model - LogisticRegression



logreg = LogisticRegression()

logreg.fit(train_X, train_y)
submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv',index_col='TransactionID')

submission['isFraud'] = logreg.predict_proba(test_df)[:,1]

submission.to_csv('Logreg_submission.csv')

submission.head()