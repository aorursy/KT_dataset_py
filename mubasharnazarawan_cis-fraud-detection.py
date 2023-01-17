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
# credit to @guiferviz for the memory reduction 

def memory_usage_mb(df, *args, **kwargs):

    """Dataframe memory usage in MB. """

    return df.memory_usage(*args, **kwargs).sum() / 1024**2



def reduce_memory_usage(df, deep=True, verbose=True):

    # All types that we want to change for "lighter" ones.

    # int8 and float16 are not include because we cannot reduce

    # those data types.

    # float32 is not include because float16 has too low precision.

    numeric2reduce = ["int16", "int32", "int64", "float64"]

    start_mem = 0

    if verbose:

        start_mem = memory_usage_mb(df, deep=deep)



    for col, col_type in df.dtypes.iteritems():

        best_type = None

        if col_type in numeric2reduce:

            downcast = "integer" if "int" in str(col_type) else "float"

            df[col] = pd.to_numeric(df[col], downcast=downcast)

            best_type = df[col].dtype.name

        # Log the conversion performed.

        if verbose and best_type is not None and best_type != str(col_type):

            print(f"Column '{col}' converted from {col_type} to {best_type}")



    if verbose:

        end_mem = memory_usage_mb(df, deep=deep)

        diff_mem = start_mem - end_mem

        percent_mem = 100 * diff_mem / start_mem

        print(f"Memory usage decreased from"

              f" {start_mem:.2f}MB to {end_mem:.2f}MB"

              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")

        

    return df
train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

# train_full=train_transaction
# categorical = train_full[train_full.select_dtypes(include=['object']).columns]

# train_full = train_full.drop(train_full.select_dtypes(include=['object']).columns, axis=1)

# train_full.shape
# corr = train_full.corr()
train_full = train_identity.merge(train_transaction, how='outer')

# train_full = train_full.fillna(0)

del train_identity

del train_transaction
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

# test_full = test_transaction

test_full = test_identity.merge(test_transaction, how='outer')

# test_full = test_full.fillna(0)

del test_identity

del test_transaction
train_full.shape, test_full.shape
labels = train_full["isFraud"]

train_full = train_full.drop(["isFraud"], axis=1)
# cor = train_full.corr()
# cor[(cor<0.5) & (cor>-0.5)]
categorical = train_full[train_full.select_dtypes(include=['object']).columns]

one_hot_train = pd.get_dummies(categorical)
categorical = test_full[test_full.select_dtypes(include=['object']).columns]

one_hot_test = pd.get_dummies(categorical)
del categorical

one_hot_train.shape, one_hot_test.shape
train_full = train_full.drop(train_full.select_dtypes(include=['object']).columns, axis=1)

test_full = test_full.drop(test_full.select_dtypes(include=['object']).columns, axis=1)
# cols_nan = train_full.isna().sum()

# columns =  cols_nan[((cols_nan/train_full.shape[0])<0.50)]

# train_full = train_full[columns.index]

# test_full = test_full[columns.index]



train_full = train_full.iloc[:,0:200]

test_full = test_full.iloc[:,0:200]

# train_full = train_full

# test_full = test_full
train_full = train_full.fillna(-1000)

test_full = test_full.fillna(-1000)
# from sklearn import preprocessing

# # normalized_X = preprocessing.normalize(train_full)

# # normalized_df=(train_full-train_full.mean())/train_full.std()

ID = test_full["TransactionID"]

# train_full = (train_full-train_full.min())/(train_full.max()-train_full.min())

# test_full = (test_full-test_full.min())/(test_full.max()-test_full.min())
train_full = reduce_memory_usage(train_full, deep=True, verbose=True)

test_full = reduce_memory_usage(test_full, deep=True, verbose=True)
one_hot_train, one_hot_test = one_hot_train.align(one_hot_test, join='inner', axis=1)

test_full = test_full.drop(test_full.select_dtypes(include=['object']).columns, axis=1)

print (test_full.shape)

test_full = pd.concat([test_full, one_hot_test], axis=1)

print (test_full.shape)

train_full = pd.concat([train_full, one_hot_train], axis=1)

del one_hot_train

del one_hot_test
# train_full = train_full.drop(train_full.select_dtypes(include=['object']).columns, axis=1)

# train_full = pd.concat([train_full, one_hot_train], axis=1)
train_full.shape, test_full.shape
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500, random_state=1)



clf.fit(train_full[:100000], labels[:100000])

print (1)

clf.fit(train_full[100001:200000], labels[100001:200000])

print (2)

clf.fit(train_full[200001:300000], labels[200001:300000])

print (3)

clf.fit(train_full[300001:400000], labels[300001:400000])

print (4)

clf.fit(train_full[400001:500000], labels[400001:500000])

print (5)

clf.fit(train_full[500001:], labels[500001:])

print (6)
del train_full

del labels
# test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

# test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")



# test_full = test_identity.merge(test_transaction, how='outer')

# test_full = test_full.fillna(0)

# del test_identity

# del test_transaction
# categorical = test_full[test_full.select_dtypes(include=['object']).columns]

# one_hot_test = pd.get_dummies(categorical)

# print (one_hot_test.shape)

# one_hot_train, one_hot_test = one_hot_train.align(one_hot_test, join='inner', axis=1)

# test_full = test_full.drop(test_full.select_dtypes(include=['object']).columns, axis=1)

# print (test_full.shape)

# test_full = pd.concat([test_full, one_hot_test], axis=1)

# print (test_full.shape)

# # test_full = test_full.fillna(0)
# test_full.isna().sum()>0

test_full = test_full.fillna(-1000)
a = test_full[:100000]

b = test_full[100000:200000]

c = test_full[200000:300000]

d = test_full[300000:400000]

e = test_full[400000:500000]

f = test_full[500000:]

# del test_full
a = clf.predict_proba(a)
b = clf.predict_proba(b)
c = clf.predict_proba(c)
d = clf.predict_proba(d)
e = clf.predict_proba(e)
f = clf.predict_proba(f)
# a = clf.predict_proba(test_full[:100000])

# b = clf.predict_proba(test_full[100000:200000])

# c = clf.predict_proba(test_full[20000:300000])

# d = clf.predict_proba(test_full[30000:400000])

# e = clf.predict_proba(test_full[40000:500000])

# f = clf.predict_proba(test_full[50000:])

pred = np.concatenate([a,b,c,d,e,f])

del a

del b

del c

del d

del e

del f
# test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

# test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

# test_full = test_transaction

# test_full = test_identity.merge(test_transaction, how='outer')

# test_full = test_full.fillna(0)

# del test_identity

# del test_transaction
p = pred[:,1]

# pred = pd.DataFrame(test_full["TransactionID"])

pred = pd.DataFrame(ID)

pred["isFraud"] = p
pred.to_csv('submission.csv', index=False)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

#     print (HTML(html))

    

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(pred)