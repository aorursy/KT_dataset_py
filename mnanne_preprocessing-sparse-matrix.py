import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy.sparse import hstack
train = pd.read_csv("../input/train.csv",index_col='id')

test = pd.read_csv("../input/test.csv", index_col='id')

train.shape

test.shape
y_train = train['loss']

train = train.drop('loss', axis=1)



numerical_train = train._get_numeric_data()

cats_train = train.select_dtypes(include=[object])



numerical_test=  test._get_numeric_data()

cats_test = test.select_dtypes(include=[object])
for column in cats_train.columns:

    le = LabelEncoder()

    cats_train.loc[:,column] = le.fit_transform(cats_train[column])

    cats_test.loc[:,column] = le.fit_transform(cats_test[column])
ohc = OneHotEncoder()

ohc = ohc.fit(pd.concat((cats_train,cats_test),axis=0))

cats_train_onehot = ohc.transform(cats_train)

cats_test_onehot = ohc.transform(cats_test)
print(cats_train_onehot.shape)

print(cats_test_onehot.shape)
train_sparse = hstack([cats_train_onehot, numerical_train], 'csr')

test_sparse = hstack([cats_test_onehot,numerical_test], 'csr')
print(cats_train_onehot.shape)

print(numerical_train.shape)

print('train sparse', train_sparse.shape)



print(cats_test_onehot.shape)

print(numerical_test.shape)

print('test sparse' , test_sparse.shape)



print('y_train', y_train.shape)


print(train_sparse[:1,:].toarray())