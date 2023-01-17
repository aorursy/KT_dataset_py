import numpy as np

import pandas as pd



train = pd.read_csv("../input/train.csv", nrows = 1000)

test = pd.read_csv("../input/test.csv", nrows = 1000)
test.head()
train_Y = train['Y']

train_Y.head()
train_Y = pd.DataFrame(train_Y)

train_Y
type(train_Y_list)
train_Y = train['Y']

train_Y_list = list(train_Y)

train_Y_list



train_Y_list = pd.DataFrame(train_Y_list)

train_Y_list.head()
type(train)
total = pd.concat([train,test],axis=0)

print(train.shape, test.shape, total.shape)

total.tail()
total = pd.concat([train,test],axis=1)

print(train.shape, test.shape, total.shape)

total.tail()
train_sort = train.sort_values('Y', ascending=False)

train_sort
train_rename = train.rename(columns = {'Y':'Wafer Thick'})

train_rename.head()
import numpy as np

import pandas as pd



train = pd.read_csv("../input/train.csv",index_col = ['ID'], nrows = 1000)

test = pd.read_csv("../input/test.csv",index_col = ['ID'], nrows = 1000)
test = test.reset_index()

test.head()
test = test.drop(['ID'],1)
test.head()
train = train.reset_index()

train = train.drop(['ID'],1)

train.head()
train_C5 = train[train['C5'].isnull()]

train_C5 = train_C5.reset_index()

train_C5_index = train_C5['index']

train_C5_index
train_C5_index = list(train_C5_index)

train = train.drop([train_C5_index],0)
train.head()
total = pd.concat([train,test], axis=1)

print(train.shape, test.shape, total.shape)
total.tail()
total.tail()
# typing your self!
train_10 = train[train['Y'] < 10]

train_10
print(train.shape)

train = train.drop_duplicates()

print(train.shape)
train_sample = train.sample(n=100)

print(train_sample.shape)

train_sample = train_sample.drop_duplicates()

print(train_sample.shape)
train_4 = train.iloc[:,0:4]

train_4
train.iloc[:,3:4]

train_S = train[['Y','C1','C2']]

train_S
train_Y = train.loc[train['Y']<1 , ['Y','A','B','C1']]

train_Y
len(train)

train['A'].unique()
train.describe()
train['Y'].max()
train['ABC'] = train['C1'] * train['C2'] * train['C3']

train[['C1','C2','C3','ABC']].head()
train['validation'] = 0

train.head()
test['validation'] = 1
test.head()
total = pd.concat([train, test],0)
total.tail()
# typing your self!
train['Y']
train_Y_qcut = pd.DataFrame(pd.qcut(train['Y'], 4))

train_Y_qcut
train['C1_Q'] = pd.DataFrame(pd.qcut(train['C1'], 4))

train.head()
train.groupby(['A','B','C1_Q'])['Y'].mean()
train['C1'].plot.hist()
train.plot.scatter(x='Y', y='C1')