import pandas as pd
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
print (train.shape)
print (test.shape)
test=test.append(train[(train.ID>200000)])
train=train[~(train.ID>200000)]
print (train.shape)
print (test.shape)

train.to_csv('../input/train_nish.csv',index=False)
del test['target']
test.to_csv('../input/test_nish.csv',index=False)
train.head()
! ls ../input
! pwd
