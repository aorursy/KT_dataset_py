import numpy as np

import pandas as pd

print('Reading files')

train  =pd.read_csv("../input/train.csv")

test   =pd.read_csv("../input/test.csv")

print('Cast')

for c in train.select_dtypes(include=['float64']).columns:

    train[c]=train[c].astype(np.float32)

    test[c]=test[c].astype(np.float32)

for c in train.select_dtypes(include=['int64']).columns[2:]:

    train[c]=train[c].astype(np.int8)

    test[c]=test[c].astype(np.int8)    

print('ok')
print('Merging')

test['target']=-1

both =test.copy()

del test

both=both.append(train)

del train

print('OK')
#OHE , Factors

import gc

print('Transform')

cat_cols = [col for col in both.columns if '_cat' in col]

bin_cols = [col for col in both.columns if '_bin' in col]



c2,c3=[],{}

print('\tBinaries, len=',len(bin_cols))

for c in bin_cols:

    if(len(both[c].unique())>2):

        c2.append(c)

        c3[c]='ohe_'+c

print('\tCategories, len=',len(cat_cols))



for c in cat_cols:

    if(len(both[c].unique())>2):

        c2.append(c)

        c3[c]='ohe_'+c

        

print(c2)

both=pd.get_dummies(both,prefix=c3,columns=c2)

gc.collect()



train=both[both['target']!=-1].copy()

test =both[both['target']==-1].copy()

del both

    

print(train.columns)

col_x= train.columns.drop(['target'])

col  = train.columns.drop(['id','target'])



print(col)

print('OK')
print('Saving OHE files')

train.to_csv("ohe_train.csv",index=False)

test.to_csv("ohe_test.csv",index=False)

print('Done')