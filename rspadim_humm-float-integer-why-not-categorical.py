#reading data :)

import numpy as np

import pandas as pd

train=pd.read_csv('../input/train.csv')

test =pd.read_csv('../input/test.csv')

test['target']=-1 # just to match columns

both=test.copy()

both=both.append(train.copy())

del test  #bye!

del train #bye!

cols=both.columns.drop(['id','target']).tolist()

print('columns: ',both.columns.tolist())

print('target values: ',both['target'].unique())
cats=[]

cats_prefix={}

for i in cols:

    unique_train=both[both['target']!=-1][i].unique()

    unique_both =both[i].unique()

    equal=(sorted(unique_train) == sorted(unique_both))

    length_train=len(unique_train)

    length_both=len(unique_both)

    print('Column: ',i,'\t unique values at train/both=',

          length_train,' / ',length_both,

          '\t <- categorical?!' if equal else ''

         )

    if(equal and length_both>2):

        cats.append(i)

        cats_prefix[i]="OHE_"+i

print("these variables should be categorical, or not?! =) ",cats)
#i will OHE to you :)

both=pd.get_dummies(both,prefix=cats_prefix,columns=cats)
both[both['target']!=-1].to_csv('train.cat.ohe.csv.gzip',index=False,compression='gzip')

both[both['target']==-1].to_csv('test.cat.ohe.csv.gzip' ,index=False,compression='gzip')
print(both.columns.tolist())