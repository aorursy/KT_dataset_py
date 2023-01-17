import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime, gc



print('loading files and re-shaping...')

train = pd.read_csv('../input/train_2.csv') 

train.fillna(0, inplace=True)

test = pd.melt(train[list(train.columns[-62:])+['Page']], id_vars='Page', var_name='D62', value_name='V62')

test['D62']= test['D62'].astype('datetime64[ns]')

test['Date']=test['D62']+ datetime.timedelta(days=74) # to match the test set starting on 13 Sep

test['D62']= test['D62'].dt.dayofweek  >= 5      # take in account weekly seasonality

    

Windows = [7,28,49,63,140]    # windows to take the median over (just a few due to time limit)

n=11   # chose the number of CV sets (up to 12 depending on max Windows size)

cvlist=list(range(62,n*62,62))



for i in range(2,n):

    tmp = pd.melt(train[list(train.columns[-62*i:-62*(i-1)])+['Page']], 

                id_vars='Page', var_name='D'+str(i*62), value_name='V'+str(i*62) )

    tmp.drop(['Page'],axis=1,inplace=True)

    tmp['D'+str(i*62)]= tmp['D'+str(i*62)].astype('datetime64[ns]').dt.dayofweek  >= 5

    test = pd.concat([test, tmp], axis=1, join_axes=[test.index])

del tmp                      

gc.collect()

def smape(t,p):  return 200.* ((t-p).abs()/(p.abs()+t.abs()).replace({0:1})).mean()

print(test.shape)

test.head(2)

# collect smape values with Windows in rows and CV sets in columns

mw=[]

for i in Windows: mw=mw+['MW'+str(i)]

colmw=[]

for cv in cvlist: colmw=colmw+['MW_'+str(cv)]

smapeMW=pd.DataFrame(0,index=mw, columns=colmw)



for cv in cvlist:

    Day='D'+str(cv)

    for i in Windows:

        print(cv,i, end=' ')

        tmp = pd.melt(train[list(train.columns[-i-cv:-cv])+['Page']], 

                  id_vars='Page', var_name=Day, value_name='MW'+str(i))

        tmp[Day]= tmp[Day].astype('datetime64[ns]').dt.dayofweek  >= 5

        tmp1 = tmp.groupby(['Page',Day]).median().reset_index()

        test = test.merge(tmp1, how='left')

    print(test.shape)

    del tmp,tmp1

    gc.collect()

    

    for i in range(0,len(mw)) :    

        smapeMW.loc[mw[i],'MW_'+str(cv)] = smape(test['V'+str(cv)],test[mw[i]])

    test.drop(mw,axis=1,inplace=True)
plt.clf()

fig = plt.figure()

smapeMW.loc[:,colmw].T.plot(kind='box',ylim=(35,55), figsize=(10,6))

plt.suptitle('WebTraffic_II_CV Medians SMAPE', size=12)

plt.savefig('WebTraffic_II_CV Medians.jpg')

plt.show()
smapeMW['std']=smapeMW.loc[:,colmw].std(axis=1)

smapeMW['mean']=smapeMW.loc[:,colmw].mean(axis=1)

smapeMW
plt.clf()

smapeMW.loc['MW28',colmw].plot(ylim=(35,55), figsize=(10,6))

plt.show()
plt.clf()

smapeMW.loc[:,colmw].plot(ylim=(30,55), figsize=(10,6))

plt.show()