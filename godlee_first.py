import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train= pd.read_csv("../input/test.csv", index_col=0)
train.head()
train_final=train[['hacdor','refrig','sanitario1','sanitario2','sanitario3','sanitario5',
                   'energcocinar1','energcocinar2','energcocinar3','energcocinar4','elimbasu1','elimbasu2',
                   'elimbasu3','elimbasu4','elimbasu5','epared1','epared2','epared3','etecho1',
                   'etecho2','etecho3','eviv1','eviv2','eviv3']]
train_final.head()

#train_final.loc[:,'hacdor'][train_final['hacdor']==0]=1
train_final.loc[:,'hacdor'].replace(0,np.nan,inplace =True)
train_final[['hacdor']].head()


train_final.loc[:,'refrig'][train_final['refrig']==1]=3
train_final.loc[:,'hacdor'][train_final['hacdor']==0]=1
train_final[['refrig']].head()
#train_final.loc[:,'sanitario1'][train_final['sanitario1']==1]=1
train_final.loc[:,'sanitario1'].replace(0,np.nan,inplace =True)
train_final[['sanitario1']].head()
train_final.loc[:,'sanitario2'][train_final['sanitario2']==1]=2
train_final.loc[:,'sanitario2'].replace(0,np.nan,inplace =True)
train_final[['sanitario2']].head()
train_final.loc[:,'sanitario3'][train_final['sanitario3']==1]=3
train_final.loc[:,'sanitario3'].replace(0,np.nan,inplace =True)
train_final[['sanitario3']].head()
train_final.loc[:,'sanitario5'][train_final['sanitario5']==1]=3
train_final.loc[:,'sanitario5'].replace(0,np.nan,inplace =True)
train_final[['sanitario5']].head()
#train_final.loc[:,'energcocinar1'][train_final['energcocinar1']==1]=3
train_final.loc[:,'energcocinar1'].replace(0,np.nan,inplace =True)
train_final[['energcocinar1']].head()
train_final.loc[:,'energcocinar2'][train_final['energcocinar2']==1]=4
train_final.loc[:,'energcocinar2'].replace(0,np.nan,inplace =True)
train_final[['energcocinar2']].head()
train_final.loc[:,'energcocinar3'][train_final['energcocinar3']==1]=4
train_final.loc[:,'energcocinar3'].replace(0,np.nan,inplace =True)
train_final[['energcocinar3']].head()
train_final.loc[:,'energcocinar4'][train_final['energcocinar4']==1]=2
train_final.loc[:,'energcocinar4'].replace(0,np.nan,inplace =True)
train_final[['energcocinar4']].head()
train_final.loc[:,'elimbasu1'][train_final['elimbasu1']==1]=4
train_final.loc[:,'elimbasu1'].replace(0,np.nan,inplace =True)
train_final[['elimbasu1']].head()
train_final.loc[:,'elimbasu2'][train_final['elimbasu2']==1]=3
train_final.loc[:,'elimbasu2'].replace(0,np.nan,inplace =True)
train_final[['elimbasu2']].head()
train_final.loc[:,'elimbasu3'][train_final['elimbasu3']==1]=2
train_final.loc[:,'elimbasu3'].replace(0,np.nan,inplace =True)
train_final[['elimbasu3']].head()
#train_final.loc[:,'elimbasu4'][train_final['elimbasu4']==1]=2
train_final.loc[:,'elimbasu4'].replace(0,np.nan,inplace =True)
train_final[['elimbasu4']].head()
#train_final.loc[:,'elimbasu4'][train_final['elimbasu4']==1]=2
train_final.loc[:,'elimbasu5'].replace(0,np.nan,inplace =True)
train_final[['elimbasu5']].head()
#train_final.loc[:,'elimbasu4'][train_final['elimbasu4']==1]=2
train_final.loc[:,'epared1'].replace(0,np.nan,inplace =True)
train_final[['epared1']].head()
train_final.loc[:,'epared2'][train_final['epared2']==1]=2
train_final.loc[:,'epared2'].replace(0,np.nan,inplace =True)
train_final[['epared2']].head()
train_final.loc[:,'epared3'][train_final['epared3']==1]=4
train_final.loc[:,'epared3'].replace(0,np.nan,inplace =True)
train_final[['epared3']].head()
#train_final.loc[:,'etecho1'][train_final['etecho1']==1]=4
train_final.loc[:,'etecho1'].replace(0,np.nan,inplace =True)
train_final[['etecho1']].head()
train_final.loc[:,'etecho2'][train_final['etecho2']==1]=2
train_final.loc[:,'etecho2'].replace(0,np.nan,inplace =True)
train_final[['etecho2']].head()
train_final.loc[:,'etecho3'][train_final['etecho3']==1]=4
train_final.loc[:,'etecho3'].replace(0,np.nan,inplace =True)
train_final[['etecho3']].head()
#train_final.loc[:,'eviv1'][train_final['eviv1']==1]=4
train_final.loc[:,'eviv1'].replace(0,np.nan,inplace =True)
train_final[['eviv1']].head()
train_final.loc[:,'eviv2'][train_final['eviv2']==1]=2
train_final.loc[:,'eviv2'].replace(0,np.nan,inplace =True)
train_final[['eviv2']].head()
train_final.loc[:,'eviv3'][train_final['eviv3']==1]=4
train_final.loc[:,'eviv3'].replace(0,np.nan,inplace =True)
train_final[['eviv3']].head()
train_final.head()
train_final['target'] = ''
train_final.head()
train_final[['target']]=train_final.mean(1)
rate=train_final[['target']]
import numpy
rate= rate+0.5
rate = rate.apply(numpy.round)
rate.head()

a=rate.reset_index()
a[['target']]=a[['target']].astype(int)


a.to_csv('first submit.csv', index = False)

