# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pc=pd.read_csv('../input/promoted_content.csv')

df_ct = pd.read_csv('../input/clicks_train.csv',nrows=1000000 )

print(df_ct.size)

print((pd.DataFrame(pc['document_id'].unique())).count())

print(pc.size)

M = df_ct.clicked.mean()

pc.groupby('document_id',as_index=False).count()['advertiser_id'].unique()



df_mrg=df_ct.merge(pc,on='ad_id' ,how='left')



df_cmpg= df_mrg.groupby('campaign_id').clicked.agg(['count' ,'sum']).reset_index()

df_cmpg['cmpg_Score']= (df_cmpg['sum'] + M) / (1 + df_cmpg['count'])

pd.DataFrame(df_cmpg).sort('cmpg_Score')

df_cmpg=df_cmpg.drop('count',1)

df_cmpg=df_cmpg.drop('sum',1)

df_cmpg.to_csv('df_cmpg.csv')



df_adv= df_mrg.groupby('advertiser_id').clicked.agg(['count' ,'sum']).reset_index()

df_adv['adv_Score']= (df_adv['sum'] + M) / (1 + df_adv['count'])

df_adv.sort('adv_Score')





#df_adv=df_adv.drop('count',1)

#df_adv=df_adv.drop('sum',1)

#df_adv.to_csv('df_adv.csv')

#df_adv



#pc=pc.merge(df_adv).merge(df_cmpg)

#pc=pc.drop('campaign_id',1)

#pc=pc.drop('advertiser_id',1)

#pc=pc.drop('document_id',1)



#pc


