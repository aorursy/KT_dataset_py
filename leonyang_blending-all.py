import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import rankdata

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
# oof=pd.read_csv("../input/lgb-2-leaves-augment/lgb_oof.csv")

# oof.head()
# oof['bins']=pd.qcut(oof['predict'],100,labels=False)

# oof.head()
# temp=oof.groupby(['bins'])['target'].agg({"mean"})

# temp=temp.reset_index()

# bins=80

# plt.plot(temp[temp.bins>bins]['bins'],temp[temp.bins>bins]['mean'])
result_1=pd.read_csv("../input/lgb-argu-0407/lgb_submission.csv")

result_2=pd.read_csv("../input/lgb-without-argu-0405/lgb_submission.csv")

result_3=pd.read_csv("../input/no-training-challenge/submission.csv")

result_4=pd.read_csv("../input/nffm-0405-xdeepfm/nffm_submission.csv")

result_5=pd.read_csv("../input/lgb-2-leaves-augment/lgb_submission.csv")

result_6=pd.read_csv("../input/lgb-2-leaves-augment-mine/lgb_submission.csv")

result_7=pd.read_csv("../input/lgb-noaug-pdf200/lgb_submission.csv")
result_1['index_1']=pd.DataFrame(rankdata(result_1['target'].tolist()))/len(result_1)

result_2['index_2']=pd.DataFrame(rankdata(result_2['target'].tolist()))/len(result_2)

result_3['index_3']=pd.DataFrame(rankdata(result_3['target'].tolist()))/len(result_3)

result_4['index_4']=pd.DataFrame(rankdata(result_4['target'].tolist()))/len(result_4)

result_5['index_5']=pd.DataFrame(rankdata(result_5['target'].tolist()))/len(result_5)

result_6['index_6']=pd.DataFrame(rankdata(result_6['target'].tolist()))/len(result_6)

result_7['index_7']=pd.DataFrame(rankdata(result_7['target'].tolist()))/len(result_7)
result_1.head()
pri_lb=np.load("../input/list-of-fake-samples-and-public-private-lb-e2795a/private_LB.npy")

pub_lb=np.load("../input/list-of-fake-samples-and-public-private-lb-e2795a/public_LB.npy")

df_test_new=result_1.iloc[list(pri_lb)+list(pub_lb),:][['ID_code']]

df_test_new.head()
df_test_new['real']=1

df_test_new.head()
result_avg=pd.merge(result_1,result_2,on='ID_code')

result_avg=pd.merge(result_avg,result_3,on='ID_code')

result_avg=pd.merge(result_avg,result_4,on='ID_code')

result_avg=pd.merge(result_avg,result_5,on='ID_code')

result_avg=pd.merge(result_avg,result_6,on='ID_code')

result_avg=pd.merge(result_avg,result_7,on='ID_code')
result_avg=result_avg.merge(df_test_new,on='ID_code',how='left')

result_avg.head()
result_avg.fillna(0,inplace=True)

result_avg.real.value_counts()
index_list=[]

for i in result_avg.columns:

    if 'index_' in i:

        index_list.append(i)

result_avg['target']=0.0

for i in index_list:

    result_avg['target']+=result_avg[i]/len(index_list)
from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

plt.figure(figsize=(8,6))

result_realtest=result_avg.loc[result_avg['real']==1]

sns.heatmap(result_avg[['ID_code','target']+index_list].iloc[:,1:].corr(),annot=True,fmt=".2f")
#result_avg['target']=(result_avg['index_1']*0.899+result_avg['index_2']*0.899+result_avg['index_3']*0.894+result_avg['index_4']*0.842+result_avg['index_5']*0.901*2)/(0.899+0.899+0.894+0.842+2*0.901)

result_avg['target']=(result_avg['index_4']*0.842+result_avg['index_5']*0.901+result_avg['index_6']*0.901)/(0.842+0.901*2)
result_avg.head()
result=result_avg[['ID_code','target']]

ymd=time.strftime("%Y%m%d")

hms=time.strftime("%H%M%S")

name="blend_all_{0}_{1}.csv".format(ymd,hms)

result.to_csv(name,encoding='utf-8',index=None)