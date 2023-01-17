import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/santa-kag1"))



# Any results you write to the current directory are saved as output.
#sub_1 = pd.read_csv("../input/santa-kag1/submission_rfc_cat.csv",names=["ID_code","target0"],skiprows=[0],header=None)

sub_2 = pd.read_csv("../input/santa-kag1/lgb.csv",names=["ID_code","target"],skiprows=[0],header=None)

#sub_3 = pd.read_csv("../input/santa-kag1/submission_cat.csv",names=["ID_code","target2"],skiprows=[0],header=None)

sub_4 = pd.read_csv("../input/santa-kag1/submission_lgb.csv",names=["ID_code","target3"],skiprows=[0],header=None)

sub_5 = pd.read_csv("../input/santa-kag1/submission.csv",names=["ID_code","target4"],skiprows=[0],header=None)

sub_6 = pd.read_csv("../input/santa-kag1/submission_xgb1.csv",names=["ID_code","target5"],skiprows=[0],header=None)

sub_7 = pd.read_csv("../input/santa-kag1/stcp2.csv",names=["ID_code","target6"],skiprows=[0],header=None)

sub_8 = pd.read_csv("../input/santa-kag1/toy_sub.csv",names=["ID_code","target7"],skiprows=[0],header=None)

sub_9 = pd.read_csv("../input/santa-kag1/LightGBM Minimize Leaves with GaussianNB.csv",names=["ID_code","target8"],skiprows=[0],header=None)

sub_10 = pd.read_csv("../input/santa-kag1/lgbsubmission1111.csv",names=["ID_code","target9"],skiprows=[0],header=None)

sub_11 = pd.read_csv("../input/santa-kag1/submission111.csv",names=["ID_code","target10"],skiprows=[0],header=None)

#sub_12 = pd.read_csv("../input/santa-kag1/sctp1.csv",names=["ID_code","target11"],skiprows=[0],header=None)

sub_13 = pd.read_csv("../input/santa-kag1/submission_lgb_cat.csv",names=["ID_code","target12"],skiprows=[0],header=None)

sub_14 = pd.read_csv("../input/santa-kag1/Customer_Transaction_rank_predictions.csv",names=["ID_code","target13"],skiprows=[0],header=None)

sub_15 = pd.read_csv("../input/santa-kag1/submission_lgb1.csv",names=["ID_code","target14"],skiprows=[0],header=None)

sub_16 = pd.read_csv("../input/santa-kag1/submission1.csv",names=["ID_code","target15"],skiprows=[0],header=None)
sub_2['target'] = (sub_2['target']+sub_4['target3']+sub_5['target4']+sub_6['target5']+

                   sub_7['target6']+sub_8['target7'] +sub_9['target8'] +sub_10['target9'] 

                   +sub_11['target10'] +sub_13['target12']+

                   sub_14['target13'] + +sub_15['target14'] +sub_16['target15'])/13.0

sub_2.to_csv('sol.csv',index=False)

sub_2.head()
#sub_1.head()
df_base = pd.merge(sub_4,sub_2,how='inner',on='ID_code')

#df_base = pd.merge(df_base,sub_3,how='inner',on='ID_code')

#df_base = pd.merge(df_base,sub_4,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_5,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_6,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_7,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_8,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_9,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_10,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_11,how='inner',on='ID_code')

#df_base = pd.merge(df_base,sub_12,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_13,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_14,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_15,how='inner',on='ID_code')

df_base = pd.merge(df_base,sub_16,how='inner',on='ID_code')

#df_base = pd.merge(df_base,sub_8,how='inner',on='ID_code')

#df_base.drop(['target0','target2','target11'],axis=1)



df_base.head()

import seaborn as sns

from matplotlib import pyplot as plt

plt.figure(figsize=(16,12))

sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")