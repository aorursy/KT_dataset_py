import os

import pandas as pd

import numpy as np



from warnings import filterwarnings

filterwarnings('ignore')
df_odi=pd.read_csv('../input/master-blaster-sachin-tendulkar-dataset/sachin_odi.csv')

df_odi.shape
df_odi.info()
col_=['wickets','runs_conceded','catches','stumps']

for col in col_:

    df_odi[col][df_odi[col]=='-']=np.nan

    df_odi[col]=df_odi[col].astype('float')

df_odi.describe()
df_odi["notout"]=0

for i in range(df_odi.shape[0]):

    if df_odi.batting_score[i]=='DNB':

        df_odi.batting_score[i]=np.nan

        df_odi.notout[i]=np.nan

    elif df_odi.batting_score[i]=='TDNB':

        df_odi.batting_score[i]=np.nan

        df_odi.notout[i]=np.nan

    elif df_odi.batting_score[i].endswith('*')==True:

        df_odi.batting_score[i]=df_odi.batting_score[i].replace('*','')

        df_odi.notout[i]=1

        

df_odi.batting_score=df_odi.batting_score.astype('float')
for i in range(df_odi.shape[0]):

    df_odi.opposition[i]=df_odi.opposition[i].replace('v ','')
df_odi.date=pd.to_datetime(df_odi.date, format='%d %b %Y')
df_odi=df_odi[['batting_score', 'notout', 'wickets', 'runs_conceded', 'catches', 'stumps',

       'opposition', 'ground', 'date', 'match_result', 'result_margin', 'toss',

       'batting_innings']]
df_odi.info()