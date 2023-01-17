import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# This function will reduce our work load of searcing many players 
def GetBestPlayers(dframe,position,num):
    dframe=dframe.sort_values([position,'Overall'],ascending=False)[['ID','Name','Position','Overall',position]].reset_index(drop=True)[:num]
    return(dframe)
df=pd.read_csv('./fifadata.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
poslst=['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

df_pos=df[['ID','Name','Position','Overall']+poslst]
for col in poslst:
    df_pos[col]=df_pos[col].fillna(0).apply(lambda x : sum([int(i) for i in str(x).split('+')]))
# Get best position persons
# Players who have high position stat as well as high overall stat
# Let us now look into 442 formation
# 2-CF, 1-LM, 1-RM, 2-CM, 1-LB, 2-CB, 1-RB
team442=['CF','CF','LM','RM','CM','CM','LB','RB','CB','CB']

lstteam=[]
lstbest=[]
for pos in team442:
    bestlist=list(GetBestPlayers(df_pos,pos,5)['Name'])
    for player in bestlist:
        if (player not in lstteam):  
            lstteam.append(player)
            print(player,pos)
            break
df_best_442=pd.DataFrame([lstteam,team442]).T.rename(columns={0:'Name',1:'Position'})
df_best_442=pd.merge(df_pos[['Name','Overall','CF','LM','RM','CM','LB','RB','CB']],df_best_442,on=['Name'])
TotalScore=np.sum(df_best_442.Overall)
print("Best 4-4-2 formation team")
print('\n')
print(df_best_442)
print('\n')
print("Score", TotalScore)





