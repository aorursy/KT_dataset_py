import csv
import pandas as pd
import numpy as np
import time

pGD = pd.read_csv('../input/game_data.csv', encoding='latin-1') 
pPI = pd.read_csv('../input/play_information.csv', encoding='latin-1')
pPR = pd.read_csv('../input/play_player_role_data.csv', encoding='latin-1')
pPD = pd.read_csv('../input/player_punt_data.csv', encoding='latin-1')
pVR = pd.read_csv('../input/video_review.csv', encoding='latin-1')
from datetime import datetime
pPI['GameDate']= pd.Series([datetime.strptime(f, '%m/%d/%Y') for f in pPI['Game_Date']], index=pPI.index)
result1 = pd.merge(pGD,pPI[['GameKey','PlayID','Game_Clock','YardLine','Quarter','Play_Type','Poss_Team','Home_Team_Visit_Team','Score_Home_Visiting','PlayDescription']],
                      on='GameKey',how='right')
result2 = pd.merge(pPR[['GameKey','PlayID','GSISID']],pVR,
                      on=['GameKey', 'PlayID','GSISID'],how='left')
result3 = pd.merge(result2,pPD,
                      on='GSISID',how='left')
FinalData = pd.merge(result3,result1,
                      on=['GameKey','PlayID'],how='left')
FinalData['YardLineNum'] = FinalData.YardLine.str.extract('(\d+)').astype('float')
ConcusData = FinalData.loc[(FinalData['Season_Year_x'] == 2016) | (FinalData['Season_Year_x'] == 2017)]
FinalData['Concus'] = np.where(((FinalData['Season_Year_x'] == 2016) | (FinalData['Season_Year_x'] == 2017)),1,0)
import pylab as plt
fig, axs = plt.subplots(9,2, figsize=(20, 50))
ConcusData['Season_Year_x'].value_counts().plot(ax=axs[0,0],kind='bar').set_title('Season Year')
ConcusData['Player_Activity_Derived'].value_counts().plot(ax=axs[0,1],kind='bar').set_title('Player_Activity_Derived')
ConcusData['Primary_Impact_Type'].value_counts().plot(ax=axs[1,0], kind='bar').set_title('Primary_Impact_Type')
ConcusData['Primary_Partner_Activity_Derived'].value_counts().plot(ax=axs[1,1], kind='bar').set_title('Primary_Partner_Activity_Derived')
ConcusData['Friendly_Fire'].value_counts().plot(ax=axs[2,0], kind='bar').set_title('Friendly_Fire')
ConcusData['Number'].value_counts().plot(ax=axs[2,1], kind='bar').set_title('Number')
ConcusData['Position'].value_counts().plot(ax=axs[3,0], kind='bar').set_title('Position')
ConcusData['Season_Type'].value_counts().plot(ax=axs[3,1], kind='bar').set_title('Season_Type')
ConcusData['Week'].value_counts().plot(ax=axs[4,0], kind='bar').set_title('Week')
ConcusData['Game_Day'].value_counts().plot(ax=axs[4,1], kind='bar').set_title('Game_Day')
ConcusData['Game_Site'].value_counts().plot(ax=axs[5,0], kind='bar').set_title('Game_Site')
ConcusData['Start_Time'].value_counts().plot(ax=axs[5,1], kind='bar').set_title('Start_Time')
ConcusData['Home_Team'].value_counts().plot(ax=axs[6,0], kind='bar').set_title('Home_Team')
ConcusData['Visit_Team'].value_counts().plot(ax=axs[6,1], kind='bar').set_title('Visit_Team')
ConcusData['Stadium'].value_counts().plot(ax=axs[7,0], kind='bar').set_title('Stadium')
ConcusData['Temperature'].value_counts().plot(ax=axs[7,1], kind='bar').set_title('Temperature')
ConcusData['Game_Clock'].value_counts().plot(ax=axs[8,0], kind='bar').set_title('Game_Clock')
ConcusData['Quarter'].value_counts().plot(ax=axs[8,1], kind='bar').set_title('Quarter')

plt.show()
FinalData.dropna(how='all', inplace = True)
import statsmodels.api as sm

for i in ['GameKey', 'PlayID', 'GSISID', 'Season_Year_x', 'Player_Activity_Derived', 'Turnover_Related', 'Primary_Impact_Type', 'Primary_Partner_GSISID', 'Primary_Partner_Activity_Derived', 'Friendly_Fire', 'Position', 'Season_Type', 'Week', 'Game_Date', 'Game_Day', 'Game_Site', 'HomeTeamCode', 'VisitTeamCode', 'Stadium', 'StadiumType', 'Turf', 'GameWeather', 'Poss_Team']:
    cat_name = i + '_cat'
    FinalData[i].fillna(0, inplace=True)
    FinalData[i] = FinalData[i].astype('category')
    FinalData[cat_name] = FinalData[i].cat.codes
mean_temp = FinalData['Temperature'].mean()
FinalData['Temperature'].fillna(mean_temp, inplace=True)
FinalData['YardLineNum'].fillna(0,inplace=True)
FinalData['Quarter'].fillna(0,inplace=True)
FinalData['Season_Year_y'].fillna(0,inplace=True)
logit = sm.Logit(FinalData['Concus'], FinalData[['Season_Year_y', 'Week', 'Temperature', 'YardLineNum', 'Quarter', 'Concus', 'GameKey_cat', 'PlayID_cat', 'GSISID_cat', 'Season_Year_x_cat', 'Player_Activity_Derived_cat', 'Turnover_Related_cat', 'Primary_Impact_Type_cat', 'Primary_Partner_GSISID_cat', 'Primary_Partner_Activity_Derived_cat', 'Friendly_Fire_cat', 'Position_cat', 'Season_Type_cat', 'Week_cat', 'Game_Date_cat', 'Game_Day_cat', 'Game_Site_cat', 'HomeTeamCode_cat', 'VisitTeamCode_cat', 'Stadium_cat', 'StadiumType_cat', 'Poss_Team_cat']].astype('float'))
try:
    result = logit.fit()
except Exception as e:
    print(e)
#shows error is Singular Matrix 
print("Singular Matrix is caused by quasi-complete separation which leads to non-existent MLE")