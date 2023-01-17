import numpy as np

import pandas as pd
tourney_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv', sep='\,', engine='python')

tourney_df = tourney_df[['Season','DayNum','WTeamID','WScore','LTeamID','LScore']]

tourney_df = tourney_df[tourney_df['Season']>2015]



tourney_df1 = tourney_df.copy()

tourney_df2 = tourney_df.copy()



tourney_df1.columns = ['Season','DayNum','T1_Team_ID','T1_Score','T2_Team_ID','T2_Score']

tourney_df1['T1_Won'] = 1



tourney_df2.columns = ['Season','DayNum','T2_Team_ID','T2_Score','T1_Team_ID','T1_Score']

tourney_df2['T1_Won'] = 0



tourney_all = tourney_df1.append(tourney_df2, sort=False)



tourney_all['Rd'] = 1

tourney_all.loc[(tourney_all['DayNum']==138) | (tourney_all['DayNum']==139), 'Rd'] = 2

tourney_all.loc[(tourney_all['DayNum']==143) | (tourney_all['DayNum']==144), 'Rd'] = 3

tourney_all.loc[(tourney_all['DayNum']==145) | (tourney_all['DayNum']==146), 'Rd'] = 4

tourney_all.loc[tourney_all['DayNum']==152, 'Rd'] = 5

tourney_all.loc[tourney_all['DayNum']==154, 'Rd'] = 6

tourney_all
df_538 = pd.read_excel(open('../input/538-data/538.xlsx', 'rb'),sheet_name='All')  

df_538 = df_538[['Season','TEAM','POWER RATING','Round1','Round2','Round3','Round4','Round5','Round6']]



# string cleaning to attach team ids

df_538['TEAM'] = df_538['TEAM'].str.lower()

df_538['TEAM'] = df_538['TEAM'].str.extract('([a-z. ]+)', expand=True) 



df_538.loc[df_538['TEAM']=="miss. state","TEAM"] = "mississippi st"

df_538.loc[df_538['TEAM']=="st. mary","TEAM"] = "saint mary's"

df_538.loc[df_538['TEAM']=="nm state","TEAM"] = "new mexico st"

df_538.loc[df_538['TEAM']=="neastern","TEAM"] = "northeastern"

df_538.loc[df_538['TEAM']=="uc","TEAM"] = "uc irvine"

df_538.loc[df_538['TEAM']=="n. kentucky","TEAM"] = "northern kentucky"

df_538.loc[df_538['TEAM']=="st. louis","TEAM"] = "saint louis"

df_538.loc[df_538['TEAM']=="old dom.","TEAM"] = "old dominion"

df_538.loc[df_538['TEAM']=="gardner","TEAM"] = "gardner webb"

df_538.loc[df_538['TEAM']=="abilene chr.","TEAM"] = "abilene chr"

df_538.loc[df_538['TEAM']=="nd state","TEAM"] = "north dakota state"

df_538.loc[df_538['TEAM']=="f. dickinson","TEAM"] = "f dickinson"

df_538.loc[df_538['TEAM']=="w. virginia","TEAM"] = "west virginia"

df_538.loc[df_538['TEAM']=="miami","TEAM"] = "miami fl"

df_538.loc[df_538['TEAM']=="texas a","TEAM"] = "texas a&m"

df_538.loc[df_538['TEAM']=="sdsu","TEAM"] = "san diego st"

df_538.loc[df_538['TEAM']=="st. bon.","TEAM"] = "st bonaventure"

df_538.loc[df_538['TEAM']=="loyola ","TEAM"] = "loyola chicago"

df_538.loc[df_538['TEAM']=="s. dakota st.","TEAM"] = "s dakota st"

df_538.loc[df_538['TEAM']=="csu","TEAM"] = "csu bakersfield"

df_538.loc[df_538['TEAM']=="txso","TEAM"] = "texas southern"

df_538.loc[df_538['TEAM']=="okla. state","TEAM"] = "oklahoma state"

df_538.loc[df_538['TEAM']=="s. carolina","TEAM"] = "south carolina"

df_538.loc[df_538['TEAM']=="mid. tenn.","TEAM"] = "middle tennessee"

df_538.loc[df_538['TEAM']=="nwestern","TEAM"] = "northwestern"

df_538.loc[df_538['TEAM']=="e. tenn. st.","TEAM"] = "east tennessee state"

df_538.loc[df_538['TEAM']=="n. dakota","TEAM"] = "north dakota"

df_538.loc[df_538['TEAM']=="jax. state","TEAM"] = "jacksonville state"

df_538.loc[df_538['TEAM']=="m. st. mary","TEAM"] = "mount st. mary's"

df_538.loc[df_538['TEAM']=="st. joseph","TEAM"] = "st joseph's"

df_538.loc[df_538['TEAM']=="n. iowa","TEAM"] = "northern iowa"

df_538.loc[df_538['TEAM']=="ark.","TEAM"] = "arkansas"

df_538.loc[df_538['TEAM']=="chatt.","TEAM"] = "chattanooga"

df_538.loc[df_538['TEAM']=="csu bakers.","TEAM"] = "csu bakersfield"



df_538.head(25)
teams_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamSpellings.csv', sep='\,', engine='python')

teams_df
df_538 = pd.merge(df_538, teams_df, left_on=['TEAM'], right_on = ['TeamNameSpelling'], how='left')

df_538 = df_538.drop(['TeamNameSpelling'], axis=1)

df_538
df_538_long = df_538.melt(id_vars=['Season','TEAM','TeamID','POWER RATING'], var_name="Rd", value_name="Prob538")

df_538_long['Rd'] = df_538_long['Rd'].str.extract('(\d)', expand=True)

df_538_long
df_538_probs = df_538_long[['Season','TeamID','Rd','Prob538','POWER RATING']]

df_538_probs['TeamID'] = df_538_probs['TeamID'].astype(int)

df_538_probs['Rd'] = df_538_probs['Rd'].astype(int)

df_538_probs
tourney_all_merged = pd.merge(tourney_all, df_538_probs, left_on=['Season','T1_Team_ID','Rd'], right_on = ['Season','TeamID','Rd'], how='left')

tourney_all_merged.rename(columns={'Prob538': 'T1_Prob538'}, inplace=True)



tourney_all_merged = pd.merge(tourney_all_merged, df_538_probs, left_on=['Season','T2_Team_ID','Rd'], right_on = ['Season','TeamID','Rd'], how='left')

tourney_all_merged.rename(columns={'Prob538': 'T2_Prob538'}, inplace=True)



tourney_all_merged = tourney_all_merged[['Season','Rd','T1_Won','T1_Team_ID','T2_Team_ID','T1_Prob538','T2_Prob538']]



tourney_all_merged
# fix missings (investigate these further)

tourney_all_merged.loc[tourney_all_merged['T1_Prob538'].isna(), 'T1_Prob538'] = 1-tourney_all_merged['T2_Prob538']

tourney_all_merged.loc[tourney_all_merged['T2_Prob538'].isna(), 'T2_Prob538'] = 1-tourney_all_merged['T1_Prob538']

tourney_all_merged
# average head-to-head

tourney_all_merged['Prob538'] = (tourney_all_merged['T1_Prob538']+(1-tourney_all_merged['T2_Prob538']))/2

tourney_all_merged 
def LogLoss(predictions, realizations):

    predictions_use = predictions.clip(0)

    realizations_use = realizations.clip(0)

    LogLoss = -np.mean( (realizations_use * np.log(predictions_use)) + 

                        (1 - realizations_use) * np.log(1 - predictions_use) )

    return LogLoss
# all rounds



losses = 0

seasons = [2016,2017,2018,2019]

for season in seasons:

    temp = tourney_all_merged[tourney_all_merged['Season']==season]

    loss = LogLoss(temp['Prob538'], temp['T1_Won'])



    print("Season",season,"valid:",loss)

    losses = losses+loss

    

losses_avg = losses/4

print("Average :",losses_avg)
# just round 1



losses = 0

seasons = [2016,2017,2018,2019]

for season in seasons:

    temp = tourney_all_merged[tourney_all_merged['Season']==season]

    temp = temp[temp['Rd']==1]

    loss = LogLoss(temp['Prob538'], temp['T1_Won'])



    print("Season",season,"valid:",loss)

    losses = losses+loss

    

losses_avg = losses/4

print("Average :",losses_avg)