# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fight_df = pd.read_csv('/kaggle/input/ufc-data/fights_v2.csv')

fight_df
fight_df['B_heavy_hands'] = (fight_df['B_avg_KD'] / fight_df['B_avg_HEAD_landed']) 

fight_df['R_heavy_hands'] = (fight_df['R_avg_KD'] / fight_df['R_avg_HEAD_landed'])  
fight_df['B_chin'] = (fight_df['B_avg_opp_KD'] / fight_df['B_avg_opp_HEAD_landed']) 

fight_df['R_chin'] = (fight_df['R_avg_opp_KD'] / fight_df['R_avg_opp_HEAD_landed'])  
fight_df['B_power_stat'] = fight_df['B_avg_KD'] / fight_df['B_avg_SIG_STR_landed']

fight_df['R_power_stat'] = fight_df['R_avg_KD'] / fight_df['R_avg_SIG_STR_landed']
#note the denominator is 1 + total time to avoid errors from dividing by zero if its a fighter is making their debut

fight_df['B_volume'] = (fight_df['B_avg_TOTAL_STR_att'] / 1 + fight_df['B_total_time_fought(seconds)']) 

fight_df['R_volume'] = (fight_df['R_avg_TOTAL_STR_att'] / 1 + fight_df['R_total_time_fought(seconds)'])  
#note the denominator is 1 + total time to avoid errors from dividing by zero if its a fighter is making their debut

fight_df['B_eff_volume'] = (fight_df['B_avg_SIG_STR_landed'] / 1 + fight_df['B_total_time_fought(seconds)']) 

fight_df['R_eff_volume'] = (fight_df['R_avg_SIG_STR_landed'] / 1 + fight_df['R_total_time_fought(seconds)'])
fight_df['B_groundnpound'] = (fight_df['B_avg_GROUND_landed'] /  fight_df['B_avg_TD_landed']) 

fight_df['R_groundnpound'] = (fight_df['R_avg_GROUND_landed'] /  fight_df['R_avg_TD_landed'])  
fight_df['B_sub_eff'] = (fight_df['B_avg_SUB_ATT'] /  fight_df['B_avg_TD_landed'])

fight_df['R_sub_eff'] = (fight_df['R_avg_SUB_ATT'] /  fight_df['R_avg_TD_landed']) 
fight_df['B_ground_atx'] = fight_df['B_sub_eff'] + fight_df['B_groundnpound'] 

fight_df['R_ground_atx'] = fight_df['R_sub_eff'] + fight_df['R_groundnpound'] 
fight_df['B_HEAD_exchange_profit'] = (fight_df['B_avg_HEAD_landed'] - fight_df['B_avg_opp_HEAD_landed'] )

fight_df['R_HEAD_exchange_profit'] = (fight_df['R_avg_HEAD_landed'] - fight_df['R_avg_opp_HEAD_landed'] ) 
fight_df['B_exchange_profit'] = fight_df['B_avg_SIG_STR_pct'] - fight_df['B_avg_opp_SIG_STR_pct']

fight_df['R_exchange_profit'] = fight_df['R_avg_SIG_STR_pct'] - fight_df['R_avg_opp_SIG_STR_pct']
fight_df['B_bmi'] = 703*(fight_df['B_Weight_lbs'] / fight_df['B_Height']**2)

fight_df['R_bmi'] = 703*(fight_df['R_Weight_lbs'] / fight_df['R_Height']**2)
fight_df['B_Reach/h'] = (fight_df['B_Reach'] / fight_df['B_Height'])

fight_df['R_Reach/h'] = (fight_df['R_Reach'] / fight_df['R_Height'])
fight_df['B_Reach_by_power'] = (fight_df['B_Reach'] / fight_df['B_power_stat'])

fight_df['R_Reach_by_power'] = (fight_df['R_Reach'] / fight_df['R_power_stat'])
fight_df['B_Reach/bmi'] = (fight_df['B_Reach'] / fight_df['B_bmi'])

fight_df['R_Reach/bmi'] = (fight_df['R_Reach'] / fight_df['R_bmi'])
fight_df['B_Reach/vol'] = (fight_df['B_Reach'] / fight_df['B_volume'])

fight_df['R_Reach/vol'] = (fight_df['R_Reach'] / fight_df['R_volume'])
new_varibles = {'ground_atx':[],'sub_eff':[],'groundnpound':[],'eff_volume':[],

                'volume':[],'power_stat':[],'chin':[],'heavy_hands':[],

                'exchange_profit':[],'HEAD_exchange_profit':[],

                'Reach/vol':[],'Reach/bmi':[],'Reach_by_power':[],'Reach/h':[],'bmi':[]}



for varible in new_varibles:

    for fight in fight_df.itertuples():

        if fight.Winner_Corner == 'Red':

            if fight_df.at[fight.Index,'R_'+varible] > fight_df.at[fight.Index,'B_'+varible]:

                new_varibles[varible].append(1)

        elif fight.Winner_Corner == 'Blue':

            if fight_df.at[fight.Index,'B_'+varible] > fight_df.at[fight.Index,'R_'+varible]:

                new_varibles[varible].append(1)

            



for var in new_varibles.items():

    print(str(var[0]) +' predicts ' +str(round(100*(sum(var[1])/len(fight_df)),3))+'% of fights \n')

        

    
predicted=[]



for fight in fight_df.itertuples():

    if fight.Winner_Corner == 'Red':

        

        if fight_df.at[fight.Index,'R_avg_HEAD_landed']  > fight_df.at[fight.Index,'B_avg_HEAD_landed']:

             predicted.append(1)

            

    elif fight.Winner_Corner == 'Blue':

        if fight_df.at[fight.Index,'B_avg_HEAD_landed']  > fight_df.at[fight.Index,'R_avg_HEAD_landed']:

            predicted.append(1)



print('This predicts ' +str(round(100*(sum(predicted)/len(fight_df)),3))+'% of fights \n')

    
predicted=[]

for fight in fight_df.itertuples():

    if fight.Winner_Corner == 'Red':

        

        if fight_df.at[fight.Index,'R_heavy_hands'] > fight_df.at[fight.Index,'B_chin']:

            predicted.append(1)

            

    elif fight.Winner_Corner == 'Blue':

        if fight_df.at[fight.Index,'B_heavy_hands'] > fight_df.at[fight.Index,'R_chin']:

            predicted.append(1)



print('Heavy hands > chin predicts ' +str(round(100*(sum(predicted)/len(fight_df)),3))+'% of fights \n')

    
predicted=[]



for fight in fight_df.itertuples():

    if fight.Winner_Corner == 'Red':

        

        if fight_df.at[fight.Index,'R_HEAD_exchange_profit']   >  fight_df.at[fight.Index,'B_HEAD_exchange_profit'] or fight_df.at[fight.Index,'R_Reach/bmi']   > fight_df.at[fight.Index,'B_Reach/bmi']:

            predicted.append(1)

            

    elif fight.Winner_Corner == 'Blue':

        if fight_df.at[fight.Index,'B_HEAD_exchange_profit']  >  fight_df.at[fight.Index,'R_HEAD_exchange_profit'] or fight_df.at[fight.Index,'B_Reach/bmi']   > fight_df.at[fight.Index,'R_Reach/bmi']:

            predicted.append(1)



print('This predicts ' +str(round(100*(sum(predicted)/len(fight_df)),3))+'% of fights \n')

    
print(round(fight_df['R_HEAD_exchange_profit'].corr(fight_df['R_HEAD'] - fight_df['B_HEAD']),3))

print(round(fight_df['B_HEAD_exchange_profit'].corr(fight_df['B_HEAD'] - fight_df['R_HEAD']),3))