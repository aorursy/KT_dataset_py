import numpy as np

import pandas as pd

pd.set_option('display.max_rows', None) 
all_User_Achievements = pd.read_csv('../input/meta-kaggle/UserAchievements.csv')

notebooks_DF = all_User_Achievements.loc[(all_User_Achievements['AchievementType'] == 'Scripts')&(all_User_Achievements['Tier'] == 3)]

#convert TierAchievementDate from object to datetime64[ns]

notebooks_DF.loc[:,'TierAchievementDate'] = pd.to_datetime(notebooks_DF['TierAchievementDate'])
# uncomment to show the whole dataframe

#notebooks_DF.sort_values('TierAchievementDate')
notebooks_DF.shape[0]
notebooks_DF['TierAchievementDate'].groupby([notebooks_DF.TierAchievementDate.dt.year, notebooks_DF.TierAchievementDate.dt.month]).agg('count')