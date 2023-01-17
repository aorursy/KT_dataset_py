# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.b
import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt
df_fifa = pd.read_csv('/kaggle/input/fifa19/data.csv', index_col=0)
df_fifa.head()
df_fifa.info()
df_fifa['Position'].unique()
df_fifa_fwd = df_fifa[df_fifa['Position'].isin(['RF', 'ST', 'LF', 'RS', 'LS', 'CF'])]
df_fifa_fwd_imp = df_fifa_fwd[['Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

                               'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping','Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 

                               'Composure','Marking', 'StandingTackle', 'SlidingTackle', 'Overall']]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf_model = rf.fit(df_fifa_fwd_imp.drop(['Overall'], axis=1), df_fifa_fwd_imp['Overall']);
df_fifa_fwd_feat_imp = pd.DataFrame(columns=['attributes', 'imp_values'])

df_fifa_fwd_feat_imp['imp_values'] = rf_model.feature_importances_

df_fifa_fwd_feat_imp['attributes'] = df_fifa_fwd_imp.drop(['Overall'], axis=1).columns
fig = plt.figure(figsize=(15,5))

sns.barplot(data = df_fifa_fwd_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10],

           x = df_fifa_fwd_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['imp_values'],

           y = df_fifa_fwd_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['attributes']);

plt.title('Important Attributes for Forwards')

plt.xlabel('Importance')

plt.ylabel('Attributes');
df_fifa['Position'].unique()
df_fifa_mid = df_fifa[df_fifa['Position'].isin(['RCM', 'LCM', 'LDM', 'RDM', 'CAM', 'CDM', 'RM', 'LM', 'RAM', 'LAM', 'CM'])]
df_fifa_mid_imp = df_fifa_mid[['Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

                               'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping','Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 

                               'Composure','Marking', 'StandingTackle', 'SlidingTackle', 'Overall']]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf_model = rf.fit(df_fifa_mid_imp.drop(['Overall'], axis=1), df_fifa_mid_imp['Overall']);
df_fifa_mid_feat_imp = pd.DataFrame(columns=['attributes', 'imp_values'])

df_fifa_mid_feat_imp['imp_values'] = rf_model.feature_importances_

df_fifa_mid_feat_imp['attributes'] = df_fifa_mid_imp.drop(['Overall'], axis=1).columns
fig = plt.figure(figsize=(15,5))

sns.barplot(data = df_fifa_mid_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10],

           x = df_fifa_mid_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['imp_values'],

           y = df_fifa_mid_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['attributes'])

plt.title('Important Attributes for Midfielders')

plt.xlabel('Importance')

plt.ylabel('Attributes');
df_fifa['Position'].unique()
df_fifa_wing = df_fifa[df_fifa['Position'].isin(['LW', 'RW'])]
df_fifa_wing_imp = df_fifa_wing[['Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

                             'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping','Stamina', 'Strength', 'LongShots', 

                             'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure','Marking', 'StandingTackle',

                             'SlidingTackle', 'Overall']]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf_model = rf.fit(df_fifa_wing_imp.drop(['Overall'], axis=1), df_fifa_wing_imp['Overall']);
df_fifa_wing_feat_imp = pd.DataFrame(columns=['attributes', 'imp_values'])

df_fifa_wing_feat_imp['imp_values'] = rf_model.feature_importances_

df_fifa_wing_feat_imp['attributes'] = df_fifa_wing_imp.drop(['Overall'], axis=1).columns
fig = plt.figure(figsize=(15,5))

sns.barplot(data = df_fifa_wing_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10],

           x = df_fifa_wing_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['imp_values'],

           y = df_fifa_wing_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['attributes'])

plt.title('Important Attributes for Wingers')

plt.xlabel('Importance')

plt.ylabel('Attributes');
df_fifa['Position'].unique()
df_fifa_def = df_fifa[df_fifa['Position'].isin(['LWB', 'RWB', 'CB', 'LCB', 'RCB', 'RB', 'LB'])]
df_fifa_def_imp = df_fifa_def[['Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

                             'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping','Stamina', 'Strength', 'LongShots', 

                             'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure','Marking', 'StandingTackle',

                             'SlidingTackle', 'Overall']]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf_model = rf.fit(df_fifa_def_imp.drop(['Overall'], axis=1), df_fifa_def_imp['Overall']);
df_fifa_def_feat_imp = pd.DataFrame(columns=['attributes', 'imp_values'])

df_fifa_def_feat_imp['imp_values'] = rf_model.feature_importances_

df_fifa_def_feat_imp['attributes'] = df_fifa_def_imp.drop(['Overall'], axis=1).columns
fig = plt.figure(figsize=(15,5))

sns.barplot(data = df_fifa_def_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10],

           x = df_fifa_def_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['imp_values'],

           y = df_fifa_def_feat_imp.sort_values(by=['imp_values'], ascending=False)[0:10]['attributes'])

plt.title('Important Attributes for Defenders')

plt.xlabel('Importance')

plt.ylabel('Attributes');
df_fifa_top_clubs = df_fifa[df_fifa['Overall']>=70].groupby(by=['Club']).agg({'Overall':'mean',

                                                                              'ID':'count'}).reset_index().sort_values(by='Overall', ascending=False).reset_index(drop=True)
df_fifa_top_clubs.rename(columns={'ID':'Player_Count'}, inplace=True)

df_fifa_top_clubs = df_fifa_top_clubs[df_fifa_top_clubs['Player_Count']>=20]
df_fifa_top_20 = df_fifa[df_fifa['Club'].isin(list(df_fifa_top_clubs['Club'][0:20]))]
fig = plt.figure(figsize=(15,8))

sns.barplot(data = df_fifa_top_clubs[0:20],

           x = df_fifa_top_clubs[0:20]['Overall'],

           y = df_fifa_top_clubs[0:20]['Club'],

           palette="Blues_d")

plt.title('Top Clubs with average player Overalls')

plt.xlabel('Overall')

plt.ylabel('Club');
young_age = df_fifa_top_20['Age'].quantile([0.33]).values[0]
df_fifa_top_20_youth = df_fifa_top_20[(df_fifa_top_20['Overall']<=df_fifa_top_20['Potential'])&(df_fifa_top_20['Age']<=young_age)&(df_fifa_top_20['Potential']>=80)]
df_fifa_top_20_youth_cnt = df_fifa_top_20_youth.groupby('Club').count().reset_index()[['Club', 'ID']]

df_fifa_top_20_youth_cnt.rename(columns={'ID':'Youth Count'}, inplace=True)
df_fifa_top_20_youth_cnt.sort_values(by='Youth Count', ascending=False, inplace=True)
fig = plt.figure(figsize=(15,8))

sns.barplot(data = df_fifa_top_20_youth_cnt[0:20],

           x = df_fifa_top_20_youth_cnt[0:20]['Youth Count'],

           y = df_fifa_top_20_youth_cnt[0:20]['Club'],

           palette="Blues_d")

plt.title('Top Clubs with the highest number of high potential youth players')

plt.xlabel('Count of Youth Players')

plt.ylabel('Club');
df_fifa_youth_superstars = df_fifa[(df_fifa['Overall']<=df_fifa['Potential'])&(df_fifa['Age']<=young_age)&(df_fifa['Potential']>=90)]  
df_fifa_youth_superstars[['Club', 'Name', 'Age', 'Position', 'Nationality', 'Overall', 'Potential']].sort_values(by='Potential', ascending=False).reset_index(drop=True)