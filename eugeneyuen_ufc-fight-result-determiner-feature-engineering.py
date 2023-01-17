# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cleaned_data.csv")

df.head()
# New DF for new features

new_features_df = pd.DataFrame()
series_body_att_b = df.iloc[:,10] # series for body att - blue

series_body_landed_b = df.iloc[:,11] # series for body landed - blue

new_features_df['effective_body_b'] = series_body_landed_b/series_body_att_b
series_clinch_att_b = df.iloc[:,12]

series_clinch_landed_b = df.iloc[:,13]

new_features_df['effective_clinch_b'] = series_clinch_landed_b / series_clinch_att_b
series_distance_att_b = df.iloc[:,14]

series_distance_landed_b = df.iloc[:,15]

new_features_df['effective_distance_shots'] = series_distance_landed_b / series_distance_att_b
series_ground_att_b = df.iloc[:,16]

series_ground_landed_b = df.iloc[:,17]

new_features_df['effective_ground_shots'] = series_ground_landed_b / series_ground_att_b
series_head_att_b = df.iloc[:,18]

series_head_landed_b = df.iloc[:, 19]

new_features_df['effective_head_shots'] = series_head_landed_b / series_head_att_b
series_leg_att_b = df.iloc[:,21]

series_leg_landed_b = df.iloc[:,22]

new_features_df['effective_leg_shots'] = series_leg_landed_b / series_leg_att_b
series_sig_strikes_att_b = df.iloc[:,25]

series_sig_strikes_landed_b = df.iloc[:,26]

new_features_df['effective_sig_strikes'] = series_sig_strikes_landed_b / series_sig_strikes_att_b
series_takedown_att_b = df.iloc[:,29]

series_takedown_landed_b = df.iloc[:,30]

new_features_df['effective_takedown_b'] = series_takedown_landed_b / series_takedown_att_b
series_total_strikes_att_b = df.iloc[:, 32]

series_total_strikes_landed_b = df.iloc[:,33]

new_features_df['series_effective_total_strikes'] = series_total_strikes_landed_b / series_total_strikes_att_b
series_height_b = df.iloc[:, 70] / 100

series_weight_kg_b = df.iloc[:, 72] / 2.2046

new_features_df['series_bmi_b'] = series_weight_kg_b / (series_height_b*series_height_b)
series_body_att_r = df.iloc[:, 76]

series_body_landed_r = df.iloc[:, 77]

new_features_df['effective_body_r'] = series_body_landed_r / series_body_att_r
series_clinch_att_r = df.iloc[:, 78]

series_clinch_landed_r = df.iloc[:, 79]

new_features_df['effective_clinch_r'] = series_clinch_landed_r / series_clinch_att_r
series_distance_att_r = df.iloc[:, 80]

series_distance_landed_r = df.iloc[:, 81]

new_features_df['effective_distance_r'] = series_distance_landed_r / series_distance_att_r
series_ground_att_r = df.iloc[:, 82]

series_ground_landed_r = df.iloc[:, 83]

new_features_df['effective_ground_r'] = series_ground_landed_r / series_ground_att_r
series_head_att_r = df.iloc[:, 84]

series_head_landed_r = df.iloc[:, 85]

new_features_df['effective_head_r'] = series_head_landed_r / series_head_att_r
series_leg_att_r = df.iloc[:, 87]

series_leg_landed_r = df.iloc[:, 88]

new_features_df['effective_leg_r'] = series_leg_landed_r / series_leg_att_r
series_sig_att_r = df.iloc[:, 91]

series_sig_landed_r = df.iloc[:, 92]

new_features_df['effective_sig_r'] = series_sig_landed_r / series_sig_att_r
series_takedown_att_r = df.iloc[:, 95]

series_takedown_landed_r = df.iloc[:, 96]

new_features_df['effective_takedown_r']  = series_takedown_landed_r / series_takedown_att_r
series_total_strikes_att_r = df.iloc[:, 98]

series_total_strikes_landed_r = df.iloc[:, 99]

new_features_df['effective_total_strikes_r']  = series_total_strikes_landed_r / series_total_strikes_att_r
# red - BMI

series_height_r = df.iloc[:, 136] / 100

series_weight_kg_r = df.iloc[:, 138] / 2.2046

new_features_df['series_bmi_r']  = series_weight_kg_r / (series_height_b*series_height_r)
new_features_df['height_diff_cms'] = abs(df['R_Height_cms'] - df['B_Height_cms'])
B_avg_TOTAL_STR_landed = df.iloc[:,33]

B_avg_opp_TOTAL_STR_landed = df.iloc[:,59]

new_features_df['B_striking_ratio'] = B_avg_TOTAL_STR_landed/B_avg_opp_TOTAL_STR_landed
R_avg_TOTAL_STR_landed = df.iloc[:,99]

R_avg_opp_TOTAL_STR_landed = df.iloc[:,125]

new_features_df['R_striking_ratio'] = R_avg_TOTAL_STR_landed/R_avg_opp_TOTAL_STR_landed
B_total_time_fought_seconds = df.iloc[:,61]

B_total_time_fought_mins = B_total_time_fought_seconds/60

new_features_df['B_Strike_Differential_per_minute'] = (B_avg_TOTAL_STR_landed-B_avg_opp_TOTAL_STR_landed)/B_total_time_fought_mins
R_total_time_fought_seconds = df.iloc[:,127]

R_total_time_fought_mins = R_total_time_fought_seconds/60

new_features_df['R_Strike_Differential_per_minute'] = (R_avg_TOTAL_STR_landed-R_avg_opp_TOTAL_STR_landed)/R_total_time_fought_mins
B_avg_TD_landed = df.iloc[:,30]

B_win_by_KO_TKO = df.iloc[:,66]

new_features_df['B_power_rating'] = (B_avg_TD_landed + B_win_by_KO_TKO )/ B_avg_TOTAL_STR_landed
#Power Rating (Red)

#(knockdowns + Knockouts/Technical Knockouts)/ (Total Strikes Landed) 

R_avg_TD_landed = df.iloc[:,96]

R_win_by_KO_TKO = df.iloc[:,132]

new_features_df['R_power_rating'] = (R_avg_TD_landed + R_win_by_KO_TKO )/ R_avg_TOTAL_STR_landed
#Checking that new features are in the df

len(new_features_df.columns)
#Create new DF with just new features and Winner col

new_feature_with_winner_df = pd.concat([df['Winner'], new_features_df], axis=1)
#Remove all Nan and inf rows

new_feature_with_winner_df = new_feature_with_winner_df.replace(float('inf'), 0)

new_feature_with_winner_df = new_feature_with_winner_df.fillna(0)

new_feature_with_winner_df.head()
df.shape #Check that the old df has 27 less columns with the new df a few cells later
new_features_df.shape
new_and_old_features_df = pd.concat([df, new_features_df], axis=1)
new_and_old_features_df.shape #Check that the old df has 27 less columns
new_and_old_features_df = new_and_old_features_df.replace(float('inf'), 0)

new_and_old_features_df = new_and_old_features_df.fillna(0)

new_and_old_features_df.head()
new_and_old_features_df.shape
#Next Step is to create a new DF that is from new_and_old_features_df but with the aggregated columns dropped.

new_and_old_features_dropped_aggregated_df = new_and_old_features_df.copy(deep=True)

new_and_old_features_dropped_aggregated_df.drop(['B_avg_BODY_att', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_DISTANCE_att', 'B_avg_DISTANCE_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_LEG_att', 'B_avg_LEG_landed', 'B_avg_SIG_STR_att', 'B_avg_SIG_STR_landed', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_avg_opp_TOTAL_STR_landed', 'B_total_time_fought(seconds)', 'B_win_by_KO/TKO', 'B_Height_cms', 'B_Weight_lbs', 'R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed', 'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_LEG_att', 'R_avg_LEG_landed', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed', 'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_TOTAL_STR_att', 'R_avg_TOTAL_STR_landed', 'R_avg_opp_TOTAL_STR_landed', 'R_total_time_fought(seconds)', 'R_win_by_KO/TKO', 'R_Height_cms', 'R_Weight_lbs'], axis=1,  inplace=True)
new_and_old_features_dropped_aggregated_df.shape
#Normalising some columns to change in the copy with aggregated data dropped

#DF to store normalised features

base_df = new_and_old_features_dropped_aggregated_df.copy(deep=True)

normalised_features_df = pd.DataFrame()

values_to_norm_dict = {0:"B_avg_KD",1:"B_avg_PASS",2:"B_avg_REV",

                       3:"B_avg_SIG_STR_pct",4:"B_avg_SUB_ATT",5:"B_avg_TD_pct",

                       6:"R_avg_KD",7:"R_avg_PASS",8:"R_avg_REV",

                       9:"R_avg_SIG_STR_pct",10:"R_avg_SUB_ATT",11:"R_avg_TD_pct"}



for i in range(len(values_to_norm_dict)):

    normalised_features_df[values_to_norm_dict[i]] = base_df[values_to_norm_dict[i]]



normalised_features_df.head()
#Normalising the values in normalised_features_df

x = normalised_features_df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

normalised_features_df = pd.DataFrame(x_scaled)

normalised_features_df.head()
#Copy of new_and_old_features_dropped_aggregated_df with normalised values

normalised_features_drop_aggregated_df = new_and_old_features_dropped_aggregated_df.copy(deep=True)

for i in range(len(values_to_norm_dict)):

    normalised_features_drop_aggregated_df.drop(columns=[values_to_norm_dict[i]], axis=1)

    normalised_features_drop_aggregated_df[values_to_norm_dict[i]] = normalised_features_df.iloc[:,i]
normalised_features_drop_aggregated_df.shape
#Normalising some columns to change in the copy without aggregated cols dropped

#DF to store normalised features

base_df_new_old_never_drop = new_and_old_features_df.copy(deep=True)

normalised_features_never_drop_df = pd.DataFrame()

values_to_norm_dict = {0:"B_avg_BODY_att",1:"B_avg_BODY_landed",2:"B_avg_CLINCH_att",3:"B_avg_CLINCH_landed",4:"B_avg_DISTANCE_att",5:"B_avg_DISTANCE_landed",

                       6:"B_avg_GROUND_att",7:"B_avg_GROUND_landed",8:"B_avg_HEAD_att",9:"B_avg_HEAD_landed",10:"B_avg_KD",11:"B_avg_LEG_att",12:"B_avg_LEG_landed",

                       13:"B_avg_PASS",14:"B_avg_REV",14:"B_avg_SIG_STR_att",15:"B_avg_SIG_STR_landed",16:"B_avg_SIG_STR_pct",17:"B_avg_SUB_ATT",18:"B_avg_TD_att",

                       19:"B_avg_TD_landed",20:"B_avg_TD_pct",21:"B_avg_TOTAL_STR_att",22:"B_avg_TOTAL_STR_landed",23:"B_avg_opp_BODY_att",24:"B_avg_opp_BODY_landed",

                       25:"B_avg_opp_CLINCH_att",26:"B_avg_opp_CLINCH_landed",27:"B_avg_opp_DISTANCE_att",28:"B_avg_opp_DISTANCE_landed",29:"B_avg_opp_GROUND_att",

                       30:"B_avg_opp_GROUND_landed",31:"B_avg_opp_HEAD_att",32:"B_avg_opp_HEAD_landed",33:"B_avg_opp_KD",34:"B_avg_opp_LEG_att",35:"B_avg_opp_LEG_landed",

                       36:"B_avg_opp_PASS",37:"B_avg_opp_REV",38:"B_avg_opp_SIG_STR_att",39:"B_avg_opp_SIG_STR_landed",40:"B_avg_opp_SIG_STR_pct",41:"B_avg_opp_SUB_ATT",42:"B_avg_opp_TD_att",

                       43:"B_avg_opp_TD_landed",44:"B_avg_opp_TD_pct",45:"B_avg_opp_TOTAL_STR_att",46:"B_avg_opp_TOTAL_STR_landed",47:"B_total_rounds_fought",48:"B_total_time_fought(seconds)",

                       49:"R_avg_BODY_att",50:"R_avg_BODY_landed",51:"R_avg_CLINCH_att",52:"R_avg_CLINCH_landed",53:"R_avg_DISTANCE_att",54:"R_avg_DISTANCE_landed",

                       55:"R_avg_GROUND_att",56:"R_avg_GROUND_landed",57:"R_avg_HEAD_att",58:"R_avg_HEAD_landed",59:"R_avg_KD",60:"R_avg_LEG_att",61:"R_avg_LEG_landed",

                       62:"R_avg_PASS",63:"R_avg_REV",64:"R_avg_SIG_STR_att",65:"R_avg_SIG_STR_landed",66:"R_avg_SIG_STR_pct",67:"R_avg_SUB_ATT",68:"R_avg_TD_att",

                       69:"R_avg_TD_landed",70:"R_avg_TD_pct",71:"R_avg_TOTAL_STR_att",72:"R_avg_TOTAL_STR_landed",73:"R_avg_opp_BODY_att",74:"R_avg_opp_BODY_landed",

                       75:"R_avg_opp_CLINCH_att",76:"R_avg_opp_CLINCH_landed",77:"R_avg_opp_DISTANCE_att",78:"R_avg_opp_DISTANCE_landed",79:"R_avg_opp_GROUND_att",

                       80:"R_avg_opp_GROUND_landed",81:"R_avg_opp_HEAD_att",82:"R_avg_opp_HEAD_landed",83:"R_avg_opp_KD",84:"R_avg_opp_LEG_att",85:"R_avg_opp_LEG_landed",

                       86:"R_avg_opp_PASS",87:"R_avg_opp_REV",88:"R_avg_opp_SIG_STR_att",89:"R_avg_opp_SIG_STR_landed",90:"R_avg_opp_SIG_STR_pct",91:"R_avg_opp_SUB_ATT",92:"R_avg_opp_TD_att",

                       93:"R_avg_opp_TD_landed",94:"R_avg_opp_TD_pct",95:"R_avg_opp_TOTAL_STR_att",96:"R_avg_opp_TOTAL_STR_landed",97:"R_total_rounds_fought",98:"R_total_time_fought(seconds)"



}



for i in range(len(values_to_norm_dict)):

    normalised_features_never_drop_df[values_to_norm_dict[i]] = base_df_new_old_never_drop[values_to_norm_dict[i]]



normalised_features_never_drop_df.head()
#Normalising the values in normalised_features_df

x_2 = normalised_features_never_drop_df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_2_scaled = min_max_scaler.fit_transform(x_2)

normalised_features_never_drop_df = pd.DataFrame(x_2_scaled)

normalised_features_never_drop_df.head()
#Copy of new_and_old_features_dropped_aggregated_df with normalised values

normalised_features_never_drop_aggregate_df = new_and_old_features_df.copy(deep=True)

for i in range(len(values_to_norm_dict)):

    normalised_features_never_drop_aggregate_df.drop(columns=[values_to_norm_dict[i]], axis=1)

    normalised_features_never_drop_aggregate_df[values_to_norm_dict[i]] = normalised_features_never_drop_df.iloc[:,i]



normalised_features_never_drop_aggregate_df.shape
#Making a DF for just pre-game data

pregame_df = df.copy(deep=True)

cols_to_drop = [6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,

                28,29,30,31,32,33,36,37,38,39,40,41,42,43,44,45,46,47,48,

                49,50,51,52,53,54,55,56,57,58,59,61,75,76,77,78,79,80,81,82,

                83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,102,103,

                104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,

                119,120,121,122,123,124,125,127]



pregame_df.drop(pregame_df.columns[cols_to_drop],axis=1,inplace=True)
#Save a copy of new DFs

new_and_old_features_df.to_csv("new_and_old_features_df.csv", index=False)

new_and_old_features_dropped_aggregated_df.to_csv("new_and_old_features_dropped_aggregated_df.csv", index=False)

new_feature_with_winner_df.to_csv("new_feature_with_winner_df.csv", index=False)

normalised_features_drop_aggregated_df.to_csv("normalised_features_drop_aggregated_df.csv", index=False)

normalised_features_never_drop_aggregate_df.to_csv("normalised_features_never_drop_aggregate_df.csv", index=False)

pregame_df.to_csv("pre_game_df.csv", index=False)

#Various ML models will be tested with these datasets