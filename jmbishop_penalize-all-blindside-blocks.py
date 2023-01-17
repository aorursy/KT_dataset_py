# Import packages
import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.preprocessing import Normalizer
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')
# Import and view concussion play data
video_review = pd.read_csv('../input/video_review.csv')
video_review
# Retrieve NGS data for concussion plays
NGS = []
for f in os.listdir('../input'):
    if f.startswith('NGS'):
        df = pd.read_csv('../input/' + f)
        df = df[(df['GameKey'].isin(list(video_review['GameKey']))) & (df['PlayID'].isin(list(video_review['PlayID'])))]
        NGS.append(df)
NGS = pd.concat(NGS)
del NGS['Season_Year'], NGS['Event']
# Merge with NGS data
filtered_NGS = video_review[['GameKey', 'PlayID', 'GSISID', 'Primary_Partner_GSISID']].merge(NGS, how='left', on=['GameKey', 'PlayID', 'GSISID'])
NGS['Primary_Partner_GSISID'] = NGS['GSISID']
filtered_NGS['Primary_Partner_GSISID'] = pd.to_numeric(filtered_NGS['Primary_Partner_GSISID'], errors='coerce')
filtered_NGS = filtered_NGS.merge(NGS.drop(['GSISID'], axis=1), how='left', on=['GameKey', 'PlayID', 'Primary_Partner_GSISID', 'Time'], suffixes=('', '_Partner'))
# Create features
filtered_NGS['PlayKey'] = filtered_NGS['GameKey'] + filtered_NGS['PlayID'] + filtered_NGS['GSISID']
filtered_NGS['player_dist'] = ((filtered_NGS['x'] - filtered_NGS['x_Partner'])**2 + (filtered_NGS['y'] - filtered_NGS['y_Partner'])**2)**0.5
filtered_NGS['Time'] = pd.to_datetime(filtered_NGS['Time'])
filtered_NGS['time_since_play_start'] = (filtered_NGS['Time'] - filtered_NGS.groupby(['PlayKey'])['Time'].transform(min)).dt.total_seconds()
filtered_NGS['impact_time'] = np.where(filtered_NGS.groupby(['PlayKey'])['player_dist'].transform(min) == filtered_NGS['player_dist'], filtered_NGS['time_since_play_start'], np.nan)
filtered_NGS['time_to_impact'] = filtered_NGS.groupby(['PlayKey'])['impact_time'].transform(min) - filtered_NGS['time_since_play_start']
filtered_NGS['o_diff'] = np.min([filtered_NGS['o'] - filtered_NGS['o_Partner'], 180 - (filtered_NGS['o'] - filtered_NGS['o_Partner'])], axis=0)
filtered_NGS['dir_diff'] = np.min([filtered_NGS['dir'] - filtered_NGS['dir_Partner'], 180 - (filtered_NGS['dir'] - filtered_NGS['dir_Partner'])], axis=0)
# Query model input
model_input = filtered_NGS[(0.5 >= filtered_NGS['time_to_impact']) & (filtered_NGS['time_to_impact'] >= 0)]
model_input = model_input.sort_values(['PlayKey', 'time_to_impact'])
model_input['obs_num'] = model_input.groupby(['PlayKey']).cumcount()
model_input = model_input.pivot(index='PlayKey', columns='obs_num', values=['o_diff', 'dir_diff', 'dis', 'dis_Partner'])
# Perform clustering
model = AffinityPropagation().fit(Normalizer().fit_transform(model_input))
labels = pd.Series(model.labels_, name= 'cluster', index=model_input.index)
print('Number of plays in each cluster:')
labels.groupby(labels).count().sort_values(ascending=False)
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153280/Wing_37_yard_punt-cPHvctKg-20181119_165941654_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153258/61_yard_Punt_by_Brett_Kern-g8sqyGTz-20181119_162413664_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153272/Haack_42_yard_punt-iP6aZSRU-20181119_165050694_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153321/Lechler_55_yd_punt-lG1K51rf-20181119_173634665_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153244/Punt_by_Brad_Wing-5hmlbMBx-20181119_155243111_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153235/Punt_Return_by_Jamison_Crowder-HnVouDMH-20181119_153041501_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284956/284956_12D27120C06E4DB994040750FB43991D_181125_284956_way_punt_3200.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153274/Haack_punts_41_yards-SRJMeOc3-20181119_165546590_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284954/284954_75F12432BA90408C92660A696C1A12C8_181125_284954_huber_punt_3200.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153326/Sanchez_27_yd_punt-r51JAWPm-20181119_174359780_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153245/Punt_by_Brad_Nortman-QiQqjFdU-20181119_155917392_5000k.mp4" type="video/mp4"></video>')
video_review['PlayKey'] = video_review['GameKey'] + video_review['PlayID'] + video_review['GSISID']
video_review = video_review.merge(labels.reset_index(), how='left', on='PlayKey').sort_values('cluster')
video_review.groupby(['cluster'])['Friendly_Fire'].value_counts()
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153273/King_62_yard_punt-BSOws7nQ-20181119_165306255_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153257/40_yard_Punt_by_Brad_Nortman-oSbtDlHu-20181119_162303930_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153247/Punt_by_Tress_Way-QsI21aYF-20181119_160141260_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153260/Bolden_runs_into_Colquitt_during_punt-LpPhshZz-20181119_162648901_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153246/Punt_by_Brad_Nortman_2-fbS6OgDd-20181119_160019423_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153242/Punt_by_Dustin_Colquitt-joFpAUDf-20181119_155010468_5000k.mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153240/Punt_by_Thomas_Morstead-eZpDKgMR-20181119_154525222_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153236/Punt_by_Brad_Wing-SMRxqgb2-20181119_153645589_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153248/Rush_by_Jon_Ryan-Csg9PS77-20181119_160437472_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153291/Palardy_53_yard_punt-XTESVMq9-20181119_170509550_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153252/44_yard_Punt_by_Justin_Vogel-n7U6IS6I-20181119_161556468_5000k.mp4" type="video/mp4"></video>')