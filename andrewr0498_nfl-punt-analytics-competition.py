import pandas as pd
import re
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
%%capture
ngs_2016_pre_conc = pd.read_csv('../input/punt-data/con_ngs_2016_pre.csv')
ngs_2016_reg_wk1_6 = pd.read_csv('../input/punt-data/con_ngs_2016_wk1_6.csv')
ngs_2016_reg_wk7_12 = pd.read_csv('../input/punt-data/con_ngs_2016_wk7_12.csv')
ngs_2016_reg_wk13_17 = pd.read_csv('../input/punt-data/con_ngs_2016_wk13_17.csv')
ngs_2017_pre_conc = pd.read_csv('../input/punt-data/con_ngs_2017_pre.csv')
ngs_2017_reg_wk1_6 = pd.read_csv('../input/punt-data/con_ngs_2017_wk1_6.csv')
ngs_2017_reg_wk7_12 = pd.read_csv('../input/punt-data/con_ngs_2017_wk7_12.csv')
ngs_2017_reg_wk13_17 = pd.read_csv('../input/punt-data/con_ngs_2017_wk13_17.csv')

ngs_list = [ngs_2016_pre_conc, ngs_2016_reg_wk1_6, ngs_2016_reg_wk7_12, ngs_2016_reg_wk13_17, ngs_2017_pre_conc, ngs_2017_reg_wk1_6, ngs_2017_reg_wk7_12, ngs_2017_reg_wk13_17]
ngs_conc = pd.concat(ngs_list)
ngs_conc['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in ngs_conc.GameKey], [str(x) for x in ngs_conc.PlayID], [str(x) for x in ngs_conc.Season_Year])]
ngs_conc = ngs_conc.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
ngs_conc.head()
video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_footage_control = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')
video_footage_injury = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
punt_data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')

regexp_no_punt = re.compile('(No)\s(Play)|(Delay)\s(of)\s(Game)|(Aborted)|(pass)|(False)\s(Start)')
no_punt = [regexp_no_punt.search(x) == None for x in punt_data.PlayDescription]

### There was a penalty on the play so the punt didn't count but there was still a concussion
no_punt[4018] = True
punt_data = punt_data[no_punt]

video_review['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in video_review.GameKey], [str(x) for x in video_review.PlayID], [str(x) for x in video_review.Season_Year])]
video_footage_injury['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in video_footage_injury.gamekey], [str(x) for x in video_footage_injury.playid], [str(x) for x in video_footage_injury.season])]
video_footage_control['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in video_footage_control.gamekey], [str(x) for x in video_footage_control.playid], [str(x) for x in video_footage_control.season])]
punt_data['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in punt_data.GameKey], [str(x) for x in punt_data.PlayID], [str(x) for x in punt_data.Season_Year])]

injury_data = video_review.merge(video_footage_injury, on='unique_id')
data = punt_data.merge(injury_data, on='unique_id', how='outer', suffixes=('', '_drop'))
data = data.iloc[:, np.r_[0:15, 19:25]]
data.info()
regexp_fair_catch = re.compile('(fair)\s(catch)')
fair_catch = [regexp_fair_catch.search(x) != None for x in punt_data.PlayDescription]
data["fair_catch"] = fair_catch

regexp_bounds = re.compile('(out)\s(of)\s(bounds)\.')
out_of_bounds = [regexp_bounds.search(x) != None for x in punt_data.PlayDescription]
data["out_of_bounds"] = out_of_bounds

regexp_touchback = re.compile('(Touchback)\.')
touchback = [regexp_touchback.search(x) != None for x in punt_data.PlayDescription]
data["touchback"] = touchback

regexp_muffs = re.compile('(MUFFS)|(FUMBLE)|(Fumble)|(fumble)')
muffed = [regexp_muffs.search(x) != None for x in punt_data.PlayDescription]
data["muffed"] = muffed

regexp_downed = re.compile('(downed)')
downed = [regexp_downed.search(x) != None for x in punt_data.PlayDescription]
data["downed"] = downed

regexp_blocked = re.compile('(BLOCKED)')
blocked = [regexp_blocked.search(x) != None for x in punt_data.PlayDescription]
data["blocked"] = blocked

regexp_returned = re.compile("[a-zA-Z\.]*\sto\s[A-Z]*\s[0-9]*\sfor\s[-0-9]*\s|[a-zA-Z\.]*\s(pushed)\sob\sat\s[A-Z]*\s[0-9]*\sfor\s[-0-9]*\s|[a-zA-Z\.]*\sto\s[0-9]*\sfor\s[-0-9]*\s|[a-zA-Z\.]*\sto\s[A-Z]*\s[0-9]*\sfor\sno\sgain|[a-zA-Z\.]*\sran\sob\sat\s[A-Z]*\s[0-9]*\sfor|[a-zA-Z\.]*\spushed\sob\sat\s[A-Z]*\s[0-9]*\sfor|[a-zA-Z\.]*\sfor\s[0-9]*\syards, TOUCHDOWN|[a-zA-Z\.]*\spushed\sob\sat\s[0-9]*\sfor\s|[a-zA-Z\.]*\sran\sob\sat\s[0-9]*\sfor\s")
returned = [regexp_returned.search(x) != None for x in punt_data.PlayDescription]
data["returned"] = returned & ~np.array(muffed) & ~np.array(fair_catch) & ~np.array(out_of_bounds) & ~np.array(touchback) & ~np.array(downed) & ~np.array(blocked)

### Manually Change Punts that were challenged

data.loc[0, 'downed'] = False
data.loc[155, 'downed'] = False
data.loc[161, 'downed'] = False
data.loc[661, 'downed'] = False
data.loc[1254, 'downed'] = False
data.loc[1415, 'downed'] = False
data.loc[1649, 'downed'] = False
data.loc[1745, 'downed'] = False
data.loc[3796, 'downed'] = False
count_data = pd.DataFrame({'Type of Outcome': ["returned", "fair_catch", "downed", "out_of_bounds", "touchback", "muffed", "blocked"],
                         'Proportion of Executed Punts' : np.array([sum(data.returned), sum(data.fair_catch), sum(data.downed), sum(data.out_of_bounds),
                                    sum(data.touchback), sum(data.muffed), sum(data.blocked)])/len(data)})

fig, ax = plt.subplots(figsize=(15, 10))
plt.title('Common Outcomes of a Punt')
sns.barplot(x='Type of Outcome', y='Proportion of Executed Punts', data=count_data);
fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Type of Contact that Resulted in a Concussion")
data.Primary_Impact_Type.value_counts().plot(kind='bar');
fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Action Taken by Concussed Player")
data.Player_Activity_Derived.value_counts().plot(kind='bar');
fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Occurence of Friendly Fire Resulting in a Concussion")
data.Friendly_Fire.value_counts().plot(kind='bar');
fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Outcomes of Punts Plays that Resulted in a Concussion  ")
data[data.Player_Activity_Derived.notna()].iloc[:, 21:].sum().plot(kind='bar');
regexp_ret_dist = re.compile('(for\sno\sgain)|(for\s[-0-9]*\s(yards|yard))')
ret_dist = np.array([regexp_ret_dist.search(x).group(0) for x in data[data.returned].PlayDescription])

ret_dist = np.array([re.search('[-0-9]+|no', x).group(0) for x in ret_dist])
ret_dist[ret_dist == 'no'] = 0
ret_dist = [int(x) for x in ret_dist]
print("Mean Punt Return Distance (yards): " + str(np.mean(ret_dist)))
print("Median Punt Return Distance (yards): " + str(np.median(ret_dist)))
fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Distribution of Punt Return Distance (yards)")
sns.distplot(ret_dist);
ngs_conc['unique_id'] = [i + j + k for i, j, k in zip([str(x) for x in ngs_conc.GameKey], [str(x) for x in ngs_conc.PlayID], [str(x) for x in ngs_conc.Season_Year])]
ids = ngs_conc.unique_id.unique()
line_set = ngs_conc[ngs_conc.Event == "line_set"]

### There is no "line_set" event for this punt so I use ball_snap instead to get starting formation
line_set = line_set.append(ngs_conc[(ngs_conc.Event == 'ball_snap') & (ngs_conc.unique_id == '56714072017')])

x_vals = []
y_vals = []

for i in np.arange(0, len(ids)):
    temp = line_set[line_set.unique_id == ids[i]]
    x_vals.append(temp.x.values)
    y_vals.append(temp.y.values)
    
    

ngs_control = pd.read_csv('../input/punt-data/ngs_control.csv')
control_line_set = ngs_control[ngs_control.Event == 'line_set']

### There is no "line_set" event for these punts so I use ball_snap instead to get starting formation
control_line_set = control_line_set.append(ngs_control[(ngs_control.Event == 'ball_snap') & ((ngs_control.unique_id == 42333332017) | (ngs_control.unique_id == 4276892017))])

control_ids = ngs_control.unique_id.unique()

control_x = []
control_y = []

for j in np.arange(0, len(control_ids)):
    temp = control_line_set[control_line_set.unique_id == control_ids[j]]
    control_x.append(temp.x.values)
    control_y.append(temp.y.values)
conc_paths = []

fig, ax = plt.subplots(37, 2, figsize=(15, 150))

for i in range(len(ids)):
    ax[i, 0].scatter(x_vals[i],y_vals[i])
    ax[i, 0].set_xlim(0, 120)
    ax[i, 0].set_ylim(0, 54)
    
for index, rows in injury_data.iterrows():
    temp = ngs_conc[(ngs_conc.GSISID == rows.GSISID) & (ngs_conc.unique_id == rows.unique_id)]
    conc_paths.append(temp)
    ax[index, 1].scatter(temp.x, temp.y)
    ax[index, 1].set_xlim(0, 120)
    ax[index, 1].set_ylim(0, 54)

fig, ax = plt.subplots(figsize=(8, 150))

for i in range(len(control_ids)):
    plt.subplot(37, 1, i+1)
    plt.scatter(control_x[i], control_y[i])
    plt.xlim(0, 120)
    plt.ylim(0, 54)