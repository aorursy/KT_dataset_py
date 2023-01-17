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



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")
def c_accuracy_group(df):

    df["accuracy_group"]=0

    for i in range(0,len(df)):

        acc = float(df["accuracy"][i])

        if (acc == float(0)):

            df["accuracy_group"][i]=0

        elif (acc < float(0.5)):

            df["accuracy_group"][i]=1

        elif (acc < float(1)):

            df["accuracy_group"][i]=2

        elif (acc == float(1)):

            df["accuracy_group"][i]=3

        else:

            df["accuracy_group"][i] = None

    return df
def test_to_label(test):

    print("Converting to label format, as of submissions done in assessment")

    test_ass = test[test.type == "Assessment"]

    test_ass_sub = test_ass[(((test.event_code == 4100) & (test.title != 'Bird Measurer (Assessment)'))) | (((test.event_code == 4110) & (test.title == 'Bird Measurer (Assessment)')))]

    test_ass_sub_inf = test_ass_sub[["installation_id","game_session","timestamp","title","event_data"]].sort_values(by=['installation_id','timestamp'])

    #df.sort_values(by=['col1'])

    test_ass_sub_inf0 = test_ass_sub_inf

    test_ass_sub_inf0["correct"] = 0

    test_ass_sub_inf0["incorrect"] = 0

    

    for i in range(0,len(test_ass_sub_inf0)):

        if "\"correct\":true" in test_ass_sub_inf0["event_data"][test_ass_sub_inf0.index[i]]:

            test_ass_sub_inf0["correct"][test_ass_sub_inf0.index[i]] = 1

        else:

            test_ass_sub_inf0["incorrect"][test_ass_sub_inf0.index[i]] = 1

    test_ass_sub_inf1 = test_ass_sub_inf0.groupby(by=["installation_id","game_session","title"],sort=False).sum()

    test_ass_sub_inf2 = test_ass_sub_inf1

    test_ass_sub_inf2 = test_ass_sub_inf2.reset_index()

    test_ass_sub_inf2["accuracy"] =float(0)

    

    for i in range(0,len(test_ass_sub_inf2)):

        corr = test_ass_sub_inf2["correct"][i]

        incor = test_ass_sub_inf2["incorrect"][i]

        test_ass_sub_inf2["accuracy"][i] = float(corr)/(incor+corr)

    

    test_ass_sub_inf3 = test_ass_sub_inf2

    test_ass_sub_inf3 = c_accuracy_group(test_ass_sub_inf3)

    return test_ass_sub_inf3
test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

test
test1 = test.sort_values(by=["installation_id","timestamp"])

test1
test_cic = test1[test1.event_data.str.contains("\"correct\":")]

test_cic
test_c = test_cic[test_cic.event_data.str.contains("\"correct\":true")].reset_index()

test_c
test_c["correct"] = 1
test_c1 = test_c[["installation_id","game_session","correct"]].groupby(["installation_id","game_session"]).sum().reset_index()

test_c1
test_ic = test_cic[test_cic.event_data.str.contains("\"correct\":false")].reset_index()

test_ic
test_ic["incorrect"] = 1
test_ic1 = test_ic[["installation_id","game_session","incorrect"]].groupby(["installation_id","game_session"]).sum().reset_index()

test_ic1
test_cic1 = test_cic.groupby(["world","installation_id","game_session","title","type"],sort=False).size().reset_index()

test_cic1
%%time

test_gen1 = test1.groupby(["world","installation_id","game_session","title","type"]).size().reset_index()

test_gen1
%time

test_cic2 = test_cic.groupby(['installation_id','game_session','title','world']).size().reset_index().merge(test_c1[["installation_id","game_session","correct"]],how='left',on=['installation_id','game_session'])

test_cic2 = test_cic2.merge(test_ic1[["installation_id","game_session","incorrect"]],how='left',on=['installation_id','game_session'])

test_cic2
test_cic2.isna().sum()
test_cic2['correct'].fillna(0,inplace=True)

test_cic2['incorrect'].fillna(0,inplace=True)
test_cic2
test_cic3 = test_cic2

test_cic3
test_cic3['cic_score'] = 0.0001

for i in range (0,len(test_cic3)):

    c = test_cic3['correct'][i]

    ic = test_cic3['incorrect'][i]

    test_cic3['cic_score'][i] = round(float((c*6)+(ic*0.5)),1)

test_cic3
test_last_time = test1[['installation_id','game_session','timestamp']].groupby(['installation_id','game_session'],sort=False).last().timestamp.reset_index()

test_last_time
test_cic3_time = test_cic3.merge(test_last_time,how='left',on=['installation_id','game_session'])

test_cic3_time
test_cic3_time1 = test_cic3_time.sort_values(by=['installation_id','timestamp'])

test_cic3_time1
test_cic2_cum = test_cic3_time1.groupby(['installation_id','game_session','title'],sort=False)[['correct','incorrect','cic_score']].sum().groupby(level=[0]).cumsum().reset_index()

test_cic2_cum
test_cic2_cum_time = test_cic2_cum.merge(test1.groupby(['installation_id','game_session'],sort=False).last().timestamp,how='inner',on=['installation_id','game_session'])

test_cic2_cum_time
test_cic2_cum_time1 = test_cic2_cum_time

test_cic2_cum_time1
test_cic2_cum_time1['acc_r'] = 0.0001

test_cic2_cum_time1['inacc_r'] = 0.0001

for i in range(0,len(test_cic2_cum_time1)):

    c = float(test_cic2_cum_time1['correct'][i])

    ic = float(test_cic2_cum_time1['incorrect'][i])

    if (c==0):

        test_cic2_cum_time1['acc_r'][i] = 0

        test_cic2_cum_time1['inacc_r'][i] = int(ic)

    elif (ic==0):

        test_cic2_cum_time1['acc_r'][i] = int(c)

        test_cic2_cum_time1['inacc_r'][i] = 0

    else:

        test_cic2_cum_time1['acc_r'][i] = round((float(c)/int(ic)),3)

        test_cic2_cum_time1['inacc_r'][i] = round((float(ic)/int(c)),3)

test_cic2_cum_time1
test_labels = test_to_label(test1)

test_labels
test_labels_time = test_labels.merge(test1.groupby(['installation_id','game_session'],sort=False).last().timestamp,how='inner',on=['installation_id','game_session'])

test_labels_time
test_labels_time1 = test_labels_time

test_labels_time1
test_labels_time1 = test_labels_time1.sort_values(by=['installation_id','timestamp'])

test_labels_time1
test_labels_time.equals(test_labels_time1)
test.groupby(["world","title"]).size()
ts = test.groupby(['installation_id']).size()

ts
ts[ts>13000]
tse = test[test.installation_id == '7b728c89'][test.world == 'TREETOPCITY'].sort_values(by=['timestamp'])['title']

tse
order = test.sort_values(by=['timestamp'])[test.world == 'TREETOPCITY'].groupby(['title','type'],sort=False).size()

order
 

TREE_DICT = {'Tree Top City - Level 1':15,'Tree Top City - Level 2':16,'Tree Top City - Level 3':17,'Ordering Spheres':1,



'All Star Sorting':2,



'Costume Box':3,



'Fireworks (Activity)':4,



'12 Monkeys':5,



'Flower Waterer (Activity)':6,



'Pirate\'s Tale':7,



'Mushroom Sorter (Assessment)':8,



'Air Show':9,



'Treasure Map':10,



'Crystals Rule':11,



'Rulers':12,



'Bug Measurer (Activity)':13,



'Bird Measurer (Assessment)':14}
TREE_DICT
test_TREE = test[test.world == 'TREETOPCITY'].reset_index()

test_TREE
test_TREE.title.unique()
len(test_TREE.title.unique())
test_TREE1 = test_TREE

test_TREE1
test_TREE1['title1'] = test_TREE['title'].map(TREE_DICT)

test_TREE1[['title','title1']]
TREE_DICT
TREE_DICT_score = {'Tree Top City - Level 1': 0,

 'Tree Top City - Level 2': 0,

 'Tree Top City - Level 3': 0,

 'Ordering Spheres': 1,

 'All Star Sorting': 2,

 'Costume Box': 3,

 'Fireworks (Activity)': 4,

 '12 Monkeys': 5,

 'Flower Waterer (Activity)': 6,

 "Pirate's Tale": 7,

 'Mushroom Sorter (Assessment)': 8,

 'Air Show': 9,

 'Treasure Map': 10,

 'Crystals Rule': 11,

 'Rulers': 12,

 'Bug Measurer (Activity)': 13,

 'Bird Measurer (Assessment)': 14}
TREE_DICT_score
test_TREE1_time = test_TREE1.sort_values(by=['installation_id','timestamp'])

test_TREE1_time[['installation_id','title','timestamp']]
test_TREE_labels = test_to_label(test_TREE1_time)

test_TREE_labels
test_TREE1_g = test_TREE1_time.groupby(['installation_id','game_session','title'],sort=False).size().reset_index()

test_TREE1_g
test_TREE1_g1 = test_TREE1_g

test_TREE1_g1
test_cic3_time1
test_cic3_time1_TREE = test_cic3_time1[test_cic3_time1.world == 'TREETOPCITY'].reset_index()

test_cic3_time1_TREE
test_cic3_time1_TREE['base_score'] = test_cic3_time1_TREE['title'].map(TREE_DICT_score)

test_cic3_time1_TREE
test_cic3_time1_TREE1 = test_cic3_time1_TREE

test_cic3_time1_TREE1
%%time

test_cic3_time1_TREE1['base_corr'] = 0.0001

test_cic3_time1_TREE1['base_incorr'] = 0.0001

test_cic3_time1_TREE1['base_cic_score'] = 0.0001

for i in range(0,len(test_cic3_time1_TREE1)):

    c = test_cic3_time1_TREE1['correct'][i]

    ic = test_cic3_time1_TREE1['incorrect'][i]

    cics = test_cic3_time1_TREE1['cic_score'][i]

    base = test_cic3_time1_TREE1['base_score'][i]

    

    test_cic3_time1_TREE1['base_corr'][i] = int(c*base)

    test_cic3_time1_TREE1['base_incorr'][i] = int(ic*base)

    test_cic3_time1_TREE1['base_cic_score'][i] = round(cics*base,3)

test_cic3_time1_TREE1

    
test_cic3_time1_TREE1['count'] = 1
test_cic3_time1_TREE1_cum = test_cic3_time1_TREE1.groupby(['installation_id','game_session','title','world','timestamp'],sort=False).sum().groupby(level=[0]).cumsum().reset_index()

test_cic3_time1_TREE1_cum
test_cic3_time1_TREE1_cum1 = test_cic3_time1_TREE1_cum

test_cic3_time1_TREE1_cum1
test_cic3_time1_TREE1_cum1['acc_r'] = 0.0001

test_cic3_time1_TREE1_cum1['inacc_r'] = 0.0001

for i in range(0,len(test_cic3_time1_TREE1_cum1)):

    c = float(test_cic3_time1_TREE1_cum1['correct'][i])

    ic = float(test_cic3_time1_TREE1_cum1['incorrect'][i])

    if (c==0):

        test_cic3_time1_TREE1_cum1['acc_r'][i] = 0

        test_cic3_time1_TREE1_cum1['inacc_r'][i] = int(ic)

    elif (ic==0):

        test_cic3_time1_TREE1_cum1['acc_r'][i] = int(c)

        test_cic3_time1_TREE1_cum1['inacc_r'][i] = 0

    else:

        test_cic3_time1_TREE1_cum1['acc_r'][i] = round((float(c)/int(ic)),3)

        test_cic3_time1_TREE1_cum1['inacc_r'][i] = round((float(ic)/int(c)),3)

test_cic3_time1_TREE1_cum1
test_cic3_time1_TREE1_cum1['base_acc_r'] = 0.0001

test_cic3_time1_TREE1_cum1['base_inacc_r'] = 0.0001

for i in range(0,len(test_cic3_time1_TREE1_cum1)):

    c = float(test_cic3_time1_TREE1_cum1['base_corr'][i])

    ic = float(test_cic3_time1_TREE1_cum1['base_incorr'][i])

    if (c==0):

        test_cic3_time1_TREE1_cum1['base_acc_r'][i] = 0

        test_cic3_time1_TREE1_cum1['base_inacc_r'][i] = int(ic)

    elif (ic==0):

        test_cic3_time1_TREE1_cum1['base_acc_r'][i] = int(c)

        test_cic3_time1_TREE1_cum1['base_inacc_r'][i] = 0

    else:

        test_cic3_time1_TREE1_cum1['base_acc_r'][i] = round((float(c)/int(ic)),3)

        test_cic3_time1_TREE1_cum1['base_inacc_r'][i] = round((float(ic)/int(c)),3)

test_cic3_time1_TREE1_cum1
test_cic3_time1_TREE1_cum1['average_score'] = test_cic3_time1_TREE1_cum1['cic_score']/test_cic3_time1_TREE1_cum1['count']

test_cic3_time1_TREE1_cum1['average_score']
test_cic3_time1_TREE1_cum1['average_corr'] = test_cic3_time1_TREE1_cum1['correct']/test_cic3_time1_TREE1_cum1['count']

test_cic3_time1_TREE1_cum1['average_corr']
test_cic3_time1_TREE1_cum1['average_incorr'] = test_cic3_time1_TREE1_cum1['incorrect']/test_cic3_time1_TREE1_cum1['count']

test_cic3_time1_TREE1_cum1['average_incorr']
test_cic3_time1_TREE1_cum1['base_average_score'] = test_cic3_time1_TREE1_cum1['base_cic_score']/test_cic3_time1_TREE1_cum1['count']

test_cic3_time1_TREE1_cum1['base_average_score']
test_cic3_time1_TREE1_cum1['base_average_corr'] = test_cic3_time1_TREE1_cum1['base_corr']/test_cic3_time1_TREE1_cum1['count']

test_cic3_time1_TREE1_cum1['base_average_corr']
test_cic3_time1_TREE1_cum1['base_average_incorr'] = test_cic3_time1_TREE1_cum1['base_incorr']/test_cic3_time1_TREE1_cum1['count']

test_cic3_time1_TREE1_cum1['base_average_incorr']
test_TREE_labels
test_cic3_time1_TREE1_cum1_labels = test_TREE_labels[['installation_id','game_session','accuracy_group']].merge(test_cic3_time1_TREE1_cum1,how='left',on=['installation_id','game_session'])

test_cic3_time1_TREE1_cum1_labels
test_bird = test_cic3_time1_TREE1_cum1_labels[test_cic3_time1_TREE1_cum1_labels.title == 'Bird Measurer (Assessment)'].reset_index(drop=True)

test_mush = test_cic3_time1_TREE1_cum1_labels[test_cic3_time1_TREE1_cum1_labels.title == 'Mushroom Sorter (Assessment)'].reset_index(drop=True)
test_bird
test_mush
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_mush[["correct","average_corr","incorrect","average_incorr","accuracy_group"]])
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_mush[["acc_r","inacc_r","base_cic_score","cic_score","accuracy_group"]])
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_mush[["average_score","cic_score","base_average_score","base_cic_score","accuracy_group"]])
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_mush[["base_acc_r","base_inacc_r","base_cic_score","cic_score","accuracy_group"]])
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_bird[["correct","incorrect","base_corr","base_incorr","accuracy_group"]])
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_bird[["acc_r","inacc_r","base_cic_score","cic_score","accuracy_group"]])
import seaborn as sns; sns.set(style="ticks", color_codes=True)

sns.pairplot(test_bird[["base_acc_r","base_inacc_r","base_cic_score","cic_score","accuracy_group"]])
test1
test1.columns
test1_TREE_gametime = test1[test1.world == 'TREETOPCITY'].groupby(['installation_id','game_session','title','type'],sort=False).last().game_time.reset_index()

test1_TREE_gametime
test1_TREE_gametime_sec = test1_TREE_gametime

test1_TREE_gametime_sec
test1_TREE_gametime_sec['game_time'] = test1_TREE_gametime['game_time']/1000

test1_TREE_gametime_sec
test1_TREE_gametime_sec.groupby(['type','title'])['game_time'].max()
test1
test1_TREE_clips = test1[test1.world == 'TREETOPCITY'][test1.type == 'Clip'].reset_index()

test1_TREE_clips
test1_TREE_clips[test1_TREE_clips.event_code != 2000]
test[test.type == 'Clip'].event_code.unique()
specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")

specs
type(specs['info'])
type(specs['info'][0])
specs_clip = specs[((specs['info'].str.contains("video")) | (specs['info'].str.contains("clip")))].reset_index()

specs_clip
specs_clip['info'][0]
specs_clip['args'][0]
specs_qs = specs[((specs['info'].str.contains("quit")) | (specs['info'].str.contains("skip")))].reset_index()

specs_qs
for i in range(0,len(specs_qs)):

    print(i)

    print(specs_qs['event_id'][i])

    print(specs_qs['info'][i])

    print()
test[test.type=='Clip'].event_id.unique()
specs[specs.event_id == '27253bdc']
specs[specs.event_id == '27253bdc'].reset_index()['info'][0]
specs[specs.event_id == '27253bdc'].reset_index()['args'][0]
test[test.type=='Clip'].title.unique()
test_end_movie = test[test.event_id == 'c189aaf2'].reset_index()

test_end_movie
for i in range(0, 20):

    print(test_end_movie['title'][i]+" "+test_end_movie['world'][i]+" "+test_end_movie['type'][i])

    print(test_end_movie['event_data'][i])

    print()
test_first_timestamp = test1.groupby(['installation_id','game_session','title','world','type'],sort=False).first().timestamp.reset_index()

test_first_timestamp
import pandas as pd

pd.set_option("display.max_rows",100)

test_first_timestamp.head(100)
test_first_timestamp['timestamp'][1]
test_first_timestamp['timestamp'][1][0:-1]
test_first_timestamp['timestamp'][1][0:-5]
test_first_timestamp['timestamp1'] = pd.to_datetime(test_first_timestamp['timestamp'])

test_first_timestamp
test_first_timestamp['timestamp1'][1]-test_first_timestamp['timestamp1'][0]
%%time

test_first_timestamp['time_diff'] = 0

for i in range(0,len(test_first_timestamp)-1):

    test_first_timestamp['time_diff'][i] = test_first_timestamp['timestamp1'][i+1]-test_first_timestamp['timestamp1'][i]

test_first_timestamp    