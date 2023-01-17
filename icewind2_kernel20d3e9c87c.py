import pandas as pd

import random

import matplotlib.pyplot as plt

import numpy as np

import json
# random sample

filename = "../input/data-science-bowl-2019/train.csv"

n = sum(1 for line in open(filename)) - 1

s = 1000000 #desired sample size

skip = sorted(random.sample(range(1,n+1),n-s))



train_data = pd.read_csv(filename, skiprows=skip)

sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test_data = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
train_data.head()
users = train_data['installation_id'].drop_duplicates()

print('unique users: {}'.format(users.size))

attempted_users = train_data[train_data['type']=='Assessment'][['installation_id']].drop_duplicates() 

print('users, who attempted assessments: {}'.format(attempted_users.size))

train_data = pd.merge(train_data, attempted_users, on="installation_id", how="inner")
names = []

values = []

type_count = train_data.groupby('type').count()

for t in train_data['type'].drop_duplicates():

    names.append(t)

    values.append(len(train_data[train_data.type == t]))



fig = plt.figure(figsize=(8, 5))

plt.bar(names, values)

plt.title('Number of events by type')

plt.show()
names = []

values = []

for t in train_data['title'].drop_duplicates():

    names.append(t)

    values.append(len(train_data[train_data.title == t]))



fig = plt.figure(figsize=(13, 15))

plt.barh(names, values)

plt.title('Number of events by title')

plt.show()
train_data.world.drop_duplicates()
print('MAGMAPEAK - {}\n'.format(pd.unique(train_data[(train_data.world == 'MAGMAPEAK') & (train_data.type == 'Assessment')].title)))

print('CRYSTALCAVES - {}\n'.format(pd.unique(train_data[(train_data.world == 'CRYSTALCAVES') & (train_data.type == 'Assessment')].title)))

print('TREETOPCITY - {}\n'.format(pd.unique(train_data[(train_data.world == 'TREETOPCITY') & (train_data.type == 'Assessment')].title)))
names = []

values = []

type_count = train_data.groupby('world').count()

for t in train_data['world'].drop_duplicates():

    names.append(t)

    values.append(len(train_data[train_data.world == t]))



fig = plt.figure(figsize=(8, 5))

plt.bar(names, values)

plt.title('Number of events by world')

plt.show()
train_data[train_data.event_code == 4100].title.drop_duplicates()
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

train_data['weekday'] = train_data['timestamp'].dt.dayofweek

train_data['hour'] = train_data['timestamp'].dt.hour
fig = plt.figure(figsize=(12, 8))

names = ['Mon', 'Tue', 'Wd', 'Thu', 'Fri', 'Sat', 'Sun']

values = []

for d in range(7):

    values.append(len(train_data[train_data.weekday == d]))

plt.bar(names, values)

plt.title('Event count by weekday')

plt.show()
fig = plt.figure(figsize=(14, 9))

names = range(24)

values = []

for h in range(24):

    values.append(len(train_data[train_data.hour == h]))

plt.bar(names, values, width=0.5)

plt.title('Event count by hour')

plt.xticks(range(24))

plt.show()
# fig = plt.figure(figsize=())

time_by_session = train_data[['game_session', 'world', 'game_time']].groupby(['game_session', 'world']).max()



attempted_users

playtime = []

for u in attempted_users['installation_id']:

    time_by_world = {'MAGMAPEAK':0,

                'TREETOPCITY':0,     

                 'CRYSTALCAVES':0,

                 'NONE':0}

    sessions_by_user = train_data[train_data.installation_id == u]['game_session'].drop_duplicates()

    for s in sessions_by_user:

        tmp = time_by_session.loc[s]['game_time'].iloc[0]

        time_by_world[time_by_session.loc[s].index.to_list()[0]] += tmp

    playtime.append(time_by_world)
fig = plt.figure(figsize=(15,8))

plt.plot([u['MAGMAPEAK'] + u['TREETOPCITY'] + u['CRYSTALCAVES'] for u in playtime])

plt.title('Users playtime')

plt.show()
fig = plt.figure(figsize=(8, 5))

val = [0, 0, 0]

for u in playtime:

    val[0] += u['MAGMAPEAK']

    val[1] += u['TREETOPCITY']

    val[2] += u['CRYSTALCAVES']

plt.bar(['MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES'], val)

plt.show()
train_labels.head(9)
train_labels[['installation_id', 'accuracy_group']].groupby(['accuracy_group']).count().plot.bar(figsize=(10, 6))

plt.show()
tasks = pd.unique(train_labels.title)

mean_wrong = []

for t in tasks:

    mean_wrong.append(train_labels[train_labels.title == t].num_incorrect.mean())

fig = plt.figure(figsize=(7, 7))

plt.pie(mean_wrong, labels=tasks)

plt.show()
def create_features(data):

    global attempted_users, playtime

    labels = ['activities', 'games', 'clips', 'assessments', 'mean_activity_daytime', 'mean_game_daytime', 'mean_clip_daytime', 'mean_assessment_daytime', 'MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES', 'Cauldron_Filler_failed', 'Cauldron_Filler_success', 'Cart_Balancer_failed', 'Cart_Balancer_success', 'Chest_Sorter_failed', 'Chest_Sorter_success','Mushroom_Sorter_failed', 'Mushroom_Sorter_success', 'Bird_Measurer_failed', 'Bird_Measurer_success']

    result = pd.DataFrame(columns=labels)

    

    for i, u in enumerate(attempted_users.installation_id):

        tmp = pd.DataFrame(columns=labels)

        cur_user = data[data.installation_id == u]

        

        sub = cur_user[cur_user.type == 'Activity']

        tmp['activities'] = pd.Series(len(sub))

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        if len(m) > 0:

            tmp['mean_activity_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_activity_daytime'] = pd.Series(None)

        

        sub = cur_user[cur_user.type == 'Game']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['games'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_game_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_game_daytime'] = pd.Series(None)

            

        sub = cur_user[cur_user.type == 'Clip']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['clips'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_clip_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_clip_daytime'] = pd.Series(None)

            

        sub = cur_user[cur_user.type == 'Assessment']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['assessments'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_assessment_daytime'] = pd.Series(m[0][0]) 

        else:

            tmp['mean_assessment_daytime'] = pd.Series(None)

        

        tmp['MAGMAPEAK'] = pd.Series(playtime[i]['MAGMAPEAK'])

        tmp['TREETOPCITY'] = pd.Series(playtime[i]['TREETOPCITY'])

        tmp['CRYSTALCAVES'] = pd.Series(playtime[i]['CRYSTALCAVES'])

       

        sub = cur_user[((cur_user.event_code == 4100) & (cur_user.title != 'Bird Measurer (Assessment)') & (cur_user.type == 'Assessment')) | ((cur_user.event_code == 4110) & (cur_user.title == 'Bird Measurer (Assessment)') & (cur_user.type == 'Assessment'))]

        tmp.loc[:, 'Cauldron_Filler_failed':] = np.ndarray(10)

        for i, r in sub.iterrows():

            if json.loads(r.event_data)['correct']:

                if r.title == 'Cauldron Filler (Assessment)':

                    tmp.Cauldron_Filler_success += 1

                elif r.title == 'Cart Balancer (Assessment)':

                    tmp.Cart_Balancer_success += 1

                elif r.title == 'Chest Sorter (Assessment)':

                    tmp.Chest_Sorter_success += 1

                elif r.title == 'Mushroom Sorter (Assessment)':

                    tmp.Mushroom_Sorter_success += 1

                elif r.title == 'Bird Measurer (Assessment)':

                    tmp.Bird_Measurer_success += 1

            else:

                if r.title == 'Cauldron Filler (Assessment)':

                    tmp.Cauldron_Filler_failed += 1

                elif r.title == 'Cart Balancer (Assessment)':

                    tmp.Cart_Balancer_failed += 1

                elif r.title == 'Chest Sorter (Assessment)':

                    tmp.Chest_Sorter_failed += 1

                elif r.title == 'Mushroom Sorter (Assessment)':

                    tmp.Mushroom_Sorter_failed += 1

                elif r.title == 'Bird Measurer (Assessment)':

                    tmp.Bird_Measurer_failed += 1

        

        result = result.append(tmp, ignore_index=True)

        

    return result
features = create_features(train_data)
features.head()