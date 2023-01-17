import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from collections import Counter, OrderedDict
fpath = '/kaggle/input/santa-2019-workshop-scheduling/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-2019-workshop-scheduling/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
data.head()
submission.head()
family_size_dict = data[['n_people']].to_dict()['n_people']



cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].to_dict()



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



# from 100 to 1

days = list(range(N_DAYS,0,-1))
# by key

def sort_dict(d):

    return dict(OrderedDict(sorted(d.items(), key=lambda t: t[0])))

# by value

def sort_dict_by_val(d):

    return [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
all_preferenses = []

for i in (data.index):

    all_preferenses += list(data.loc[i][cols])*data.loc[i].n_people
# make dict

count_preferenses = Counter(all_preferenses)

#sorted by keys

count_preferenses = sort_dict(count_preferenses)
dates = []

CD = datetime.datetime.strptime('2019-12-25',"%Y-%m-%d")

for d in count_preferenses.keys():

    s = datetime.datetime.strftime(CD - datetime.timedelta(days=d),"%Y-%m-%d")

    dates.append(s)

converter = dict(zip(dates,count_preferenses.keys()))

count_preferenses_by_dates = dict(zip(dates,count_preferenses.values()))
# colors = ['r','g','y','b']

# fig = plt.figure(figsize=(20,8))

# ax1 = fig.add_subplot(111)

# ax1.set_xticks(np.arange(len(count_preferenses.keys())))

# ax1.bar(count_preferenses.keys(), count_preferenses.values(), color=colors, align="center")

# ax1.set_xticklabels(count_preferenses.keys(), rotation='vertical')

# plt.xlabel('Day')

# plt.ylabel('Number of people who want to attract Santa')

# plt.show()
# throught all data

colors = ['g','g','r','y']

fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(111)

ax1.set_xticks(np.arange(len(count_preferenses.keys())))

ax1.bar(count_preferenses_by_dates.keys(), 

         count_preferenses_by_dates.values(), 

         color=colors, 

         align="center")

ax1.set_xticklabels(count_preferenses_by_dates.keys(), 

                    rotation='vertical')

plt.xlabel('Day')

plt.ylabel('Number of people who want to attract Santa')

plt.show()
# distributions preferense days by choices

d = {col:sort_dict(Counter(data[col].values)) for col in cols }
# thought choice_0

colors = ['g','g','r','y']

fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(111)

ax1.set_xticks(np.arange(len(d.keys())))

ax1.bar(d['choice_0'].keys(), 

         d['choice_0'].values(), 

         color=colors, 

         align="center")

ax1.set_xticklabels(d['choice_0'].keys(), 

                    rotation='vertical')

plt.xlabel('Day')

plt.ylabel('Number of people who want to attract Santa')

plt.show()
data.head()
# there are no columns without any blank field with  choice

data.isna().sum()
statistic_n_people = Counter(data['n_people'])

plt.bar(statistic_n_people.keys(),statistic_n_people.values(),color=['red','green'])
distrb_prefferenses_by_n_people = {}

for n_person in data.n_people.unique():

    distrb_prefferenses_by_n_people[n_person] = list(data[data.n_people == n_person].values.flatten())
plt.figure(figsize=[12,8])

sns.distplot(distrb_prefferenses_by_n_people[2])

sns.distplot(distrb_prefferenses_by_n_people[8])
# distributions preferense days by all choices

d = {col:sort_dict(Counter(data[col].values)) for col in cols }
preferense_by_num_choice = {}

for n_person in data.n_people.unique():

    temp = {}

    for col in cols:

        temp[col] = list(data[data.n_people==n_person][col].values)

    preferense_by_num_choice[n_person] = temp
# vise versa

preferense_by_choice_num = {}

for col in cols:

    temp = {}

    for n_person in data.n_people.unique():

        temp[n_person] = list(data[data.n_people==n_person][col].values)

    preferense_by_choice_num[col] = temp
# The less number of desires for each group of

ch0 = preferense_by_choice_num['choice_0']

for n in ch0.keys():

    temp = sort_dict_by_val(Counter(ch0[n]))

    print(n, dict(temp[-20:]).keys())
def cost_function(prediction):



    penalty = 0



    # We'll use this to count the number of people scheduled each day

    daily_occupancy = {k:0 for k in days}

    

    # Looping over each family; d is the day for each family f

    for f, d in enumerate(prediction):



        # Using our lookup dictionaries to make simpler variable names

        n = family_size_dict[f]

        choice_0 = choice_dict['choice_0'][f]

        choice_1 = choice_dict['choice_1'][f]

        choice_2 = choice_dict['choice_2'][f]

        choice_3 = choice_dict['choice_3'][f]

        choice_4 = choice_dict['choice_4'][f]

        choice_5 = choice_dict['choice_5'][f]

        choice_6 = choice_dict['choice_6'][f]

        choice_7 = choice_dict['choice_7'][f]

        choice_8 = choice_dict['choice_8'][f]

        choice_9 = choice_dict['choice_9'][f]



        # add the family member count to the daily occupancy

        daily_occupancy[d] += n



        # Calculate the penalty for not getting top preference

        if d == choice_0:

            penalty += 0

        elif d == choice_1:

            penalty += 50

        elif d == choice_2:

            penalty += 50 + 9 * n

        elif d == choice_3:

            penalty += 100 + 9 * n

        elif d == choice_4:

            penalty += 200 + 9 * n

        elif d == choice_5:

            penalty += 200 + 18 * n

        elif d == choice_6:

            penalty += 300 + 18 * n

        elif d == choice_7:

            penalty += 300 + 36 * n

        elif d == choice_8:

            penalty += 400 + 36 * n

        elif d == choice_9:

            penalty += 500 + 36 * n + 199 * n

        else:

            penalty += 500 + 36 * n + 398 * n



    # for each date, check total occupancy

    #  (using soft constraints instead of hard constraints)

    for _, v in daily_occupancy.items():

        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

            penalty += 100000000



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_cost = max(0, accounting_cost)

    

    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = daily_occupancy[days[0]]

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count



    penalty += accounting_cost



    return penalty
# Start with the sample submission values

best = submission['assigned_day'].tolist()

start_score = cost_function(best)



new = best.copy()

# loop over each family

for fam_id, _ in enumerate(best):

    # loop over each family choice

    for pick in range(10):

        day = choice_dict[f'choice_{pick}'][fam_id]

        temp = new.copy()

        temp[fam_id] = day # add in the new pick

        if cost_function(temp) < start_score:

            new = temp.copy()

            start_score = cost_function(new)



submission['assigned_day'] = new

score = cost_function(new)

submission.to_csv(f'submission_{score}.csv')

print(f'Score: {score}')