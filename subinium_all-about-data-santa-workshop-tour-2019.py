# default

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



# util

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
data.head()
fig, ax = plt.subplots(5, 2, figsize=(30,20))



for i in range(10):

    sns.countplot(data[f'choice_{i}'], ax=ax[i//2][i%2])



plt.show()
fig, ax = plt.subplots(1,1, figsize=(12,5))

sns.countplot(data['n_people'], ax=ax)

plt.show()
fig, ax = plt.subplots(5, 2, figsize=(30,20))



data_people = data.groupby('n_people')

for i in range(10):

    sns.heatmap(data_people[f'choice_{i}'].value_counts().unstack().fillna(0), ax=ax[i//2][i%2])

plt.show()

corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 14))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
family_size_dict = data[['n_people']].to_dict()['n_people']



cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].to_dict()



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



# from 100 to 1

days = list(range(N_DAYS,0,-1))



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



    return penalty+accounting_cost, penalty, accounting_cost
# Start with the sample submission values

best = submission['assigned_day'].tolist()

start_score = cost_function(best)



new = best.copy()

# loop over each family

penalty_list, cost_list = [], [] 



for fam_id, _ in enumerate(best):

    # loop over each family choice

    for pick in range(10):

        day = choice_dict[f'choice_{pick}'][fam_id]

        temp = new.copy()

        temp[fam_id] = day # add in the new pick

        if cost_function(temp)[0] < start_score[0]:

            new = temp.copy()

            start_score = cost_function(new)

            penalty_list.append(start_score[1])

            cost_list.append(start_score[2])
fig = go.Figure()

fig.add_trace(go.Line(x=list(range(len(penalty_list))), y=penalty_list, name='Preference cost',  marker_color="forestgreen"  ))

fig.update_layout(title="Prefrence Cost")

fig.show()

fig = go.Figure()

fig.add_trace(go.Line(x=list(range(len(cost_list))), y=cost_list, name='Accounting cost', marker_color="salmon" ))

fig.update_layout(title="Accounting Cost")

fig.show()

fig = go.Figure()

fig.add_trace(go.Line(x=list(range(len(penalty_list))), y=penalty_list, name='Preference cost',  marker_color="forestgreen"  ))

fig.add_trace(go.Line(x=list(range(len(cost_list))), y=cost_list, name='Accounting cost',  marker_color="salmon"  ))



fig.update_layout(title="Compare 2 Cost : Preference & Accounting")

fig.show()

            