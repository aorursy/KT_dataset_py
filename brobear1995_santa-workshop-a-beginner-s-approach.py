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
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

plt.figure(figsize=(15, 15))

sns.set_style('whitegrid')

sns.set_context('paper')



df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

print('Santa must attend {} families with over {} people!'.format(df.shape[0], df.n_people.sum()))

print('Let\'s look at the preference list')

display(df.head())
demand = [0 for _ in range(101)]

def add_demand(x):

    for i in range(0,10):

        y = x.at['choice_'+str(i)]

        demand[y] = demand[y] + x.at['n_people']

    return x

df = df.apply(lambda x: add_demand(x), axis=1)

demand = demand[1:]

demand = [float(x)/sum(demand) * 100 for x in demand]

demand_dict = dict(enumerate(demand, 1))

sns.scatterplot(y=demand, x=list(range(1,101)))
from scipy import stats

avg_weekly_demand = np.average(np.array(demand[:98]).reshape(-1,7), axis=1)

y=np.array(list(range(1,15)))

slope, intercept, r_value, p_value, std_err = stats.linregress(y, avg_weekly_demand)



sns.lineplot(y, avg_weekly_demand)

ax = sns.regplot(y, avg_weekly_demand, color='b', 

 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

ax.legend()
new_demand = (avg_weekly_demand-np.min(avg_weekly_demand))/(np.max(avg_weekly_demand)-np.min(avg_weekly_demand))

new_weekly_demand = new_demand *(300-125) + 125

slope, intercept, r_value, p_value, std_err = stats.linregress(y, new_weekly_demand)

sns.lineplot(y, new_weekly_demand)

ax = sns.regplot(y, new_weekly_demand, color='b', 

 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

ax.legend()
week1=np.array(demand[:7])

sns.lineplot(np.array(list(range(1,8))),week1,  label='Week 1')

week2=np.array(demand[7:14])

sns.lineplot(np.array(list(range(1,8))),week2, label='Week 2')

week3=np.array(demand[14:21])

sns.lineplot(np.array(list(range(1,8))),week3, label='Week 3')

week4=np.array(demand[21:28])

sns.lineplot(np.array(list(range(1,8))),week4, label='Week 4')

week5=np.array(demand[28:35])

sns.lineplot(np.array(list(range(1,8))),week5, label='Week 5')

week6=np.array(demand[35:42])

sns.lineplot(np.array(list(range(1,8))),week6, label='Week 6')

week7=np.array(demand[42:49])

sns.lineplot(np.array(list(range(1,8))),week7, label='Week 7')
week1=np.array(demand[:7])

sns.lineplot(np.array(list(range(1,8))),week1,  label='Week 1')
difference = []

for i in range(1,100):

    difference.append(demand[i]-demand[i-1])

abs_difference = np.abs(np.array(difference))

abs_difference = (abs_difference-np.min(abs_difference))/((np.max(abs_difference) - np.min(abs_difference))) * 50

abs_difference = [int(x) for x in abs_difference]

final_fluctuation = [abs_difference[i] * (2*(difference[i] >=0) -1) for i in range(len(abs_difference))]

days_start = [300]

print('{} number of people would attend Day {}'.format(300,1))

for i in range(15):

    num_people_week = int(round(-10*i+274,0))

    num_people_week=min(num_people_week,300)

    num_people_week = max(num_people_week,130)

    print('{} number of people would attend Day {}'.format(num_people_week, i*7+2))

    days_start.append(num_people_week)

    for j in range(1,7):

        if i*7+j >=99:

            break

        people_today_fluctuation = final_fluctuation[i*7+j]

        num_people_week += people_today_fluctuation

        num_people_week=min(num_people_week,300)

        num_people_week = max(num_people_week,130)

        num_people_week = int(round(num_people_week,0))

        print('{} number of people would attend Day {}'.format(num_people_week, i*7+j+2))

        days_start.append(num_people_week)

sns.lineplot(list(range(1, 101)), days_start)
assert sum(days_start)/len(days_start) >= df.n_people.sum()/df.shape[0]
def calculate_eagerness(x):

    eagerness = 0

    for i in range(0,9):

        eagerness = eagerness + (9-i)*demand_dict[x.at['choice_'+str(i)]]

        eagerness = eagerness + 0.1 * x.at['n_people']

    return round(eagerness,0)

df['eagerness'] = df.apply(lambda x: calculate_eagerness(x), axis=1)
max_cap = days_start

people_attending = [0 for i in range(0,100)]

demand_arr = np.array(demand)

df['assigned'] = 0

df = df.sort_values(by='eagerness', ascending=False)

demand_dict = dict(zip(list(range(1,101)), demand_arr))

low_threshold = 130



def assign_family(x):

    global low_threshold

    assigned = int(x.at['choice_0']) - 1

    i = 0

    a = max_cap[assigned] - x.at['n_people']

    b = people_attending[assigned] + x.at['n_people']

    while a < 0:

        i = i + 1

        if i >= 10:

            assigned = -100

            break

        else:

            assigned = int(x.at['choice_'+str(i)]) -1

        a = max_cap[assigned] - x.at['n_people']

    if assigned != -100:

        #print('Family {} got their {} preference'.format(x.at['family_id'], i+1))

        max_cap[assigned] -= x.at['n_people']

        people_attending[assigned] += x.at['n_people']

        x['assigned'] = assigned + 1

    else:

        #Populate the days which has less people attending

        low_days = np.where(np.array(people_attending) <= low_threshold)[0]

        while len(low_days) == 0:

            low_threshold += 10

            low_days = np.where(np.array(people_attending) <= low_threshold)[0]

        low_days = low_days + 1

        low_days = sorted(low_days, key=demand_dict.get)

        for low_day in low_days:

            assigned = low_day

            a = max_cap[assigned - 1] - x.at['n_people']

            b = people_attending[assigned - 1] + x.at['n_people']

            if a < 0:

                continue

            else:

                break

        #print('Family {} got alternate day!!! Their appointment is at {}'.format(x.at['family_id'], assigned))

        x['assigned'] = assigned

        max_cap[assigned -1] -= x.at['n_people']

        people_attending[assigned -1] += x.at['n_people']   

    return x
df = df.apply(lambda x: assign_family(x), axis=1)

df['assigned'] = df['assigned'].astype(int)

df.head()
for i in range(1,101):

    no_of_people = np.sum(df[df.assigned==i]['n_people'].values)

    if (no_of_people >= 125 and no_of_people <= 300):

        pass

    else:

        print(i, no_of_people)
df = df.sort_values(by='family_id')

df2 = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')

df2['assigned_day'] = df['assigned']

df2.to_csv('sample_sub.csv', index=False)