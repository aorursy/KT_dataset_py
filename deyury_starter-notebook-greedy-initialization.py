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
DAILY_QUOTA = 250



daily_quotas = N_DAYS * [DAILY_QUOTA]

assigned_days = len(data)*[-1]
non_assigned_families = []



for i in data.iterrows():

    n_people = i[1]['n_people']

    assigned = False

    for j in range(9):

        if daily_quotas[i[1][f'choice_{j}'] - 1] > n_people: 

            daily_quotas[i[1][f'choice_{j}'] - 1] -= n_people

            assigned_days[i[0]] = i[1][f'choice_{j}']

            assigned = True

            break

            

    if not assigned:

        non_assigned_families.append(i[0])

for family_id in non_assigned_families:

    day_id = np.argsort(daily_quotas)[-1]

    daily_quotas[day_id] -= data.iloc[family_id].n_people

    assigned_days[family_id] = day_id + 1
submission['assigned_day'] = assigned_days
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



        

    #print ("PA: ", penalty, accounting_cost)

    penalty += accounting_cost



    return penalty
def partial_accounting_cost_helper(visitors_per_day, idx):

    res = 0



    nd = visitors_per_day[idx]

    if idx != 99:

        ndplus = visitors_per_day[idx + 1]

    else:

        ndplus = nd

    

    res += (nd - 125) / 400. * (nd ** (0.5 + abs(nd - ndplus)/50.))

    

    if idx != 0:

        ndminus = visitors_per_day[idx - 1]

        res += (ndminus - 125) / 400. * (ndminus ** (0.5 + abs(ndminus - nd)/50.))

    

    return res



def partial_accounting_cost(visitors_per_day, old_day, new_day):

    res = partial_accounting_cost_helper(visitors_per_day, old_day) + partial_accounting_cost_helper(visitors_per_day, new_day)

    

    #if abs(old_day - new_day) == 1:

    #    d = max(old_day, new_day)

    #    res += (visitors_per_day[d - 1] - 125) / 400. * (visitors_per_day[d - 1] ** (0.5 + abs(visitors_per_day[d] - visitors_per_day[d - 1])/50.))

        

    return res



def partial_penalty_cost(family_id, day):

    family_data = data.iloc[family_id]

    n = family_data.n_people

    d = day + 1



    # Calculate the penalty for not getting top preference

    if d == family_data.choice_0:

        penalty = 0

    elif d == family_data.choice_1:

        penalty = 50

    elif d == family_data.choice_2:

        penalty = 50 + 9 * n

    elif d == family_data.choice_3:

        penalty = 100 + 9 * n

    elif d == family_data.choice_4:

        penalty = 200 + 9 * n

    elif d == family_data.choice_5:

        penalty = 200 + 18 * n

    elif d == family_data.choice_6:

        penalty = 300 + 18 * n

    elif d == family_data.choice_7:

        penalty = 300 + 36 * n

    elif d == family_data.choice_8:

        penalty = 400 + 36 * n

    elif d == family_data.choice_9:

        penalty = 500 + 36 * n + 199 * n

    else:

        penalty = 500 + 36 * n + 398 * n



    return penalty
# Start with the sample submission values

best = submission['assigned_day'].tolist()

visitors_per_day = submission.merge(data.n_people, on="family_id").groupby('assigned_day').n_people.sum().tolist()



score = cost_function(best)

new = None

# loop over each family

for fam_id, _ in enumerate(best):

    # loop over each family choice

    n_people = data.iloc[fam_id].n_people

    for pick in range(10):

        new_day = choice_dict[f'choice_{pick}'][fam_id] - 1

        old_day = best[fam_id] - 1

        new_score = score



        if old_day == new_day:

            break

            

        if visitors_per_day[old_day] - n_people < 125 or visitors_per_day[new_day] + n_people > 300:

            break

        

        new_score -= partial_penalty_cost(fam_id, old_day)

        new_score -= partial_accounting_cost(visitors_per_day, old_day, new_day)

        

        visitors_per_day[old_day] -= n_people

        visitors_per_day[new_day] += n_people

            

        new_score += partial_penalty_cost(fam_id, new_day)

        new_score += partial_accounting_cost(visitors_per_day, old_day, new_day)

   

        if abs(new_day - old_day) == 1 or old_dat:

            temp = best.copy()

            temp[fam_id] = new_day + 1 # add in the new pick

            new_score = cost_function(temp)



        if new_score < score:

            score = new_score

            best[fam_id] = new_day + 1

            print (score, cost_function(best), old_day, new_day)

        else:

            visitors_per_day[old_day] += n_people

            visitors_per_day[new_day] -= n_people

        

submission['assigned_day'] = best

#score = cost_function(new)



#submission.to_csv(f'submission_{score}.csv')

#print(f'Score: {score}')