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
fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath)
fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath)
def get_choice_cost(choice, n):

    if choice == 0:

        return 0

    elif choice == 1:

        return 50

    elif choice == 2:

        return 50+9*n

    elif choice == 3:

        return 100+9*n

    elif choice == 4:

        return 200+9*n

    elif choice == 5:

        return 200+18*n

    elif choice == 6:

        return 300+18*n

    elif choice == 7:

        return 300+36*n

    elif choice == 8:

        return 400+36*n

    elif choice == 9:

        return 500+(36+199)*n

    else:

        return 500+434*n
def get_weights():

    #dff = CSV.read("family_data.csv")

    



    # preference cost matrix

    preference_mat = np.zeros((5000,100))

   # preference_mat = preference_mat.astype(int)

    # first fill every column with the maximum cost for that family

    for ind,row in data.iterrows():

        preference_mat[ind] = get_choice_cost(10, row['n_people'])

    for ind,row in data.iterrows():

        for i in range(10):

            

            

            choice =  row[f'choice_{i}']

            choice_cost = get_choice_cost(i, row['n_people'])

            preference_mat[ind,(choice-1)] = choice_cost

#             for k in range(((choice-1)*50),(choice)*50):

#                 preference_mat[ind,k] = choice_cost

    #preference_mat = preference_mat.astype(int)

    return preference_mat

                

                    

                    





weights = get_weights()

weights.shape
weights = weights.reshape(5000,100,1)

weights = np.tile(weights,(1,1,50))

weights = weights.reshape(5000,5000)
!pip install lap
from lap import lapjv

least_cost, col, row = lapjv(weights)
least_cost
submission['assigned_day'] = col//50 +1
# family_size_dict = data[['n_people']].to_dict()['n_people']

# cols = [f'choice_{i}' for i in range(10)]

# choice = np.array(data[cols])

# choice_dict = data[cols].to_dict()

# # print(choice_dict)



# N_DAYS = 100

# MAX_OCCUPANCY = 300

# MIN_OCCUPANCY = 125



# # from 100 to 1

# days = list(range(N_DAYS,0,-1))



# family_size_ls = list(family_size_dict.values())

# choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]



# def cost_function(prediction):

#     penalty = 0



#     # We'll use this to count the number of people scheduled each day

#     daily_occupancy = {k: 0 for k in days}

#     for f, d in enumerate(prediction):

#         n = family_size_dict[f]

#         choice_0 = choice_dict['choice_0'][f]

#         choice_1 = choice_dict['choice_1'][f]

#         choice_2 = choice_dict['choice_2'][f]

#         choice_3 = choice_dict['choice_3'][f]

#         choice_4 = choice_dict['choice_4'][f]

#         choice_5 = choice_dict['choice_5'][f]

#         choice_6 = choice_dict['choice_6'][f]

#         choice_7 = choice_dict['choice_7'][f]

#         choice_8 = choice_dict['choice_8'][f]

#         choice_9 = choice_dict['choice_9'][f]



#         # add the family member count to the daily occupancy

#         daily_occupancy[d] += n



#         # Calculate the penalty for not getting top preference

#         if d == choice_0:

#             penalty += 0

#         elif d == choice_1:

#             penalty += 50

#         elif d == choice_2:

#             penalty += 50 + 9 * n

#         elif d == choice_3:

#             penalty += 100 + 9 * n

#         elif d == choice_4:

#             penalty += 200 + 9 * n

#         elif d == choice_5:

#             penalty += 200 + 18 * n

#         elif d == choice_6:

#             penalty += 300 + 18 * n

#         elif d == choice_7:

#             penalty += 300 + 36 * n

#         elif d == choice_8:

#             penalty += 400 + 36 * n

#         elif d == choice_9:

#             penalty += 500 + 36 * n + 199 * n

#         else:

#             penalty += 500 + 36 * n + 398 * n



#     # for each date, check total occupancy

#     #  (using soft constraints instead of hard constraints)

#     for v in daily_occupancy.values():

#         if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

#             penalty += 100000000

#             return penalty

#     # Calculate the accounting cost

#     # The first day (day 100) is treated special

#     accounting_cost = (daily_occupancy[days[0]] - 125.0) / 400.0 * daily_occupancy[days[0]] ** (0.5)

#     # using the max function because the soft constraints might allow occupancy to dip below 125

#     accounting_cost = max(0, accounting_cost)



#     # Loop over the rest of the days, keeping track of previous count

#     yesterday_count = daily_occupancy[days[0]]

#     for day in days[1:]:

#         today_count = daily_occupancy[day]

#         diff = abs(today_count - yesterday_count)

#         accounting_cost += max(0, (daily_occupancy[day] - 125.0) / 400.0 * daily_occupancy[day] ** (0.5 + diff / 50.0))

#         yesterday_count = today_count



#     penalty += accounting_cost



#     return penalty

# awesome kernel -https://www.kaggle.com/xhlulu/santa-s-2019-300x-faster-cost-function-37-s

import os



from numba import njit

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125

family_size = data.n_people.values

days_array = np.arange(N_DAYS, 0, -1)

choice_dict = data.loc[:, 'choice_0': 'choice_9'].T.to_dict()
choice_array_num = np.full((data.shape[0], N_DAYS + 1), -1)



for i, choice in enumerate(data.loc[:, 'choice_0': 'choice_9'].values):

    for d, day in enumerate(choice):

        choice_array_num[i, day] = d

        

penalties_array = np.array([

    [

        0,

        50,

        50 + 9 * n,

        100 + 9 * n,

        200 + 9 * n,

        200 + 18 * n,

        300 + 18 * n,

        300 + 36 * n,

        400 + 36 * n,

        500 + 36 * n + 199 * n,

        500 + 36 * n + 398 * n

    ]

    for n in range(family_size.max() + 1)

])



@njit

def cost_function(prediction, penalties_array, family_size, days):

    penalty = 0



    # We'll use this to count the number of people scheduled each day

    daily_occupancy = np.zeros((len(days)+1))

    N = family_size.shape[0]

    

    # Looping over each family; d is the day, n is size of that family, 

    # and choice is their top choices

    for i in range(N):

        # add the family member count to the daily occupancy

        n = family_size[i]

        d = prediction[i]

        choice = choice_array_num[i]

        

        daily_occupancy[d] += n



        # Calculate the penalty for not getting top preference

        penalty += penalties_array[n, choice[d]]



    # for each date, check total occupancy

    #  (using soft constraints instead of hard constraints)

    relevant_occupancy = daily_occupancy[1:]

    incorrect_occupancy = np.any(

        (relevant_occupancy > MAX_OCCUPANCY) | 

        (relevant_occupancy < MIN_OCCUPANCY)

    )

    

    if incorrect_occupancy:

        penalty += 100000000



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    init_occupancy = daily_occupancy[days[0]]

    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_cost = max(0, accounting_cost)

    

    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = init_occupancy

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = np.abs(today_count - yesterday_count)

        accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0))

        yesterday_count = today_count



    penalty += accounting_cost



    return penalty
best = submission['assigned_day'].values

start_score = cost_function(best, penalties_array, family_size, days_array)
print(start_score)
i = 0

while i <=20:

    new = best.copy()

    # loop over each family

    for fam_id in tqdm(range(len(best))):

        # loop over each family choice

        for pick in range(10):

            day = choice_dict[fam_id][f'choice_{pick}']

            temp = new.copy()

            temp[fam_id] = day # add in the new pick

            if cost_function(temp, penalties_array, family_size, days_array) < start_score:

                new = temp.copy()

                start_score = cost_function(new, penalties_array, family_size, days_array)



    score = cost_function(new, penalties_array, family_size, days_array)

    print(f'Score: {score}')

    submission['assigned_day'] = new

    i +=1

#submission.to_csv(f'submission_{score}.csv')
submission.to_csv("jonker_improved.csv",index = False)