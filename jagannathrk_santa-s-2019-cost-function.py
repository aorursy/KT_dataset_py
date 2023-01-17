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



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import os



from numba import njit

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm
base_path = '/kaggle/input/santa-workshop-tour-2019/'

data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')

submission = pd.read_csv(base_path + 'sample_submission.csv', index_col='family_id')
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
# Start with the sample submission values

best = submission['assigned_day'].values

start_score = cost_function(best, penalties_array, family_size, days_array)
%timeit cost_function(best, penalties_array, family_size, days_array)
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

submission.to_csv(f'submission_{score}.csv')