from ortools.linear_solver import pywraplp

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



NUMBER_DAYS = 100

NUMBER_FAMILIES = 5000

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

#submission = pd.read_csv('/kaggle/input/c-stochastic-product-search-65ns/submission.csv')

submission = pd.read_csv("../input/santa-submission-69818/submission_69818.csv")

assigned_days = submission['assigned_day'].values

columns = data.columns[1:11]

DESIRED = data[columns].values

COST_PER_FAMILY        = [0,50,50,100,200,200,300,300,400,500]

COST_PER_FAMILY_MEMBER = [0, 0, 9,  9,  9, 18, 18, 36, 36,235]

N_PEOPLE = data['n_people'].values



def get_daily_occupancy(assigned_days):

    daily_occupancy = np.zeros(100, int)

    for fid, assigned_day in enumerate(assigned_days):

        daily_occupancy[assigned_day-1] += N_PEOPLE[fid]

    return daily_occupancy

    

def cost_function(prediction):

    N_DAYS = 100

    MAX_OCCUPANCY = 300

    MIN_OCCUPANCY = 125

    penalty = 0

    days = list(range(N_DAYS,0,-1))

    tmp = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')

    family_size_dict = tmp[['n_people']].to_dict()['n_people']



    cols = [f'choice_{i}' for i in range(10)]

    choice_dict = tmp[cols].to_dict()



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

        if  (v < MIN_OCCUPANCY): #(v > MAX_OCCUPANCY) or

            penalty += 100000000



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_costs = [max(0, accounting_cost)]

    diffs = [0]

    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = daily_occupancy[days[0]]

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_costs.append(max(0, (today_count-125.0) / 400.0 * today_count**(0.5 + diff / 50.0)))

        yesterday_count = today_count



    return penalty, sum(accounting_costs), penalty + sum(accounting_costs)
from ortools.linear_solver import pywraplp

MAX_BEST_CHOICE = 6

NUM_SWAP = 3000

NUM_SECONDS = 3600

NUM_THREADS = 4

for _ in range(40):

    solver = pywraplp.Solver('Optimization preference cost', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    daily_occupancy = get_daily_occupancy(assigned_days).astype(float)

    fids = np.random.choice(range(NUMBER_FAMILIES), NUM_SWAP, replace=False)

    PCOSTM, B = {}, {}

    for fid in range(NUMBER_FAMILIES):

        if fid in fids:

            for i in range(MAX_BEST_CHOICE):

                PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]

                B[     fid, DESIRED[fid][i]-1] = solver.BoolVar('')

        else:

            daily_occupancy[assigned_days[fid]-1] -= N_PEOPLE[fid]



    solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)

    solver.SetNumThreads(NUM_THREADS)



    for day in range(NUMBER_DAYS):

        if daily_occupancy[day]:

            solver.Add(solver.Sum([N_PEOPLE[fid] * B[fid, day] for fid in range(NUMBER_FAMILIES) if (fid,day) in B]) == daily_occupancy[day])

        

    for fid in fids:

        solver.Add(solver.Sum(B[fid, day] for day in range(NUMBER_DAYS) if (fid, day) in B) == 1)



    solver.Minimize(solver.Sum(PCOSTM[fid, day] * B[fid, day] for fid, day in B))

    sol = solver.Solve()

    

    status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']

    if status[sol] in ['OPTIMAL', 'FEASIBLE']:

        tmp = assigned_days.copy()

        for fid, day in B:

            if B[fid, day].solution_value() > 0.48:

                tmp[fid] = day+1

        if cost_function(tmp)[2] < cost_function(assigned_days)[2]:

            assigned_days = tmp

            submission['assigned_day'] = assigned_days

            submission.to_csv('submission.csv', index=False)

        print('Result:', status[sol], cost_function(tmp))

    else:

        print('Result:', status[sol])