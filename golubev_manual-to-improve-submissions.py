from ortools.linear_solver import pywraplp

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



NUMBER_DAYS = 100

NUMBER_FAMILIES = 5000

MAX_BEST_CHOICE = 5

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

submission = pd.read_csv('/kaggle/input/c-stochastic-product-search-65ns/submission.csv')

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



def days_plot(assigned_days):

    daily_occupancy = get_daily_occupancy(assigned_days)

    best_choices = get_daily_occupancy(DESIRED[:,0])

    plt.rcParams['figure.figsize'] = [20, 5]

    plt.xticks(np.arange(1, 101, step=1), rotation=90)

    plt.axhline(y=125, color='gray', linestyle=':')

    plt.axhline(y=300, color='gray', linestyle=':')

    mondays125     = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1 and daily_occupancy[day] == 125])

    other_mondays  = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1 and daily_occupancy[day] != 125])

    weekends       = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 in [2,3,4] or day == 0])

    not_weekends   = np.array([(day+1, daily_occupancy[day]) for day in range(1, 100) if day % 7 in [0,5,6]])

    plt.bar(*weekends.transpose()      , color = 'y', label = 'Weekends')

    plt.bar(*not_weekends.transpose()  , color = 'b', label = 'Thu-Wed-Tue')

    plt.bar(*other_mondays.transpose() , color = 'm', label = 'Mondays > 125')

    plt.bar(*mondays125.transpose()    , color = 'g', label = 'Mondays = 125')

    plt.plot(range(1,101), best_choices, color = 'k', label = 'Best choices')

    plt.ylim(0, 500)

    plt.xlim(0, 101)

    plt.xlabel('Days before Christmas', fontsize=14)

    plt.ylabel('Occupancy', fontsize=14)

    plt.legend()

    plt.show()

    

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

days_plot(assigned_days)

print("Score: ", cost_function(assigned_days))
days_for_fix = np.array([44, 51, 58])

daily_occupancy = get_daily_occupancy(assigned_days)

fids = np.where(np.isin(assigned_days, days_for_fix))[0] # Ids of family for move



solver = pywraplp.Solver('Setup occupation of days', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

PCOSTM, B = {}, {}

for fid in fids:

    for i in range(MAX_BEST_CHOICE):

        B[fid, DESIRED[fid][i]-1] = solver.BoolVar(f'b{fid, i}')

        PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]



lower_bounds = np.zeros(100)

upper_bounds = 300. - daily_occupancy

upper_bounds[np.arange(100)%7 == 1] = 0 # don't move to Mondays



# Daily occupation for special Mondays only 125

lower_bounds[days_for_fix-1] = 125

upper_bounds[days_for_fix-1] = 125





for j in range(NUMBER_DAYS):

    I = solver.IntVar(lower_bounds[j], upper_bounds[j], f'I{j}')

    solver.Add(solver.Sum([N_PEOPLE[i] * B[i, j] for i in range(NUMBER_FAMILIES) if (i,j) in B]) == I)

    

for i in fids:

    solver.Add(solver.Sum(B[i, j] for j in range(NUMBER_DAYS) if (i,j) in B) == 1)



solver.Minimize(solver.Sum(PCOSTM[i, j] * B[i, j] for i, j in B))

sol = solver.Solve()



status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']

if status[sol] == 'OPTIMAL':

    for i, j in B:

        if B[i, j].solution_value() > 0.5:

            assigned_days[i] = j+1

            

print('Solution: ', status[sol])

print("Score: ", cost_function(assigned_days))

days_plot(assigned_days)

submission['assigned_day'] = assigned_days

submission.to_csv('submission.csv', index=False)