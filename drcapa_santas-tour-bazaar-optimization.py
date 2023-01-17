import numpy as np

import pandas as pd

import scipy.special

import matplotlib.pyplot as plt

import os
path_in = '../input/santa-workshop-tour-2019/'

path_start_solution = '../input/santa-workshop-tour-start-solution/'

print('input_data:', os.listdir(path_in))

print('start solution:', os.listdir(path_start_solution))
data = pd.read_csv(path_in+'family_data.csv')

data.index = data['family_id']

samp_subm = pd.read_csv(path_in+'sample_submission.csv')

start_solution = pd.read_csv(path_start_solution+'santa_workshop_tour_start_solution_01.csv', index_col=0)
num_days = 100

lower = 125

upper = 300

days = list(range(num_days, 0, -1))
def calc_family_costs(family):

    assigned_day = family['assigned_day']

    number_member = family['n_people']

    if assigned_day == family['choice_0']:

        penalty = 0

    elif assigned_day == family['choice_1']:

        penalty = 50

    elif assigned_day == family['choice_2']:

        penalty = 50 + 9 * number_member

    elif assigned_day == family['choice_3']:

        penalty = 100 + 9 * number_member

    elif assigned_day == family['choice_4']:

        penalty = 200 + 9 * number_member

    elif assigned_day == family['choice_5']:

        penalty = 200 + 18 * number_member

    elif assigned_day == family['choice_6']:

        penalty = 300 + 18 * number_member

    elif assigned_day == family['choice_7']:

        penalty = 300 + 36 * number_member

    elif assigned_day == family['choice_8']:

        penalty = 400 + 36 * number_member

    elif assigned_day == family['choice_9']:

        penalty = 500 + 36 * number_member + 199 * number_member

    else:

        penalty = 500 + 36 * number_member + 398 * number_member

    return penalty
def calc_accounting_cost(data):

    accounting_cost = 0

    daily_occupancy = {k:0 for k in days}

    family_size_dict = data[['n_people']].to_dict()['n_people']

    for f, d in enumerate(data['assigned_day']):

        n = family_size_dict[f]

        daily_occupancy[d] += n



    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)

    accounting_cost = max(0, accounting_cost)

    

    yesterday_count = daily_occupancy[days[0]]

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count

    return accounting_cost
def plot_results(data, name):

    x = data.columns

    y = data.loc[name]

    plt.plot(x, y, 'ro')

    plt.grid()

    plt.title(name)

    plt.xlabel('steps')

    plt.ylabel('value')

    plt.show()
def check_day(data, day):

    group_data = data.groupby('assigned_day').sum()['n_people'].to_frame()

    if (125 <= group_data.loc[day, 'n_people']) & (group_data.loc[day, 'n_people'] <= 300):

        return True

    else:

        return False
for i in range(num_days):

    data.loc[i*50:(i+1)*50-1, 'assigned_day'] = i+1

data['assigned_day'] = data['assigned_day'].astype(int)
data['assigned_day'] = start_solution['assigned_day']
data.head()
family_id = 100

calc_family_costs(data.iloc[family_id])
data['penalty_cost'] = data.apply(calc_family_costs, axis=1)
data.head()
acc_costs = calc_accounting_cost(data)

acc_costs
print('Total costs:', data['penalty_cost'].sum()+ acc_costs)
def check_swap_day(data, family, choice):

    data_copy = data.copy()

    data_copy.loc[family, 'assigned_day'] = data_copy.loc[family, 'choice_'+str(choice)]

    data_copy.loc[family, 'penalty_cost'] = calc_family_costs(data_copy.iloc[family])

    

    penalty_before = data.loc[family, 'penalty_cost']

    accounting_before = calc_accounting_cost(data)

    

    penalty_after = data_copy.loc[family, 'penalty_cost']

    accounting_after = calc_accounting_cost(data_copy)

    

    # Check conditions

    day_before = check_day(data_copy, data.loc[family, 'assigned_day'])

    day_after = check_day(data_copy, data_copy.loc[family, 'assigned_day'])



    if(day_before==True and day_after==True):

        improvement = (penalty_before-penalty_after)+(accounting_before-accounting_after)

    else:

        improvement = -1

    

    return improvement
family_id = 386

check_swap_day(data, family_id, 0)
def check_swap_family(data, family, choice):

    family1 = family

    day_family1 = data.loc[family1, 'assigned_day']

    penalty1 = data.loc[family1, 'penalty_cost']

    member_family1 = data.loc[family1, 'n_people']

    

    day_member_list = data.groupby('assigned_day')['family_id'].apply(list).to_frame()

    

    improvements = {}

    for member in day_member_list.loc[data.loc[family1, 'choice_'+str(choice)], 'family_id']:

        family2 = member

        day_family2 = data.loc[family2, 'assigned_day']

        member_family2 = data.loc[family2, 'n_people']

        penalty2 = data.loc[family2, 'penalty_cost']

        

        # simulate the swap with another family

        data_copy = data.copy()

        data_copy.loc[family2, 'assigned_day'] = data_copy.loc[family1, 'assigned_day']

        data_copy.loc[family1, 'assigned_day'] = data_copy.loc[family1, 'choice_'+str(choice)]

        # calc the new penalty cost for both families

        new_penalty1 = calc_family_costs(data_copy.iloc[family1])

        new_penalty2 = calc_family_costs(data_copy.iloc[family2])

        # check both days before and after swaping

        day_before = check_day(data_copy, data.loc[family1, 'assigned_day'])

        day_after = check_day(data_copy, data_copy.loc[family1, 'choice_'+str(choice)])

        # calc the accounting costs before and after swaping

        accounting_before = calc_accounting_cost(data)

        accounting_after = calc_accounting_cost(data_copy)

        if(day_before==True and day_after==True):

            improvement = (penalty1-new_penalty1) + (penalty2-new_penalty2) + (accounting_before-accounting_after)

        else:

            improvement = -1

        improvements.update({member:improvement})

   

    maximum = max(zip(improvements.values(), improvements.keys()))

    family_swap = maximum[1]

    return improvement, family_swap
family_id = 386

check_swap_family(data, family_id, 0)
family_id = 386

choice = 0

improvement_day = check_swap_day(data, family_id, choice)

improvement_family, family_swap = check_swap_family(data, family_id, choice)

improvement_day, improvement_family, family_swap
def go_to_bazaar(data, family):

    family1 = family

    day_family1 = data.loc[family1, 'assigned_day']

    penalty1 = data.loc[family1, 'penalty_cost']

    member_family1 = data.loc[family1, 'n_people']

    

    status = False

    

    for choice in range(10):

        """ Should i swap the day? """

        improvement_day = check_swap_day(data, family1, choice)

        """ Should i swap with another family? """

        improvement_family, family2 = check_swap_family(data, family1, choice)

    

        if(improvement_day >= 0 or improvement_family >= 0):

            if(improvement_day > improvement_family):

                #print('swap day')

                data.loc[family, 'assigned_day'] = data.loc[family, 'choice_'+str(choice)]

                data.loc[family, 'penalty_cost'] = calc_family_costs(data.iloc[family])

                status = True

            else:

                #print('swap family')

                data.loc[family2, 'assigned_day'] = data.loc[family1, 'assigned_day']

                data.loc[family1, 'assigned_day'] = data.loc[family1, 'choice_'+str(choice)]

        

                data.loc[family1, 'penalty_cost'] = calc_family_costs(data.iloc[family1])

                data.loc[family2, 'penalty_cost'] = calc_family_costs(data.iloc[family2])

                status = True

            if(status==True):

                break
family_id = 386

#go_to_bazaar(data, family_id)

#print('Total costs:', data['penalty_cost'].sum(), calc_accounting_cost(data))
results = pd.DataFrame()

results[0] = data['penalty_cost'].describe()

results.loc['costs', 0] = data['penalty_cost'].sum()+calc_accounting_cost(data)
num_steps = 10



for step in range(num_steps):

    print('step: ', step)

    families_high_scored = list(data[data['penalty_cost']>0].index)

    print('# families: ', len(families_high_scored),

          'first:', families_high_scored[0],

          'last:', families_high_scored[-1])

    for family in families_high_scored:

        #print('   family:', family)

        go_to_bazaar(data, family)

    data['penalty_cost'] = data.apply(calc_family_costs, axis=1)

    print('costs:', data['penalty_cost'].sum(), calc_accounting_cost(data))

    results[step+1] = data['penalty_cost'].describe()

    results.loc['costs', step+1] = data['penalty_cost'].sum()+calc_accounting_cost(data)
results = results.reindex(sorted(results.columns), axis=1)
plot_results(results, 'costs')
plot_results(results, 'mean')
print('Total costs:', data['penalty_cost'].sum() + calc_accounting_cost(data))
data = data.sort_index()

output = pd.DataFrame({'family_id': samp_subm.index,

                       'assigned_day': data['assigned_day']})

output.to_csv('submission.csv', index=False)
import pandas as pd

santa_workshop_tour_start_solution_01 = pd.read_csv("../input/santa-workshop-tour-start-solution/santa_workshop_tour_start_solution_01.csv")