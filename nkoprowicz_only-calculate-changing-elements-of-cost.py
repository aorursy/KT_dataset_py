import numpy as np 

import pandas as pd 

import time

import os
df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col = 'family_id')

submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col = 'family_id')
family_size_dict = df[['n_people']].to_dict()['n_people']



cols = [f'choice_{i}' for i in range(10)]

choice_dict = df[cols].to_dict()
N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



days = list(range(N_DAYS, 0, -1))
def cost_function(prediction):

    

    penalty = 0

    daily_occupancy_cost = {k: 0 for k in days}

    

    for f, d in enumerate(prediction):

        

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

        

        daily_occupancy_cost[d] += n

        

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

            

    for _, v in daily_occupancy_cost.items():

        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

            penalty += 100000000



    accounting_cost = (daily_occupancy_cost[days[0]] - 125.0) / 400.0 * daily_occupancy_cost[days[0]]**(0.5)

    accounting_cost = max(0, accounting_cost)



    yesterday_count = daily_occupancy_cost[days[0]]



    for day in days[1:]:

        today_count = daily_occupancy_cost[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy_cost[day] - 125.0) / 400.0 * daily_occupancy_cost[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count

    penalty += accounting_cost



    return penalty

            
def calc_family_penalty(f, d, daily_occupancy_fn):

    penalty = 0

    

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

            

    for _, v in daily_occupancy_fn.items():

        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

            penalty += 100000000

            

    return penalty
def accounting_day(day, daily_occupancy_2):

    if(day != 100):

        diff = abs(daily_occupancy_2[day + 1] - daily_occupancy_2[day])

        return max(0, (daily_occupancy_2[day] - 125.0) / 400.0 * daily_occupancy_2[day]**(0.5 + diff / 50.0))

    else:

        return (daily_occupancy_2[day] - 125.0) / 400.0 * daily_occupancy_2[day]**(0.5)

    
best = submission['assigned_day'].tolist()

start_score = cost_function(best)

print(start_score)

new = best.copy()

'''

for fam_id, _ in enumerate(best):

    for pick in range(10):

        day = choice_dict[f'choice_{pick}'][fam_id]

        temp = new.copy()

        temp[fam_id] = day

        new_cost = cost_function(temp)

        if(new_cost < start_score):

            new = temp.copy()

            start_score = new_cost

'''
daily_occupancy = {k: 0 for k in days}



for f, d in enumerate(new):   

        n = family_size_dict[f]

        daily_occupancy[d] += n
def change_family_days():

    

    global daily_occupancy

    

    for fam_id in range(5000):

            

            # try moving them to every day

            for i in range(1, 101):

                

                fam1_day = new[fam_id]

                

                # accounting cost for the family's current day, before switching them

                accounting_current_before_1 = accounting_day(fam1_day, daily_occupancy)

                

                # accounting cost for the day before the family's current day, before switching them

                if((fam1_day != 1) & (i != fam1_day - 1)):

                    accounting_current_before_2 = accounting_day(fam1_day - 1, daily_occupancy)

                else: 

                    accounting_current_before_2 = 0

                    

                # accounting cost for the day we're going to try to move the family to, before switching them

                accounting_future_before_1 = accounting_day(i, daily_occupancy)

                

                # accounting cost for the day before the day we're going to try to move the family to, before switching them

                if((i != 1) & (fam1_day != i - 1)):

                    accounting_future_before_2 = accounting_day(i - 1, daily_occupancy)

                else: 

                    accounting_future_before_2 = 0

                    

                # the penalty based on the family's choices, before moving them

                fam1_penalty_before = calc_family_penalty(fam_id, fam1_day, daily_occupancy)

                

                # the total cost of the things that will change after we move the family

                before = accounting_current_before_1 + accounting_current_before_2 + fam1_penalty_before + accounting_future_before_1 + accounting_future_before_2



                # move the family by updating the daily occupancy

                daily_occupancy_temp = daily_occupancy.copy()

                daily_occupancy_temp[fam1_day] = daily_occupancy_temp[fam1_day] - family_size_dict[fam_id]

                daily_occupancy_temp[i] = daily_occupancy_temp[i] + family_size_dict[fam_id]



                # accounting cost for the day the family is moving to, after we've moved them

                accounting_future_after_1 = accounting_day(i, daily_occupancy_temp)

                

                # accounting cost for the day before the one the family is moving to, after we've moved them

                if((i != 1) & (fam1_day != i - 1)):

                    accounting_future_after_2 = accounting_day(i - 1, daily_occupancy_temp)

                else: 

                    accounting_future_after_2 = 0

                

                # accounting cost for the day the family started on, after we've moved them

                accounting_day_current_after_1 = accounting_day(fam1_day, daily_occupancy_temp)

                

                # accounting cost for the day before the one the family started on, after we've moved them

                if((fam1_day != 1) & (i != fam1_day - 1)):

                    accounting_current_after_2 = accounting_day(fam1_day - 1, daily_occupancy_temp)

                else: 

                    accounting_current_after_2 = 0

                

                # the penalty based on the family's choices, after moving them

                fam1_penalty_post = calc_family_penalty(fam_id, i, daily_occupancy_temp)

                

                # the total cost of the things that will change after we've moved the family

                after = accounting_future_after_1 + accounting_future_after_2 + fam1_penalty_post + accounting_day_current_after_1 + accounting_current_after_2



                # if the overall cost has decreased, move them

                if(before > after):

                    print(f'Switching {fam_id} from day {fam1_day} to day {i}')

                    new[fam_id] = i 

                    daily_occupancy = daily_occupancy_temp.copy()
def swap_families():

    

    global daily_occupancy

    

    for fam_id in range(5000):

        for fam_id2 in range(5000):

            if fam_id2 == fam_id:

                continue

  

            fam1_day = new[fam_id]

            fam2_day = new[fam_id2]

            

            accounting_fam1_before_1 = accounting_day(fam1_day, daily_occupancy)

            

            if((fam1_day != 1) & (fam1_day - 1 != fam2_day)):

                accounting_fam1_before_2 = accounting_day(fam1_day - 1, daily_occupancy)

            else: 

                accounting_fam1_before_2 = 0

                

            accounting_fam2_before_1 = accounting_day(fam2_day, daily_occupancy)

            

            if((fam2_day != 1) & (fam2_day - 1 != fam1_day)):

                accounting_fam2_before_2 = accounting_day(fam2_day - 1, daily_occupancy)

            else:

                accounting_fam2_before_2 = 0

                

            fam1_penalty_pre = calc_family_penalty(fam_id, fam1_day, daily_occupancy)

            fam2_penalty_pre = calc_family_penalty(fam_id2, fam2_day, daily_occupancy)

            

            # The cost of everything that will change, before the families are switched

            before = accounting_fam1_before_1 + accounting_fam1_before_2 + accounting_fam2_before_1 + accounting_fam2_before_2 + fam1_penalty_pre + fam2_penalty_pre



            # Switch the families by updating the daily occupancy

            daily_occupancy_temp = daily_occupancy.copy()

            daily_occupancy_temp[fam1_day] = daily_occupancy_temp[fam1_day] - family_size_dict[fam_id] + family_size_dict[fam_id2]

            daily_occupancy_temp[fam2_day] = daily_occupancy_temp[fam2_day] - family_size_dict[fam_id2] + family_size_dict[fam_id]



            accounting_fam1_after_1 = accounting_day(fam1_day, daily_occupancy_temp)

            

            if((fam1_day != 1) & (fam1_day - 1 != fam2_day)):

                accounting_fam1_after_2 = accounting_day(fam1_day - 1, daily_occupancy_temp)

            else:

                accounting_fam1_after_2 = 0

                

            accounting_fam2_after_1 = accounting_day(fam2_day, daily_occupancy_temp)

            

            if((fam2_day != 1) & (fam2_day - 1 != fam1_day)):

                accounting_fam2_after_2 = accounting_day(fam2_day - 1, daily_occupancy_temp)

            else:

                accounting_fam2_after_2 = 0

                

            fam1_penalty_post = calc_family_penalty(fam_id, fam2_day, daily_occupancy_temp)

            fam2_penalty_post = calc_family_penalty(fam_id2, fam1_day, daily_occupancy_temp)

            

            # The cost of everything that will change, after the families are switched

            after = accounting_fam1_after_1 + accounting_fam1_after_2 + accounting_fam2_after_1 + accounting_fam2_after_2 + fam1_penalty_post + fam2_penalty_post



            # If the cost decreases, swap them

            if(before > after):

                daily_occupancy = daily_occupancy_temp.copy()

                new[fam_id] = fam2_day

                new[fam_id2] = fam1_day
import random
cost = cost_function(new)

i = 0

b = 0

print('Working on minimizing cost')

print(f'current cost: {cost}')

while(True):

    i = i + 1

    change_family_days()

    new_cost = cost_function(new)

    if(new_cost < cost):

        print(f'Round {i}: Changed family days.  New cost: {new_cost}')

        cost = new_cost

        continue

    swap_families()

    new_cost = cost_function(new)

    if(new_cost < cost):

        print(f'Round {i}: Swapped families.  New cost: {new_cost}')

        cost = new_cost

        continue

    else:

        break
print('making submission')

submission['assigned_day'] = new
submission.to_csv('submission.csv')