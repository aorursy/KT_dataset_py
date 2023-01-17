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
data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')

submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id')
# A dictionary of families with choices array and a size



families = [{

    'choices': [v for k, v in row[0:-1].items()],

    'size': row[-1]

} for (index, row) in data.iterrows()]
days = list(range(100, 0, -1))

max_occ = 300

min_occ = 125



def cost(prediction):

    penalty = 0

    occupancy = { k: 0 for k in days }

    

    for f, d in enumerate(prediction):

        n = families[f]['size']

        occupancy[d] += n

        

        choice = families[f]['choices'].index(d) if d in families[f]['choices'] else None

        

        if choice != None:

            penalty += [

                0, 

                50, 

                50 + 9 * n, 

                100 + 9 * n,

                200 + 9 * n,

                200 + 18 * n,

                300 + 18 * n,

                300 + 36 * n,

                400 + 36 * n,

                500 + 36 * n + 199 * n

            ][choice]

        else:

            penalty += 500 + 36 * n + 398 * n

            

    for (_, v) in occupancy.items():

        if v < min_occ or v > max_occ:

            penalty += 100000000000

            

    accounting_cost = max(0, (occupancy[days[0]] - 125) / 400 * (occupancy[days[0]] ** 0.5))

    

    for day in days[1:]:

        accounting_cost += max(0, (occupancy[day] - 125) / 400 * (occupancy[day] ** (0.5 + abs(occupancy[day] - occupancy[day + 1]) / 50)))

    

    return penalty + accounting_cost
def day_rating(fam_id, day):

    return families[fam_id]['choices'].index(day) if day in families[fam_id]['choices'] else 10



def analytical(start):

    start = start.copy()

    

    blacklist = []

    

    for family1 in range(len(families)):

        if family1 in blacklist:

            continue

            

        for family2 in range(len(families)):

            if family2 in blacklist:

                continue

                

            profit = day_rating(family1, start[family1]) + day_rating(family2, start[family2]) - day_rating(family1, start[family2]) - day_rating(family2, start[family1])

            if profit > 0:

                start[family1], start[family2] = start[family2], start[family1]

                blacklist += [family1, family2]

                

    return start
import time



def analytical_mod(start):

    start = start.copy()

    blacklist = []

    swaplist = []

    

    for family1 in range(len(families)):

        for family2 in range(len(families)):

            profit = day_rating(family1, start[family1]) + day_rating(family2, start[family2]) - day_rating(family1, start[family2]) - day_rating(family2, start[family1])

            if profit > 0:

                swaplist += [{

                    'score': profit,

                    'families': [family1, family2]

                }]

                

    swaplist.sort(key=lambda item: item['score'])

    swaplist.reverse()

    for item in swaplist:

        family1, family2 = item['families']

        if family1 in blacklist or family2 in blacklist:

            continue

        

        start[family1], start[family2] = start[family2], start[family1]

        blacklist += [family1, family2]

    return start





def analytical_mod_list(start):

    result = start.copy()

    cost_prev = cost(result)

    

    while True:

        result = analytical_mod(result)

        cost_current = cost(result)

        if cost_current == cost_prev:

            break

        cost_prev = cost_current

        yield result, cost_current



time_start = time.time()

for result, score in analytical_mod_list(submission['assigned_day'].tolist()):

    time_elapsed = time.time() - time_start

    print(f'This iteration\'s cost is {score}, and computing took {time_elapsed}s')

    res = result

    time_start = time.time()

def optimize(data):

    data = data.copy()

    score = cost(data)

    

    for family in range(len(families)):

        options = []

        for day in range(len(days) + 1)[1:]:

            if data[family] == day:

                continue

                

            drating = day_rating(family, data[family]) - day_rating(family, day)

            if drating > 0:

                options += [{

                    'drating': drating,

                    'day': day

                }]

        options.sort(key=lambda item: item['drating'])

        options.reverse()

        for option in options:

            data_m = data.copy()

            data_m[family] = option['day']

            new_cost = cost(data_m)

            if new_cost < score:

                score = new_cost

                data = data_m

    return data





while True:

    prev_cost = cost(res)

    time_start = time.time()

    res = optimize(res)

    new_cost = cost(res)

    time_end = time.time() - time_start

    print(f'A dataset with cost {prev_cost} improved to cost {new_cost} over {time_end}s')

    if new_cost == prev_cost:

        break



# The final result is now in res

output = pd.DataFrame(data={

    'assigned_day': res

})

score = cost(res)

output.to_csv(f'submission_{score}.csv', index_label='family_id')

print(score)