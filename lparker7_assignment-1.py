import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')

fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')

family_size_dict = data[['n_people']].to_dict()['n_people'] 

cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].to_dict()

days = list(range(100, 0, -1))
def calc_penalty(table):

    penalty = 0    # 1

    people_scheduled = {k: 0 for k in days}    # n

    for family_id, day in enumerate(table):    # n

        number_of_people = family_size_dict[family_id]    # 2

        people_scheduled[day] += number_of_people    # 3

        # At most, this if statement has to do 10 evaluations, which all take 3

        if day == choice_dict['choice_0'][family_id]:

            penalty += 0    # 2

        elif day == choice_dict['choice_1'][family_id]:

            penalty += 50    # 2

        elif day == choice_dict['choice_2'][family_id]:

            penalty += 50 + 9 * number_of_people    # 4

        elif day == choice_dict['choice_3'][family_id]:

            penalty += 100 + 9 * number_of_people    # 4

        elif day == choice_dict['choice_4'][family_id]:

            penalty += 200 + 9 * number_of_people    # 4

        elif day == choice_dict['choice_5'][family_id]:

            penalty += 200 + 18 * number_of_people    # 4

        elif day == choice_dict['choice_6'][family_id]:

            penalty += 300 + 18 * number_of_people    # 4

        elif day == choice_dict['choice_7'][family_id]:

            penalty += 300 + 36 * number_of_people    # 4

        elif day == choice_dict['choice_8'][family_id]:

            penalty += 400 + 36 * number_of_people    # 4

        elif day == choice_dict['choice_9'][family_id]:

            penalty += 500 + 36 * number_of_people + 199 * number_of_people    # 6

        else:

            penalty += 500 + 36 * number_of_people + 398 * number_of_people    # 6



    for _, occupancy in people_scheduled.items():    # n

        if (occupancy < 125) or (occupancy > 300):    # 2 at worst

            # Use occupancy in penalty to incentivise picking under-occupied days

            penalty += (9999999999 - occupancy*10000)    # 4



    return penalty # 1
submission = pd.read_csv(fpath, index_col='family_id')



def find_best_score(family, table, pick=0):

    day = choice_dict[f'choice_{pick}'][family]    # 3

    test_table = table.copy()    # n

    test_table[family] = day    # 2

    if pick == 9:    # 1

        return test_table    # 1

    else:

        new_table = find_best_score(family, test_table, pick + 1)    # 1

        test_score = calc_penalty(test_table)     # 48n + 3

        new_score = calc_penalty(new_table)    # 48n + 3 

        if new_score < test_score:    # 1

            return new_table    # 1

        else:

            return test_table    # 1





start = time.process_time()    # 1



table = submission['assigned_day'].tolist()    # n

new = table.copy()    # n

for fam_id, _ in enumerate(table):    # n

    current_score = calc_penalty(new)    # 48n + 3

    new_table = find_best_score(fam_id, new).copy()    # n(97n + 16)

    new_score = calc_penalty(new_table)    # 48n + 3

    if new_score < current_score:    # 1

        new = new_table.copy()    # n

submission['assigned_day'] = new    # n

score = calc_penalty(new)    # 48n + 3

print(f'Recursion Score: {score}')

print(f'Recursion Time: {time.process_time() - start}')

submission.to_csv(f'submission_{score}.csv')
submission = pd.read_csv(fpath, index_col='family_id')



start = time.process_time()    # 1



table = submission['assigned_day'].tolist()    # n

new2 = table.copy()    # n

for fam_id, _ in enumerate(new2):    # n

    current_score = calc_penalty(new2)    # 48n + 3

    trial = new2.copy()    # n

    new_scores = list(range(0, len(choice_dict)))    # 5 (3 for range, 1 for len, 1 for assignment)

    for i in new_scores:    # n

        trial[fam_id] = choice_dict[f'choice_{i}'][fam_id]    # 4

        new_scores[i] = calc_penalty(trial)    # 48n + 4

    if current_score > min(new_scores):    # n + 1

        best_choice = new_scores.index(min(new_scores))    # 2n + 1

        new2[fam_id] = choice_dict[f'choice_{best_choice}'][fam_id]    # 4



submission['assigned_day'] = new2    # n

score = calc_penalty(new2)    # 48n + 3

print(f'For Loop Score: {score}')

print(f'For Loop Time: {time.process_time() - start}')

submission.to_csv(f'submission_{score}.csv')
submission = pd.read_csv(fpath, index_col='family_id')



# Stack reversal code was sourced from https://stackoverflow.com/questions/32975344/reversing-a-stack-in-python

def reverse(stack):    # Total: 3n + 1

    items = []    # 1

    while stack:    # n

        items.append(stack.pop())    # 2

    for item in items:    # n

        stack.append(item)    # 1



        

start = time.process_time()    # 1

table = submission['assigned_day'].tolist()    # n

answer = []    # 1

fam_id = 0    # 1

my_table = table.copy()    # n

reverse(my_table)    # 3n + 1



while my_table:    # n

    answer.append(my_table.pop())    # 2

    for i in choice_dict:    # n

        current_score = calc_penalty(answer)    # 48((n+1)/2) + 3 (since answer is building up to n in size, sub n with of (n+1)/2 to average n)

        current = answer.pop()    # 2

        answer.append(choice_dict[i][fam_id])    # 3

        new_score = calc_penalty(answer)    # 48((n+1)/2) + 3 (same as above)

        if current_score < new_score:    # 1

            answer.pop()    # 1

            answer.append(current)    # 1

    fam_id += 1    # 2



submission['assigned_day'] = answer    # n

score = calc_penalty(answer)    # 48n + 3

print(f'For Loop (Stacks) Score: {score}')

print(f'For Loop (Stacks) Time: {time.process_time() - start}')

submission.to_csv(f'submission_{score}.csv')
# Import our modules that we are using

import matplotlib.pyplot as plt



# Create the vectors X and Y

x = np.array(range(100),dtype='int64')

y_1 = np.array(range(100),dtype='int64')

y_2 = np.array(range(100),dtype='int64')

y_3 = np.array(range(100),dtype='int64')



for i in x:

    y_1[i] = 97*(i**3) + 110*(i**2) + 56*i + 3

    y_2[i] = 48*(i**3) + 58*(i**2) + 64*i + 3

    y_3[i] = 48*(i**3) + 62*(i**2) + 58*i + 7





# Create the plot

plt.plot(x,y_1,label='Recursion')

plt.plot(x,y_2,label='For loop')

plt.plot(x,y_3,label='For loop (with stacks)')





# Add a title

plt.title('Runtime comparison')



# Add X and y Label

plt.xlabel('Inputs')

plt.ylabel('# of primitive operations')



# Add a Legend

plt.legend()