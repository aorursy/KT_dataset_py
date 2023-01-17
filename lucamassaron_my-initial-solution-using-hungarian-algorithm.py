# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!conda install -c psi4 hungarian --yes



import hungarian

help('hungarian')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm



import hungarian

from itertools import product

from functools import partial

from numba import njit



pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)
fpath = '../input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '../input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
top_kernel_submission = pd.read_csv("../input/best-submission-31-12a/submission.csv")
def stacked_plot(solution):

    data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')

    data['assigned_day'] = solution

    

    for c in range(10):

        data[f'n_people_{c}'] = np.where(data[f'choice_{c}'] == data['assigned_day'], data['n_people'], 0)

        

    for c in range(1, 10):

        d = c -1

        data[f'n_people_{c}'] = data[f'n_people_{d}'] + data[f'n_people_{c}']

        

    agg_data = data.groupby(by=['assigned_day'])['n_people', 'n_people_0', 'n_people_1', 'n_people_2', 'n_people_3', 'n_people_4', 'n_people_5', 'n_people_6', 'n_people_7', 'n_people_8', 'n_people_9'].sum().reset_index()

    

    f, ax = plt.subplots(figsize=(12, 20))

    sns.set_color_codes("pastel")

    sns.barplot(x='n_people_9', y='assigned_day', data=agg_data, label='choice_9', orient='h', color='k')

    sns.barplot(x='n_people_8', y='assigned_day', data=agg_data, label='choice_8', orient='h', color='k')

    sns.barplot(x='n_people_7', y='assigned_day', data=agg_data, label='choice_7', orient='h', color='k')

    sns.barplot(x='n_people_6', y='assigned_day', data=agg_data, label='choice_6', orient='h', color='k')

    sns.barplot(x='n_people_5', y='assigned_day', data=agg_data, label='choice_5', orient='h', color='k')

    sns.barplot(x='n_people_4', y='assigned_day', data=agg_data, label='choice_4', orient='h', color='r')

    sns.barplot(x='n_people_3', y='assigned_day', data=agg_data, label='choice_3', orient='h', color='y')

    sns.barplot(x='n_people_2', y='assigned_day', data=agg_data, label='choice_2', orient='h', color='g')

    sns.barplot(x='n_people_1', y='assigned_day', data=agg_data, label='choice_1', orient='h', color='c')

    sns.barplot(x='n_people_0', y='assigned_day', data=agg_data, label='choice_0', orient='h', color='b')

    ax.axvline(125, color="k", clip_on=False)

    ax.axvline(300, color="k", clip_on=False)

    ax.legend(ncol=2, loc="lower right", frameon=True)

    ax.set(xlabel="Occupancy")
desired = data.values[:, :-1]

family_size = data.n_people.values

penalties = np.asarray([

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

    ] for n in range(family_size.max() + 1)

])



def get_penalty(n, choice):

    return penalties[n, choice]



@njit(fastmath=True)

def jited_cost(prediction, desired, family_size, penalties,

               n_days=100, min_occupancy=125, max_occupancy=300):

    """

    From: https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit

    """

    total_penalty = 0

    daily_occupancy = np.zeros(n_days + 1, dtype=np.int64)

    penalties_by_day = np.zeros(n_days)

    accounting_cost_by_day = np.zeros(n_days)

    penalties_by_family = np.zeros(len(prediction))

    

    for i in range(len(prediction)):

        n = family_size[i]

        pred = prediction[i]

        n_choice = 0

        for j in range(len(desired[i])):

            if desired[i, j] == pred:

                break

            else:

                n_choice += 1

        

        daily_occupancy[pred - 1] += n

        estimated_penalty = penalties[n, n_choice]

        penalties_by_day[pred] += estimated_penalty

        penalties_by_family[i] = estimated_penalty

        total_penalty += estimated_penalty



    accounting_cost = 0

    n_out_of_range = 0

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(n_days):

        n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        n_out_of_range += (n > max_occupancy) or (n < min_occupancy)

        diff = abs(n - n_next)

        estimated_accounting_cost = max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

        accounting_cost_by_day[day] = estimated_accounting_cost

        accounting_cost += estimated_accounting_cost

        

    total_penalty += accounting_cost

    return (total_penalty, n_out_of_range, daily_occupancy,

            penalties_by_day, accounting_cost_by_day, penalties_by_family)



cost_evaluation = partial(jited_cost, 

                          desired=desired, 

                          family_size=family_size, 

                          penalties=penalties)

prediction = top_kernel_submission['assigned_day'].values



evaluated, *errors = cost_evaluation(prediction)

print("Score:", evaluated)

print("Violations of daily occupancies:", errors[0])

print("Penalty costs:", np.sum(errors[3]))

print("Accountancy costs:", np.sum(errors[4]))
def map_days_to_hungarian(vec, n_days=100):



    pos_to_day = dict()

    day_to_pos = {(i+1): list() for i in range(n_days)}



    k = 0

    for i in range(n_days):

        for j in range(vec[i]):

            pos_to_day[k] = (i+1)

            day_to_pos[i+1].append(k)

            k += 1

    

    return pos_to_day, day_to_pos
def solve_hungarian(dff, pos_to_day, day_to_pos):

    preference_mat = np.zeros((5000, 5000)).astype(int)

    # first fill every column with the maximum cost for that family

    for ind, row in dff.iterrows():

        preference_mat[ind] = get_penalty(row['n_people'], 10)

    for ind, row in dff.iterrows():

        for i in range(10):

            choice = row[f'choice_{i}']

            choice_cost = get_penalty(row['n_people'], i)

            for k in range(min(day_to_pos[choice]), (max(day_to_pos[choice])+1)):

                    preference_mat[ind, k] = choice_cost

    preference_mat = preference_mat.astype(int)

    

    # solving

    assignment, cost = hungarian.lap(preference_mat)

    pred = np.array([pos_to_day[a] for a in assignment])

    

    return pred
hypothesis = (top_kernel_submission.merge(data.reset_index()[['family_id', 'n_people']], on='family_id')

                                   .groupby('assigned_day')

                                   .sum()

                                   .apply(lambda x : x //4)

                                   .loc[:,'n_people']

                                   .values

             )



hypothesis = np.round((hypothesis / np.sum(hypothesis)) * 5000).astype(int)

hypothesis[0] -= np.sum(hypothesis) - 5000
attempts = 0



while True:

    attempts += 1

    pos_to_day, day_to_pos = map_days_to_hungarian(hypothesis)

    solution = solve_hungarian(data, pos_to_day, day_to_pos)



    evaluated, *errors = cost_evaluation(solution)

    if errors[0] > 0:

        # Let's relax the hypothesis

        ordered = np.argsort(hypothesis)

        hypothesis[ordered[:50]] += 1

        hypothesis[ordered[50:]] -= 1

        print('.', end='', flush=True)

    else:

        print(f"\nOptimized after {attempts} times relaxing allocation")

        break
pos_to_day, day_to_pos = map_days_to_hungarian(hypothesis)

solution = solve_hungarian(data, pos_to_day, day_to_pos)



evaluated, *errors = cost_evaluation(solution)

print("Score:", evaluated)

print("Violations of daily occupancies:", errors[0])

print("Penalty costs:", np.sum(errors[3]))

print("Accountancy costs:", np.sum(errors[4]))



S = pd.DataFrame({'day':solution, 'size': family_size}).groupby('day').sum().reset_index()

plt.bar(S['day'], S['size'], width=1);



stacked_plot(solution)
family_cost_matrix = np.concatenate(data.n_people.apply(lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))

for fam in data.index:

    for choice_order, day in enumerate(data.loc[fam].drop("n_people")):

        family_cost_matrix[fam, day - 1] = penalties[data.loc[fam, "n_people"], choice_order]



accounting_cost_matrix = np.zeros((500, 500))

for n in range(accounting_cost_matrix.shape[0]):

    for diff in range(accounting_cost_matrix.shape[1]):

        accounting_cost_matrix[n, diff] = max(0, (n - 125.0) / 400.0 * n**(0.5 + diff / 50.0))



@njit(fastmath=True)

def cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix):

    N_DAYS = family_cost_matrix.shape[1]

    MAX_OCCUPANCY = 300

    MIN_OCCUPANCY = 125

    penalty = 0

    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int16)

    for i, (pred, n) in enumerate(zip(prediction, family_size)):

        daily_occupancy[pred - 1] += n

        penalty += family_cost_matrix[i, pred - 1]



    accounting_cost = 0

    n_low = 0

    n_high = 0

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        n_high += (n > MAX_OCCUPANCY) 

        n_low += (n < MIN_OCCUPANCY)

        diff = abs(n - n_next)

        accounting_cost += accounting_cost_matrix[n, diff]



    return np.asarray([penalty, accounting_cost, n_low, n_high])



def get_cost_consolidated(prediction): 

    fc, ac, l, h = cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix)

    return (fc + ac) + (l + h) * 1000000
arrangement = solution
# Fine tuning by assigning possible desiderata to families



start_score = get_cost_consolidated(arrangement)

print(f'Starting score from previous optimization: {start_score}')



fam_size_order = np.argsort(family_size)



for i in range(100):

    # loop over each family

    if i > 0:

        if iter_score == start_score:

            print(f"Early stop at loop {i+1}")

            break

    iter_score = start_score

    for fam_id in fam_size_order:

        # loop over each family choice

        for pick in range(7):

            day = desired[fam_id, pick]

            temp = arrangement.copy()

            temp[fam_id] = day # add in the new pick

            temp_cost = get_cost_consolidated(temp)

            if temp_cost < start_score and errors==0:

                arrangement = temp.copy()

                start_score = temp_cost

                print('Score: '+str(start_score), end='\r', flush=True)
# Repeated swap algorithm

best_score = get_cost_consolidated(arrangement)

fam_size_order = np.argsort(family_size)



print(f'Score after upgrading optimization: {best_score}')



for r in range(10):

    if r > 0 and best_score == last_iter_score:

        break

    last_iter_score = best_score

    print(f'--- Loop no {r+1} ---')

    for k, i in enumerate(tqdm(fam_size_order)):

            if (k%10==0):

                print(k, best_score,'     ',end='\r')

            for j in fam_size_order:

                if j != i:

                    temp = arrangement.copy()

                    temp[i], temp[j] = temp[j], temp[i]

                    cur_score = get_cost_consolidated(temp)

                    if cur_score < best_score:

                        best_score = cur_score

                        arrangement = temp.copy()
choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values



def stochastic_product_search(top_k, fam_size, original, choice_matrix,

                              cost_function=get_cost_consolidated,

                              disable_tqdm=False, verbose=10000,

                              n_iter=500, random_state=2019):

    """

    original (np.array): The original day assignments.

    

    At every iterations, randomly sample fam_size families. Then, given their top_k

    choices, compute the Cartesian product of the families' choices, and compute the

    score for each of those top_k^fam_size products.

    """

    

    best = original.copy()

    best_score = cost_function(best)

    

    np.random.seed(random_state)



    for i in tqdm(range(n_iter), disable=disable_tqdm):

        fam_indices = np.random.choice(range(choice_matrix.shape[0]), size=fam_size)

        changes = np.array(list(product(*choice_matrix[fam_indices, :top_k].tolist())))



        for change in changes:

            new = best.copy()

            new[fam_indices] = change



            new_score = cost_function(new)



            if new_score < best_score:

                best_score = new_score

                best = new

        

        if new_score < best_score:

            best_score = new_score

            best = new

    

        if verbose and i % verbose == 0:

            print(f"Iteration #{i}: Best score is {best_score:.2f}")

    

    print(f"Final best score is {best_score:.2f}")

    return best



attempts = 2



for repeat in range(attempts):

    arrangement = stochastic_product_search(

        choice_matrix=choice_matrix, 

        top_k=2,

        fam_size=8, 

        original=arrangement, 

        n_iter=1000000,

        disable_tqdm=True,

        verbose=50000

    )
evaluated, *errors = cost_evaluation(arrangement)

(n_out_of_range, 

 daily_occupancy,

 penalties_by_day, 

 accounting_cost_by_day, 

 penalties_by_family) = errors



print("Score:", evaluated)

print("Violations of daily occupancies:", errors[0])

print("Penalty costs:", np.sum(errors[3]))

print("Accountancy costs:", np.sum(errors[4]))



S = pd.DataFrame({'day':arrangement, 'size': family_size}).groupby('day').sum().reset_index()

plt.bar(S['day'], S['size'], width=1);

stacked_plot(arrangement)
submission['assigned_day'] = arrangement

score, *error = cost_evaluation(arrangement)

submission.to_csv(f'submission_{int(score)}.csv')

print(f'Saved score: {score}')

print(f'Sanity check: {int(error[0])} days are violating the constraints')