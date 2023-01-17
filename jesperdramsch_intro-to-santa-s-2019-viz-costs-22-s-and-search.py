# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from numba import njit

from itertools import product

from time import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

start_time = time()

end_time = start_time + (7.5 *60 *60)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-s-2019-stochastic-product-search/submission_76177.csv'

prediction = pd.read_csv(fpath, index_col='family_id').assigned_day.values
family_sizes = data.n_people.values.astype(np.int8)
data.head()
print("Average number of people per day:", sum(data['n_people'])//100)
family_size = data['n_people'].value_counts().sort_index()

family_size /= 50



plt.figure(figsize=(14,6))

ax = sns.barplot(x=family_size.index, y=family_size.values)



ax.set_ylim(0, 1.1*max(family_size))

plt.xlabel('Family Size', fontsize=14)

plt.ylabel('Percentage', fontsize=14)

plt.title('Members in Family', fontsize=20)

plt.show()
fave_day = data['choice_0'].value_counts().sort_index()



plt.figure(figsize=(14,6))

ax = sns.barplot(x=fave_day.index, y=fave_day.values)





plt.xlabel('$\Leftarrow$ Christmas this way!', fontsize=14)

plt.ylabel('Wishes', fontsize=14)

plt.title('Primary Choice', fontsize=20)

plt.show()
ninthfave_day = data['choice_9'].value_counts().sort_index()



plt.figure(figsize=(14,6))

ax = sns.barplot(x=ninthfave_day.index, y=ninthfave_day.values)





plt.xlabel('$\Leftarrow$ Christmas this way!', fontsize=14)

plt.ylabel('Wishes', fontsize=14)

plt.title('Last Choice', fontsize=20)

plt.show()
cost_dict = {0:  [  0,  0],

             1:  [ 50,  0],

             2:  [ 50,  9],

             3:  [100,  9],

             4:  [200,  9],

             5:  [200, 18],

             6:  [300, 18],

             7:  [300, 36],

             8:  [400, 36],

             9:  [500, 36 + 199],

             10: [500, 36 + 398],

            }



def cost(choice, members, cost_dict):

    x = cost_dict[choice]

    return x[0] + members * x[1]
all_costs = {k: pd.Series([cost(k, x, cost_dict) for x in range(2,9)], index=range(2,9)) for k in cost_dict.keys()}

df_all_costs = pd.DataFrame(all_costs)

plt.figure(figsize=(14,11))

sns.heatmap(df_all_costs.drop([0, 1],axis=1), annot=True, fmt="g")
plt.figure(figsize=(14,11))

sns.heatmap(df_all_costs.drop([0, 1, 9, 10],axis=1), annot=True, fmt="g")
family_cost_matrix = np.zeros((100,len(family_sizes))) # Cost for each family for each day.



for i, el in enumerate(family_sizes):

    family_cost_matrix[:, i] += all_costs[10][el] # populate each day with the max cost

    for j, choice in enumerate(data.drop("n_people",axis=1).values[i,:]):

        family_cost_matrix[choice-1, i] = all_costs[j][el] # fill wishes into cost matrix
plt.figure(figsize=(40,10))

sns.heatmap(family_cost_matrix)

plt.show()
def accounting(today, previous):

    return ((today - 125) / 400 ) * today ** (.5 + (abs(today - previous) / 50))
acc_costs = np.zeros([176,176])



for i, x in enumerate(range(125,300+1)):

    for j, y in enumerate(range(125,300+1)):

        acc_costs[i,j] = accounting(x,y)



plt.figure(figsize=(10,10))

plt.imshow(np.clip(acc_costs, 0, 4000))

plt.title('Accounting Cost')

plt.colorbar()



print("The maximum cost is a ridiculous:", acc_costs.max())
@njit(fastmath=True)

def cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix):

    N_DAYS = 100

    MAX_OCCUPANCY = 300

    MIN_OCCUPANCY = 125

    penalty = 0

    accounting_cost = 0

    max_occ = False

    

    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int16)

    for i, (pred, n) in enumerate(zip(prediction, family_size)):

        daily_occupancy[pred - 1] += n

        penalty += family_cost_matrix[pred - 1, i]

        

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        max_occ += MIN_OCCUPANCY > n

        max_occ += MAX_OCCUPANCY < n

        accounting_cost += accounting_cost_matrix[n-MIN_OCCUPANCY, n_next-MIN_OCCUPANCY]

    if max_occ: 

        return 1e11

    return penalty + accounting_cost
cost_function(prediction, family_sizes, family_cost_matrix, acc_costs)
%timeit cost_function(prediction, family_sizes, family_cost_matrix, acc_costs)
def stochastic_product_search(original, choice_matrix, top_k=2, num_fam=8, 

                              n_time=None, n_iter=None, early_stop=np.inf,

                              decrease=np.inf, random_state=42, verbose=1e4):

    """Randomly sample families, reassign and evaluate.

    

    At every iterations, randomly sample num_fam families. Then, given their top_k

    choices, compute the Cartesian product of the families' choices, and compute the

    score for each of those top_k^num_fam products.

    

    Both n_time and n_iter can be set, optimization stops when first one is reached.

    

    Arguments:

        original {1d Array} -- Initial assignment of families

        choice_matrix {2d Array} -- Choices of each family

    

    Keyword Arguments:

        top_k {int} -- Number of choices to include (default: {2})

        num_fam {int} -- Number of families to sample (default: {8})

        n_time {int} -- Maximum execution time (default: {None})

        n_iter {int} -- Maximum number of executions (default: {None})

        early_stop {int} -- Stop after number of stagnant iterations (default: {np.inf})

        decrease {int} -- Decrease num_fam after number of stagnant iterations (default: {np.inf})

        random_state {int} -- Set NumPy random state for reproducibility (default: {42})

        verbose {int} -- Return current best after number of iterations (default: {1e4})

    

    Example:

        best = stochastic_product_search(

        choice_matrix=data.drop("n_people", axis=1).values, 

        top_k=3,

        num_fam=16, 

        original=prediction,

        n_time=int(3600),

        n_iter=int(1e5),

        early_stop=5e6,

        decrease=5e4,

        verbose=int(5e3),

        )

    

    Returns:

        [1d Array] -- Best assignment of families

    """

    np.random.seed(random_state)

    

    i = 0

    early_i = 0

    opt_time = time()  

    

    if n_time:

        max_time = opt_time + n_time

    else:

        max_time = opt_time

    

    if n_iter:

        max_iter = n_iter

    else:

        max_iter = 0

    

    best = original.copy()

    best_score = cost_function(best, family_sizes, family_cost_matrix, acc_costs)

    

    while ((max_time - time() > 0) or (not n_time)) and ((i < max_iter) or (not n_iter)) and (early_i < early_stop):

        fam_indices = np.random.choice(choice_matrix.shape[0], size=num_fam)

        

        for change in np.array(np.meshgrid(*choice_matrix[fam_indices, :top_k])).T.reshape(-1,num_fam):

            new = best.copy()

            new[fam_indices] = change



            new_score = cost_function(new, family_sizes, family_cost_matrix, acc_costs)



            if new_score < best_score:

                best_score = new_score

                best = new

                early_i = 0

            else:

                early_i += 1/num_fam

        

        if (early_i > decrease) and (num_fam > 2):

            num_fam -= 1

            early_i = 0

            if verbose:

                print(f"Decreasing sampling size to {num_fam}.")

                

            

        if verbose and i % verbose == 0:

            print(f"Iteration {i:05d}:\t Best score is {best_score:.2f}\t after {(time()-opt_time)//60:.0f} minutues {(time()-opt_time)%60:.0f} seconds.")

        i += 1

    

    print(f"Final best score is {best_score:.2f} after {(time()-opt_time)//60:.0f} minutues {(time()-opt_time)%60:.0f} seconds.")

    print(f"Each loop took {1000*(time()-opt_time)/i:.2f} ms.")

    return best
best = stochastic_product_search(

    prediction,

    choice_matrix=data.drop("n_people", axis=1).values, 

    top_k=1,

    num_fam=32, 

    n_time=int(60),

    early_stop=5e6,

    decrease=5e4,

    verbose=int(5e3),

)
submission['assigned_day'] = best

final_score = cost_function(best, family_sizes, family_cost_matrix, acc_costs)

submission.to_csv(f'submission_{int(final_score)}.csv')