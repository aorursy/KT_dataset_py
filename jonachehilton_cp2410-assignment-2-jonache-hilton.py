# Retrieved from https://www.kaggle.com/xhlulu/santa-s-2019-stochastic-product-search
# Retrieved from https://www.kaggle.com/xhlulu/santa-s-2019-faster-cost-function-24-s
from itertools import product
from time import time
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from numba import njit, prange
import matplotlib.pyplot as plt

base_path = '/kaggle/input/santa-2019-revenge-of-the-accountants/'
data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')
submission = pd.read_csv(base_path + 'sample_submission.csv', index_col='family_id')


def _build_choice_array(data, n_days):
    choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
    choice_array_num = np.full((data.shape[0], n_days + 1), -1)

    for i, choice in enumerate(choice_matrix):
        for d, day in enumerate(choice):
            choice_array_num[i, day] = d
    
    return choice_array_num


def _precompute_accounting(max_day_count, max_diff):
    accounting_matrix = np.zeros((max_day_count+1, max_diff+1))
    # Start day count at 1 in order to avoid division by 0
    for today_count in range(1, max_day_count+1):
        for diff in range(max_diff+1):
            #revenge of the accountants adjustment
            for j in range (1, 6):
                accounting_cost = (today_count - 125.0) / 400.0 * (today_count**(0.5 + diff / 50.0))/j**2
                accounting_matrix[today_count, diff] = max(0, accounting_cost)
    
    return accounting_matrix


def _precompute_penalties(choice_array_num, family_size):
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
    
    penalty_matrix = np.zeros(choice_array_num.shape)
    N = family_size.shape[0]
    for i in range(N):
        choice = choice_array_num[i]
        n = family_size[i]
        
        for j in range(penalty_matrix.shape[1]):
            penalty_matrix[i, j] = penalties_array[n, choice[j]]
    
    return penalty_matrix


@njit
def _compute_cost_fast(prediction, family_size, days_array, 
                       penalty_matrix, accounting_matrix, 
                       MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS):

    N = family_size.shape[0]

    daily_occupancy = np.zeros(len(days_array)+1, dtype=np.int64)
    penalty = 0
    
    for i in range(N):
        n = family_size[i]
        d = prediction[i]
        
        daily_occupancy[d] += n
        penalty += penalty_matrix[i, d]

    relevant_occupancy = daily_occupancy[1:]
    incorrect_occupancy = np.any(
        (relevant_occupancy > MAX_OCCUPANCY) | 
        (relevant_occupancy < MIN_OCCUPANCY)
    )
    
    penalty = 100000000


    init_occupancy = daily_occupancy[days_array[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)

    accounting_cost = max(0, accounting_cost)
    

    yesterday_count = init_occupancy
    for day in days_array[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += accounting_matrix[today_count, diff]
        yesterday_count = today_count

    return penalty, accounting_cost, daily_occupancy


def build_cost_function(data, N_DAYS=100, MAX_OCCUPANCY=300, MIN_OCCUPANCY=125):

    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)


    choice_array_num = _build_choice_array(data, N_DAYS)
    penalty_matrix = _precompute_penalties(choice_array_num, family_size)
    accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY, max_diff=MAX_OCCUPANCY)
    

    def cost_function(prediction):
        penalty, accounting_cost, daily_occupancy = _compute_cost_fast(
            prediction=prediction,
            family_size=family_size, 
            days_array=days_array, 
            penalty_matrix=penalty_matrix, 
            accounting_matrix=accounting_matrix,
            MAX_OCCUPANCY=MAX_OCCUPANCY,
            MIN_OCCUPANCY=MIN_OCCUPANCY,
            N_DAYS=N_DAYS
        )
        
        return penalty + accounting_cost
    
    return cost_function

# Build your "cost_function"
cost_function = build_cost_function(data)

# Run it on default submission file
original = submission['assigned_day'].values
original_score = cost_function(original)

def stochastic_product_search(top_k, fam_size, original, choice_matrix, 
                              disable_tqdm=False, verbose=10000,
                              n_iter=500, random_state=2019):

    best = original.copy()
    best_score = cost_function(best)
    
    np.random.seed(random_state)

    for i in tqdm(range(n_iter), disable=disable_tqdm):
        t1 = time()
        time_array.append(t1 - start_time)
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
    return best 
choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values

time_array = []
start_time = time()


# Running the algorithm with three rounds
best = stochastic_product_search(
    choice_matrix=choice_matrix, 
    top_k=5,
    fam_size=5, 
    original=original, 
    n_iter=1,
    disable_tqdm=False,
    verbose=2000
)

end_time = time()
total_time = end_time-start_time
time_array.append(total_time)
print(f"Algorithm 1 took", total_time, "seconds")

bars = range(1, len(time_array) + 1)
y_pos = np.arange(len(bars))
plt.plot(y_pos, time_array)

plt.title('Stochastic Time Graph')
plt.xlabel('No. of Iterations')
plt.ylabel('Time (Sec)')

plt.show()

#retreived from https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
prediction = submission['assigned_day'].values
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
@njit()
def jited_cost(prediction, desired, family_size, penalties):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
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
        penalty += penalties[n, n_choice]

    accounting_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        for j in range (1, 6):
            accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))/j**2
        

    penalty += accounting_cost
    return penalty

start_time = time()

print("Best Score:", jited_cost(prediction, desired, family_size, penalties))

end_time = time()
total_time = end_time-start_time
print(f"Algorithm 2 took", total_time, "seconds")