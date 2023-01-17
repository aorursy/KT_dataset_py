import numpy as np

import pandas as pd

from numba import njit, prange
data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')

family_size = data.n_people.values.astype(np.int8)



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





def score(prediction):

    fc, ac, l, h = cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix)

    return (fc + ac) + (l + h) * 1000000



fam = pd.read_csv("/kaggle/input/santa-workshop-tour-2019/family_data.csv")

pref = fam.values[:,1:-1]
pred = pd.read_csv('/kaggle/input/santa-ip/submission.csv', index_col='family_id').assigned_day.values

init_score = score(pred)



print(init_score)
sub = pd.read_csv('/kaggle/input/santa-ip/submission.csv')



# !cp /kaggle/input/santa-ip/submission.csv ./submission_{best_score}.csv

!cp /kaggle/input/santa-ip/submission.csv ./submission_72398.91780918743.csv
def seed_finding(seed, prediction_input):

    prediction = prediction_input.copy()

    np.random.seed(seed)

    best_score = score(prediction)

    original_score = best_score

    print("SEED: {}   ORIGINAL SCORE: {}".format(seed, original_score))

    for t in range(100):

        for i in range(5000):

            for j in range(10):

                di = prediction[i]

                prediction[i] = pref[i, j]

                cur_score = score(prediction)



                KT = 1

                if t < 5:

                    KT = 1.5

                elif t < 10:

                    KT = 4.5

                else:

                    if cur_score > best_score + 100:

                        KT = 3

                    elif cur_score > best_score + 50 :

                        KT = 2.75

                    elif cur_score > best_score + 20:

                        KT = 2.5

                    elif cur_score > best_score + 10:

                        KT = 2

                    elif cur_score > best_score:

                        KT = 1.5

                    else:

                        KT = 1



                prob = np.exp(-(cur_score - best_score) / KT)

                if np.random.rand() < prob:

                    best_score = cur_score

                else:

                    prediction[i] = di

        if best_score < original_score:

            print("NEW BEST SCORE on seed {}: {}".format(seed, best_score))

            sub.assigned_day = prediction

            sub.to_csv(f'submission_{best_score}.csv', index=False)

            break



    if best_score >= original_score:

        print("UNLUCKY on seed {} for 100 runs, no impovement.".format(seed))



    return prediction, best_score
best_score = init_score



for seed in range(1201, 1225):

    pred, best_score = seed_finding(seed, pred)

    if best_score < init_score:

        init_score = best_score

    else:

        best_score = init_score

    pred = pd.read_csv(f'submission_{best_score}.csv', index_col='family_id').assigned_day.values
!ls