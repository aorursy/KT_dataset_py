import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm
def get_one_hot_data(filedata):

    total_days = 1099

    n_weeks = total_days//7

    n_clients = filedata.shape[0]



    visits_strings = filedata.values[:, 1]

    all_visits = np.zeros((n_clients, n_weeks, 7))

    for i in tqdm(range(n_clients), total=n_clients):

        visits = np.array(visits_strings[i].split(), dtype=int )

        visits -= 1

        visits_one_hot = np.eye(total_days)[visits].sum(axis=0)    

        visits_one_hot = visits_one_hot.reshape((1, n_weeks, 7))

        all_visits[i] = visits_one_hot

    return all_visits
filedata = filedata = pd.read_csv('../input/train.csv')

data = get_one_hot_data(filedata)
def get_weights(n_weeks, n_dropped=15):

    weights = (-np.arange(n_weeks)+n_weeks)  / n_weeks

    weights = np.power(weights, 1.15)

    weights = weights / weights.sum()

    weights[-n_dropped:]=0

    return weights[::-1]
n_weeks = 1099 // 7 

n_clients = 300000

weights = get_weights(n_weeks).reshape((-1, 1))

visits = data[:n_clients,:n_weeks]

answer = []

for weeks in tqdm(visits, total=len(visits)):    

    weeks = weeks * weights    

    weeks = weeks.mean(axis=0)

    day = np.argmax(weeks) + 1

    answer.append(day)

answer = np.array(answer)
inds = np.arange(1, 300001).reshape((-1, 1))

answer = answer.reshape((-1, 1))

arr = np.concatenate((inds, answer), axis=1).astype(int)

df = pd.DataFrame(arr, columns=["id","nextvisit"])

df.to_csv("/kaggle/working/solution.csv", index=False)