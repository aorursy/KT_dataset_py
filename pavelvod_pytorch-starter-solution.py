import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from pathlib import Path



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable





root_path = Path(r'/kaggle/input/santa-workshop-tour-2019')



fpath = root_path / 'family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = root_path / 'sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
n_people = data['n_people'].values

m = data.iloc[:, :-1].values

mat = np.zeros(shape=(5000, 100))

penalties = {n: [0, 50, 50 + 9 * n, 100 + 9 * n, 200 + 9 * n, 200 + 18 * n, 300 + 18 * n, 300 + 36 * n, 400 + 36 * n, 500 + 36 * n + 199 * n] for n in np.unique(n_people)}

for f in np.arange(mat.shape[0]):

    mat[f] = 500 + 36 * n_people[f] + 398 * n_people[f]

    

for f in np.arange(m.shape[0]):

    for c in np.arange(m.shape[1]):

        mat[f, m[f, c]-1] = penalties[n_people[f]][c]
family_size_dict = data[['n_people']].to_dict()['n_people']



cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].T.to_dict()



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



# from 100 to 1

days = list(range(N_DAYS,0,-1))



family_size_ls = list(family_size_dict.values())

choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]



# Computer penalities in a list

penalties_dict = {

    n: [

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

    for n in range(max(family_size_dict.values())+1)

} 



def cost_function(prediction):

    penalty = 0

    violations = 0

    # We'll use this to count the number of people scheduled each day

    daily_occupancy = {k:0 for k in days}

    

    # Looping over each family; d is the day, n is size of that family, 

    # and choice is their top choices

    for n, d, choice in zip(family_size_ls, prediction, choice_dict_num):

        # add the family member count to the daily occupancy

        daily_occupancy[d] += n



        # Calculate the penalty for not getting top preference

        if d not in choice:

            penalty += penalties_dict[n][-1]

        else:

            penalty += penalties_dict[n][choice[d]]



    # for each date, check total occupancy

    #  (using soft constraints instead of hard constraints)

    for v in daily_occupancy.values():

        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

            violations += 1

#             penalty += 100000000



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_cost = max(0, accounting_cost)

    

    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = daily_occupancy[days[0]]

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count



    return penalty, accounting_cost, violations
class Model(nn.Module):

    def __init__(self, mat, n_people):

        super().__init__()

        self.mat = torch.from_numpy(mat).type(torch.float32)

        self.n_people = torch.from_numpy(n_people).type(torch.float32)

        self.weight = torch.nn.Parameter(data=torch.Tensor(5000, 100), requires_grad=True)

        self.weight.data.uniform_(0, 5)   

        

    def forward(self):

        x = (F.softmax(self.weight) * self.mat).sum()

        y = ((torch.transpose(F.softmax(self.weight), 0 , 1)@ self.n_people - 200) ** 2).sum()

        return  x, y
model = Model(mat, n_people)

best_score = 10e10

best_pos = None

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(15000):  # loop over the dataset multiple times

    # zero the parameter gradients

    optimizer.zero_grad()

    x, y = model()

    loss = x + y

    loss.backward()

    optimizer.step()

    



    if epoch % 100 == 0:

        pos = model.weight.argmax(1).numpy()

        a, b, v = cost_function(pos+1)

        score = a + b

        if score < best_score:

            best_score = score

            best_pos = pos

        x = np.round(x.item(),3)

        y = np.round(y.item(),3)

        print(f'{epoch}\t{x}\t{y}\t{score}\t{a}\t{b}\t{v}')
best_score
submission['assigned_day'] = best_pos+1

submission.to_csv(f'submission_{best_score}.csv')