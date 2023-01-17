import numpy as np

import pandas as pd

import os

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



train_data_file = '../input/train.csv'



def get_7day(m):

    return [(int(x)-1)%7 for x in m]



def get_part(n=1100):

    d = {}

    for i, (k,v) in enumerate(sorted(day_visits.items())):

        if i >= n:

            break

        d[k] = v

    return d
if not os.path.exists(train_data_file):

    print(f'Train file {train_data_file} not found')

else:

    train = pd.read_csv(train_data_file)

    train = train.visits.str.split()

    visitors = np.array([get_7day(x) for x in train.tolist()])
day_visits = {}

for visitor in train:

    for day in visitor:

        day_visits[int(day)] = day_visits.get(int(day), 0) + 1
plt.title('Visiting histogram')

plt.xlabel('Days (3 first weeks)')

plt.ylabel('Visitors')

d = get_part(21)

plt.bar(d.keys(), d.values(), color='g')

plt.show()
pwd = [0 for _ in range(7)]

week_w = []

for k,v in day_visits.items():

    day = (int(k)-1)%7

    pwd[day] += v

for d in pwd:

    week_w.append(d/sum(pwd))

plt.title('Day of Week Popularity')

plt.xlabel('Day of Week')

plt.ylabel('Visitors')

plt.bar(range(1,8), pwd, color='g')

plt.show()

print('Days weights:', week_w)

visitors_weights = []

for i, visitor in enumerate(visitors):

    print(f'{i*100//len(visitors)}%', end='\r')

    v_weight = []

    weight = 1.0

    for k, day in enumerate(visitor):

#         weight = weight + week_w[day] * 0.1

        weight += 0.1

        v_weight.append(weight)

    visitors_weights.append(v_weight)

print('Builded!')
from sklearn.utils.extmath import weighted_mode



predict = []

for i in range(300000):

    print(f'{i*100//300000}%', end='\r')

    res = int(weighted_mode(visitors[i], visitors_weights[i])[0][0]) + 1

    predict.append(f' {res}')

print('Complete!')

df = pd.DataFrame({'id': np.arange(1, 300001), 'nextvisit' : predict})

df.to_csv('./solution.csv', index=False)

print('Saved!')