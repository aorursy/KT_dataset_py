"""
A simple example of linear regression to predict a county's log(population).
"""

import json, math

import matplotlib.pyplot as plt
import numpy as np

with open('/kaggle/input/us-county-data/states.json', 'r') as f:
    states = json.load(f)

X, Y = [], []
num_skipped = 0
for state_name in states:
    for county_name in states[state_name]:
        county = states[state_name][county_name]
        if county['unemployment_rate'] is None:
            num_skipped += 1
            continue
        if 'elections' not in county:
            num_skipped += 1
            continue
        elections = county['elections']["2016"]
        X.append([
            county['area'],
            county['avg_income'],
            county['unemployment_rate'],
            elections['gop'] / elections['total'],
            county['male'] / county['population'],
            1
        ])
        Y.append(math.log(county['population']))

print(f'Skipped {num_skipped} counties')
print(f'n = {len(X)}')

X = np.array(X)
Y = np.array(Y)

# Normalize features (ignoring the bias term)
X[:,:-1] -= X.mean(0)[:-1]
X[:,:-1] /= X.std(0)[:-1]

# Perform linear regression
w = np.linalg.lstsq(X, Y, rcond=-1)[0]
print('w =', w)

