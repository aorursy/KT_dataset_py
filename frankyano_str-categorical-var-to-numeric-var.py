import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



raw_data = {'patient': [1, 1, 1, 2, 2], 

        'obs': [1, 2, 3, 1, 2], 

        'treatment': [0, 1, 0, 1, 0],

        'score': ['strong', 'weak', 'normal', 'weak', 'strong']} 

df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])

df
#Create a function that converts all values of df['score'] into numbers

def score_to_numeric(x):

    if x == 'strong':

        return 3

    if x == 'normal':

        return 2

    if x == 'weak':

        return 1
df['score_num'] = df['score'].apply(score_to_numeric)

df