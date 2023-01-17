import os,sys,time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
fpath = '/kaggle/input/santa-2019-workshop-scheduling/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-2019-workshop-scheduling/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
data.head()
submission.head()
data['cost_0'] = 0

data['cost_1'] = 50

data['cost_2'] = data[['choice_2','n_people']].apply(lambda row:50+9*row.n_people, axis=1)

data['cost_3'] = data[['choice_3','n_people']].apply(lambda row:100+9*row.n_people, axis=1)

data['cost_4'] = data[['choice_4','n_people']].apply(lambda row:200+9*row.n_people, axis=1)

data['cost_5'] = data[['choice_5','n_people']].apply(lambda row:200+18*row.n_people, axis=1)

data['cost_6'] = data[['choice_6','n_people']].apply(lambda row:300+18*row.n_people, axis=1)

data['cost_7'] = data[['choice_7','n_people']].apply(lambda row:300+36*row.n_people, axis=1)

data['cost_8'] = data[['choice_8','n_people']].apply(lambda row:400+36*row.n_people, axis=1)

data['cost_9'] = data[['choice_9','n_people']].apply(lambda row:500+36*row.n_people+199*row.n_people, axis=1)

data['cost_otherwise'] = data[['choice_9','n_people']].apply(lambda row:500+36*row.n_people+398*row.n_people, axis=1)
data['visit_idx'] = -1

data['visit_day'] = -1

data['actual_cost'] = -1

workshop = {day:0 for day in range(1,101,1)}

data = data.sort_index(by=['n_people'],ascending=False)

for idx in range(len(data)):

    row = data.iloc[idx]

    checked = False

    for _choice in range(0,10,1):

        _idx = _choice

        _choice = 'choice_'+str(_choice)

        _day = row[_choice]

        if workshop[_day]+row['n_people']<300:

            row['visit_idx'] = _idx

            row['visit_day'] = _day

            row['actual_cost'] = row['cost_'+str(_idx)]

            workshop[_day] += row['n_people']

            checked = True

            break
data['actual_cost'] = data[['visit_day','cost_otherwise','actual_cost']].apply(lambda row:row['cost_otherwise'] if row['visit_day']==-1 else row['actual_cost'], axis=1)

def illegal_day(row):

    if row['visit_day']!=-1:

        return row['visit_day']

    for day in range(1,101,1):

        if workshop[day]+row['n_people']<300:

            workshop[day] += row['n_people']

            return day

data['visit_day'] = data[['visit_day','n_people']].apply(illegal_day, axis=1)
workshop
data.actual_cost.sum()
data[data.visit_day==2].n_people.sum()
submission['assigned_day'] = data['visit_day'].tolist()

score = data.actual_cost.sum()

submission.to_csv(f'submission_{score}.csv')

print(f'Score: {score}')