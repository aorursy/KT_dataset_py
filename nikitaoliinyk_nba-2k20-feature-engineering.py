import numpy as np

import pandas as pd

import csv

from datetime import datetime

import re

import math



data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")

data.head()
data['team'].unique()
data.info()
data.describe()
def clean_data(data):

    data['salary'] = data['salary'].apply(lambda x: int(x[1:]))

    data['jersey'] = data['jersey'].apply(lambda x: int(x[1:]))

    data['b_day'] = data['b_day'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').date())

    data['height'] = data['height'].apply(lambda x: float(x[2+x.find('/'):]))

    data['weight'] = data['weight'].apply(lambda x: float(x[2+x.find('/'):-4]))

    data['draft_round'] = data['draft_round'].apply(lambda x: int(x) if len(x) == 1 else 0)

    data['draft_peak'] = data['draft_peak'].apply(lambda x: int(x) if 1<=len(x)<=2 else 0)

    data['college'] = data['college'].fillna('no education')

    data['team'] = data['team'].fillna('no team')



clean_data(data)
#find age of each player



def age_(birthday):

    today = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d').date()

    age = today.year - birthday.year

    return int(age)



data['age'] = data['b_day'].apply(lambda x: age_(x))
data.info()
data
from sklearn import preprocessing

from scipy import stats



data_dummy = pd.get_dummies(data, columns=['team', 'position','draft_round', 'country'], drop_first= True)

data_dummy = data_dummy.drop(['full_name', 'draft_peak', 'b_day', 'jersey', 'college'], axis = 1)

X, y = data_dummy.drop(['salary'], axis = 1), data_dummy['salary']

data_dummy