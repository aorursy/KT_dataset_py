# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import csv





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/lottery-br/megas.xls')

df
rb1 = rb2 = rb3 = rb4 = rb5 = rb6 = df

rb1 = rb1.drop(['lottery','ball_02','ball_03','ball_04','ball_05','ball_06' ],axis=1)

rb2 = rb2.drop(['lottery','ball_01','ball_03','ball_04','ball_05','ball_06' ],axis=1)

rb3 = rb3.drop(['lottery','ball_02','ball_01','ball_04','ball_05','ball_06' ],axis=1)

rb4 = rb4.drop(['lottery','ball_02','ball_03','ball_01','ball_05','ball_06' ],axis=1)

rb5 = rb5.drop(['lottery','ball_02','ball_03','ball_04','ball_01','ball_06' ],axis=1)

rb6 = rb6.drop(['lottery','ball_02','ball_03','ball_04','ball_05','ball_01' ],axis=1)



rb1 = rb1.to_numpy(dtype=str)

rb2 = rb2.to_numpy(dtype=str)

rb3 = rb3.to_numpy(dtype=str)

rb4 = rb4.to_numpy(dtype=str)

rb5 = rb5.to_numpy(dtype=str)

rb6 = rb6.to_numpy(dtype=str)



register_ball = np.concatenate((rb1, rb2, rb3, rb4, rb5, rb6), axis=0)

#register_ball.sort(axis=1)

register_ball

#register_ball = pd.DataFrame({'Date': register_ball[:, 0], 'Number': register_ball[:, 1]})

#register_ball.groupby(['Date'])
df2 = pd.DataFrame({'Draw1': register_ball[:, 0], 'Draw2': register_ball[:, 1]})

df2
df3 = df2.drop_duplicates(subset='Draw2', keep='first')

df3.sort_values(by=['Draw1'], ascending=True)
df3.sort_values(by=['Draw2'], ascending=True)