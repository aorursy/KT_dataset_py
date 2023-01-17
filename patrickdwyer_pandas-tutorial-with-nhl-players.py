# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
player = pd.read_csv("../input/predict-nhl-player-salaries/train.csv",encoding = "ISO-8859-1").fillna(0)
player
player.head(10)
player = player.sort_values('Salary', ascending=False)
top10 = player.head(10)
print(top10)
player = player.set_index(['Last Name','First Name'])
print(player)
size = player.shape
print(size)
columns = player.columns
for i in columns:
    print(i)
print(player.loc['Kane'])
print(player.loc['Kane','Patrick'][0:5])
print(player.loc['Kane'].iloc[:1])
scorers = player.head(10).G.sum()
print("The 10 Highest Paid Players Scored",scorers,"total goals last season")
best_scorer = player.G.idxmax()
best_score = player.G.max()
print(best_scorer,"was the best goal scorer by scoring",best_score,"goals")
can_salaries = player[player.Nat == "CAN"].Salary.mean()
print("The Average Salary for a Canadian is $",can_salaries)
draft_results = player.groupby("DftRd").G.mean()
print(draft_results)
nationalities = player.groupby("Nat").size()
print(nationalities)
nationalities.plot.pie()