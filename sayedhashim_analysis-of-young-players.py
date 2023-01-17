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
pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

df = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')


greater_than_90 = df[df['potential']>90]

sort_by_age = greater_than_90.sort_values('age',ascending=True)

sort_by_age.head(n=7)
less_than_19 = df[df['age']<=19]

sort_by_potential = less_than_19.sort_values('potential',ascending=False)

sort_by_potential.head(n=20)
players = df[(df['age']<=21) & (df['potential']>=85)]

clubs = players.groupby('club')

print (clubs.size().sort_values(ascending=False))
players = df[(df['age']<=21) & (df['potential']>=87)]

players_sorted = players.sort_values('value_eur')

players_sorted.head(n=20)