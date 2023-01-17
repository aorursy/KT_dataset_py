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
players = pd.read_csv('/kaggle/input/fifa19/data.csv')

players.head()
players.shape
players.iloc[0]
players.query('Age >= 40')
players.Age.mean()
players.groupby('Age').size().plot.barh()
players.groupby('Nationality').size().sort_values(ascending=False).head(10)
min_country = players.groupby('Nationality').size().sort_values(ascending=False).idxmin()

players[ players.Nationality == min_country ]




