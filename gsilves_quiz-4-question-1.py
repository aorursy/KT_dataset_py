# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
NFL = pd.read_csv('/kaggle/input/nfl-offense-cleaned-2017to2007/nfl_offense_cleaned.csv')
NFL.head()
NFL.shape
NFL.sort_values('YDS', ascending=False).iloc[:5]
NFL.loc[NFL.COMP.idxmax()].PLAYER
NFL.groupby("LONG").max()
NFL.query('RATE > 158.2').loc[:, ['PLAYER']]
NFL.mean().round(2)
NFL.ATT.plot.hist()
NFL.POS.value_counts().plot.pie()