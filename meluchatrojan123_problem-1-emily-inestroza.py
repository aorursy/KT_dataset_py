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
dodgers = pd.read_csv('/kaggle/input/dodgers-game-day-information/dodgers.csv')

dodgers.head(10)
dodgers.shape[0]
dodgers.iloc[-5:]
dodgers.attend.mean()
dodgers.attend.max()
dodgers.query('attend == 56000')
lad = dodgers.groupby(['cap', 'shirt', 'fireworks', 'bobblehead']).attend.mean().sort_values(ascending=False).plot.bar()

lad.set_xticklabels(['bobblehead', 'shirt', 'fireworks', 'none', 'cap'], rotation=0, fontsize=12)

lad.set_xlabel("Promotion Type", fontsize=12)

lad.set_ylabel("Attendance", fontsize=12)
dodgers.bobblehead.value_counts()