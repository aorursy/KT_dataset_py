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
import random

import collections
def how_many_rolls(n,verbose=False):

    sides=list(range(n))

    keep_rolling=True

    rolls=0

    while keep_rolling:

        sides=random.choices(sides,k=6)

        rolls+=1

        if verbose: print(f"After {rolls} rolls, new sides: {sides}")

        keep_rolling=n>sum([x==sides[0] for x in sides])

    return(rolls)
how_many_rolls(6,True)

trials=1000000

sides_on_a_die=6

roll_tallies=[]

for _ in range(trials):

    roll_tallies.append(how_many_rolls(sides_on_a_die))

print(f"After {trials} trials, there were a total of {sum(roll_tallies)} rolls, for an average of {sum(roll_tallies)/trials} rolls per trial.")

roll_tallies.sort()

collections.Counter(roll_tallies)