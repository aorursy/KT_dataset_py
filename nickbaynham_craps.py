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



def roll():

    roll1 = random.randrange(1,7)

    roll2 = random.randrange(1,7)

    return roll1, roll2



def display(roll):

    roll1, roll2 = roll

    print(f'{roll1} and {roll2} = {sum(roll)}')

    

nextRoll = roll()

display(nextRoll)

score = sum(nextRoll)



if score in (7, 11):

    print("You won!")

elif score in (2, 3, 12):

    print("Craps! Sorry, you lost!")

else:

    print("Point: ", score)

    point = score

    go = 1

    while go:

        nextRoll = roll()

        display(nextRoll)

        score = sum(nextRoll)

        if score in (7, 11):

            print("You won!")

            go = 0

        elif score in (2, 3, 12):

            print("Craps! Sorry, you lost!")

            go = 0

        elif score == point:

            print("Point made, you win!")

            go = 0