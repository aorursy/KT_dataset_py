# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from csv import reader # using list of list to print top 10 AOE

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
open_file = open('/kaggle/input/college-basketball-dataset/cbb.csv')

read_file = reader(open_file)

ncaa = list(read_file)



ncaa = ncaa[1:]
def sort(dictionary, show=False):

    table_display = []

    for key in dictionary:

      if float(key) > 100:

        key_val_as_tuple = (dictionary[key], key)

        table_display.append(key_val_as_tuple)



    table_sorted = sorted(table_display, key=lambda x: x[1], reverse=True)

    if show == True:

        for entry in table_sorted:

            print(entry[1], ':', entry[0])

    return(table_sorted)



aoe_dict = {}



for team in ncaa:

  name = team[0] + '_' + team[23]

  aoe = team[4]

  if name not in aoe_dict:

    aoe_dict[aoe] = name
print(aoe_dict)
aoe = sort(aoe_dict)

for entry in aoe[:9]:

  print(entry[1], ':', entry[0])
x = []

y = []

for entry in aoe[:10]:

  x.append(entry[0])

  y.append(float(entry[1]))



plt.scatter(y,x)

plt.show()