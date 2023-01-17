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
# import data

drugs = pd.read_csv('/kaggle/input/drugs.csv')



# Opiods has no header so add one

opiods = pd.read_csv('/kaggle/input/opiods.csv', names=['opiods'])

drugs.columns
drugs = drugs.drop(columns=['Unnamed: 2',

                    'Unnamed: 3',

                    'Unnamed: 4',

                    'Unnamed: 5',

                   'Unnamed: 6',

                   'Unnamed: 7',

                   'Unnamed: 8',

                   'Unnamed: 9',

                   'Unnamed: 10'])
# Look at data

drugs.head()

# This way of reading is messy but gets our data into a simple list

import csv



with open('/kaggle/input/opiods.csv', newline='') as f:

    # Reads each row into a list

    reader = csv.reader(f)

    opiods_temp = list(reader) 



opiods = []

for row in opiods_temp:

    opiods.append(row[0])    # Get the string from each list



# Idk why the first item reads weird might just be this machine

opiods[0] = opiods[0][1:]

    

    

print(opiods_temp[:10])    # Print first ten items of original data

print(opiods)    # list of opiods
generics = drugs['generic_name']

genericsList = list(generics)

print(genericsList[:10])
booleans = []



for d in genericsList:

    if d in opiods:

        booleans.append(1)

    else:

        booleans.append(0)

        

print(booleans[:20])
# assign booleans to a new column

drugs['is_opiod'] = booleans  

drugs.head(20)