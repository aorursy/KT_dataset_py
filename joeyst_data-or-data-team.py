# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pickle

import random

ethnicity = ['English','Australian','Irish','Scottish','Italian','German','Chinese','Indian','Greek','Dutch','Aboriginal/Torres strait islander','other']

nationality = ['Africa', 'Australia', 'Asia', 'Europe', 'North America', 'South America']

age = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+']

gender = ['Male', 'Female', 'Other']

dataset_size = 10000

dataset = []

for i in range(dataset_size):

    dataset.append({})

    dataset[i]['Ethnicity'] = random.choice(ethnicity)

    dataset[i]['Nationality'] = random.choice(nationality)

    dataset[i]['Age'] = random.choice(age)

    dataset[i]['Gender'] = random.choice(gender)

pickle.dump(dataset, open('dataset.pickle', 'wb'))

print('Done')