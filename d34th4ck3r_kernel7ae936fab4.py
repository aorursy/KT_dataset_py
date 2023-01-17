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
import pandas as pd
inp = pd.read_csv("/kaggle/input/celebrity-deaths/celebrity_deaths_4.csv", encoding= 'unicode_escape')

inp
inp['age'].hist(bins=10)
inp['age'].describe()
a = inp[inp['age']<68]
b = inp[inp['age']>=68][inp['age']<80]
c = inp[inp['age']>=80][inp['age']<87]
d = inp[inp['age']>=87]
sample_size = 100

frames = [a.sample(sample_size), b.sample(sample_size), c.sample(sample_size), d.sample(sample_size)]

balanced_data = pd.concat(frames)
balanced_data
balanced_data.describe()
balanced_data['age'].hist(bins=10)
inp.describe()
inp['age'].quantile(0.10) #This is 10%tile
inp['age'].quantile(0.25) #This is 25%tile. Notice that it is same as above