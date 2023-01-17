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

brazil_fire = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv", encoding = 'latin1')
# checking few entries



brazil_fire.head(20)
# dimension of the data

brazil_fire.shape
brazil_fire['date']
# We have the data for January to December for all the years from 1998 to 2017 for the different states of the Brazil



# Checking the last year in the data



brazil_fire.tail(4)
# Type of the column called as 'date'.

type(brazil_fire['date'])
# Converting the Series data into the date type

brazil_fire['date'] = pd.to_datetime(brazil_fire['date'], format = '%Y-%m-%d', dayfirst=True)
# Extracted the day and month in numerical format from the date 

brazil_fire['day'], brazil_fire['month_num'] = brazil_fire['date'].dt.day, brazil_fire['date'].dt.month
brazil_fire.head(10)
brazil_fire.describe()
# Getting the unique states from the state column

brazil_fire.state.unique()
# Getting the number of unique states from the state column



len(brazil_fire.state.unique())