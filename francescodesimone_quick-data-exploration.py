# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt







data = pd.read_csv( "../input/data.csv")

data.head()



# Any results you write to the current directory are saved as output.
data.groupby("Event").count().plot(title="Event type count", kind='bar')


#this is marathon classific

maraton = data[data['Event'] == "Marathon"]



maraton
#this is half marathon leaderboard

h_maraton = data[data['Event'] == "Half marathon"]

h_maraton