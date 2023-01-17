# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/us-accidents/US_Accidents_May19.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import pyplot

from datetime import datetime

import dateutil.parser
dataset = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
type(dataset["Start_Time"][1])
dt_object = dataset["Start_Time"][1:30].apply(lambda x: dateutil.parser.parse(x))
dt_object.head()
dataset["Time_added"] = dataset["Start_Time"].apply(lambda x: dateutil.parser.parse(x))
dataset["Time_added"][0:4].apply(lambda timestamp: timestamp.hour)
a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.countplot(x=dataset["Time_added"].apply(lambda timestamp: timestamp.hour))
a4_dims = (10, 7)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.countplot(x=dataset["Time_added"].apply(lambda timestamp: timestamp.month))
time_list = dataset["Time_added"].apply(lambda timestamp: timestamp.hour).value_counts()
time_list