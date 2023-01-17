import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#imports

import pandas as pd

import numpy as np
#load dataset 

electricity = pd.read_csv("../input/bdgp2-further-cleaned-datasets/electricity_cleaned_new.csv")
electricity.head()
#show types of the values 

electricity.dtypes
#drop unnamed column 

electricity = electricity.drop("Unnamed: 0", axis = 1)
#change to DateTime format

electricity["timestamp"] = pd.to_datetime(electricity["timestamp"], format = "%Y-%m-%d %H:%M:%S")
#set time as index 

electricity = electricity.set_index("timestamp")
#average by week

electricity = electricity.resample("W").mean()
mouse_electricity = pd.DataFrame()

M = [col for col in electricity.columns if 'Mouse' in col]

mouse_electricity[M] = electricity[M]
mouse_electricity.head()
#sum of columns in kWh

mouse_electricity.sum(axis = 0)