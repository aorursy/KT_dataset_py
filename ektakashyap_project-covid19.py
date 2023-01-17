# 1.1 Call libraries and modules

%reset -f

# 1.2 For data manipulations

import numpy as np

import pandas as pd

# 1.3 For plotting

import matplotlib.pyplot as plt

import matplotlib as mlp

import seaborn as sns

sns.set(font_scale=1.4)

import plotly.express as px



import datetime as dt

from scipy import stats

import datetime as dt



# 1.4 OS related

import os



# 1.5 For data processing

from sklearn.preprocessing import StandardScaler



import sys

import warnings



warnings.filterwarnings('ignore')

%matplotlib inline
# 1.6 Display output not only of last command but all commands in a cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# 1.7 Set pandas options to display results

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 1000
# 2.0 Read file,

patients = pd.read_csv("../input/combined_raw-data_123.csv")
patients.dtypes
print("Data Shape : Rows = {} , Columns = {}".format(patients.shape[0],patients.shape[1]))
patients.head()
from datetime import timedelta

patients['Status Change Date'] = pd.to_datetime(patients['Status Change Date'])

patients['Date Announced'] = pd.to_datetime(patients['Date Announced'])



patients['Duration of Any Status'] = patients['Status Change Date'] - patients['Date Announced']

patients['Duration of Any Status'] = patients['Duration of Any Status'].dt.days



patients['Status Change Date'] = patients['Status Change Date'].dt.strftime('%d-%m-%Y')

patients['Date Announced'] = patients['Date Announced'].dt.strftime('%d-%m-%Y')
patients.info()
patients.drop(['Notes', 'Contracted from which Patient (Suspected)', 'Source_1', 'Source_2', 'Source_3'], axis = 1, inplace = True)

patients.head(10)
patients.info()
age = patients['Age Bracket']

status = patients['Current Status']

age_bins = [0,20,30,40,50,60,70,80,90,100]

# Set the color palette

sns.set_palette(sns.color_palette("bright"))

# Plot

plt.figure(figsize=(12,6))

fig1 = sns.countplot(x=pd.cut(age, age_bins), hue=status)

plt.xticks(rotation=0)

fig1.set_facecolor('black')

plt.xlabel("Age Buckets")

plt.yscale('log')

plt.title("Covid-19: Age Buckets")

plt.grid(True)

plt.show()