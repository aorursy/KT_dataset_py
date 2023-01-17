# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import libraries and magic functions

import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline
# read data frame

df = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')

df.head()
# Data Overview

df.info()

df.Measure.unique()

df.Border.unique()
# check for nan values

df.isna().sum()
# change type of Date column from object to time

df['Date'] = pd.to_datetime(df['Date']).dt.date
# filter dataframe to only Passengers (no Vehicles)

df2 = df[df['Measure'].isin(['Personal Vehicle Passengers','Bus Passengers','Pedestrians','Train Passengers'])]

df2.head()
# show Passenger Count by Border

df_passengers_total = df2.groupby(['Border']).agg({'Value': 'sum'})

df_passengers_total
# Plot % of Passenger Count by Border

df_passengers_total.plot(kind='pie', subplots=True, legend=False, autopct='%.2f%%', figsize=(7,7), title= "Comparison Count of Border Crossings", fontsize=12)
# Show Passenger Count by Measure of Crossing and by Border

df_measure = df2.groupby(['Border','Measure']).agg({'Value':'sum'})

df_measure
# Plot Passenger Count by Measure of Crossing and by Border

plt.figure(figsize=(12,7))

sns.barplot(data=df2, x="Border", y="Value", hue="Measure")

plt.box(False)

plt.title("Measures of Border Crossing - by Border", fontweight='bold', fontsize=15)
# Plotting all Crossings by Measure and Border

plt.figure(figsize=(15,7))

plt.box(False)

sns.barplot(data=df, x='Border', y='Value', hue='Measure')

plt.title("Crossings by all Measures and by Border")
# plotting Border Crossings throughout Time - Passengers + Vehicles

plt.figure(figsize=(15,7))

sns.lineplot(data=df, x='Date',y='Value', hue='Border')

plt.box(False)

plt.title('Border Crossings throughout Time - Passengers & Vehicles', fontweight='bold', fontsize=15)
# plotting Border Crossings throughout Time - only Passengers

plt.figure(figsize=(15,7))

sns.lineplot(data=df2, x='Date',y='Value', hue='Border')

plt.box(False)

plt.title('Border Crossings throughout Time - only Passengers', fontweight='bold')
# Plotting Crossings throughout Time by Measure Type

plt.figure(figsize=(20,10))

sns.lineplot(data=df, x='Date', y='Value',hue='Measure', palette='bright')

plt.box(False)