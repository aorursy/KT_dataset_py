# for numerical analysis

import numpy as np

from numpy import inf



# to store and process data in dataframe

import pandas as pd



# to interface with operating system

import os



# for basic visualization

import matplotlib.pyplot as plt



# for advanced visualization

import seaborn as sns; sns.set()



# for interactive visualization

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go



# for offline interactive visualization

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)



# for trendlines

import statsmodels



# data manipulation

from datetime import datetime as dt

from scipy.stats.mstats import winsorize
# gdp data

gdp = pd.read_csv('../input/gdp-data-updated/GDP Data updated.csv') 

#call the package pandas, call the method: read_csv, assign the csv file to the data frame: country_wise



# Replace missing values '' with NAN and then 0

gdp = gdp.replace('', np.nan).fillna(0)



gdp.info() #tells you what are the datatypes for each variable, and their dimensions

gdp.head(63) #quick glimpse into the first 10 observations of the data
# Import Data

df = pd.read_csv('../input/gdp-data-updated/GDP Data updated.csv')



# Get the Peaks and Troughs

data = df['traffic'].values

doublediff = np.diff(np.sign(np.diff(data)))

peak_locations = np.where(doublediff == -2)[0] + 1



doublediff2 = np.diff(np.sign(np.diff(-1*data)))

trough_locations = np.where(doublediff2 == -2)[0] + 1



# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

plt.plot('date', 'traffic', data=df, color='tab:blue', label='Air Traffic')

plt.scatter(df.date[peak_locations], df.traffic[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')

plt.scatter(df.date[trough_locations], df.traffic[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')



# Annotate

for t, p in zip(trough_locations[1::5], peak_locations[::3]):

    plt.text(df.date[p], df.traffic[p]+15, df.date[p], horizontalalignment='center', color='darkgreen')

    plt.text(df.date[t], df.traffic[t]-35, df.date[t], horizontalalignment='center', color='darkred')



# Decoration

plt.ylim(50,750)

xtick_location = df.index.tolist()[::6]

xtick_labels = df.date.tolist()[::6]

plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)

plt.title("Gross Domestic Product - Singapore", fontsize=22)

plt.yticks(fontsize=12, alpha=.7)



# Lighten borders

plt.gca().spines["top"].set_alpha(.0)

plt.gca().spines["bottom"].set_alpha(.3)

plt.gca().spines["right"].set_alpha(.0)

plt.gca().spines["left"].set_alpha(.3)



plt.legend(loc='upper left')

plt.grid(axis='y', alpha=.3)

plt.show()
