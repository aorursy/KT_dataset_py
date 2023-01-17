import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date

%config InlineBackend.figure_format = 'retina'

sns.set_style("whitegrid")

dat = pd.read_csv('../input/database.csv', parse_dates=['Date'])
f, ax = plt.subplots(figsize=(16,6))

pd.pivot_table(dat, index=[dat.Date.dt.year, dat.Race], aggfunc="size").unstack(

                                            level=1).plot(kind='bar', stacked=True, ax=ax)

ax.set_title("Annual Executions by Race in the US (1976-2016)", fontweight='bold', fontsize=20)

ax.set_xlabel("")

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.xaxis.grid(False)

ax.set_ylabel("");
f, ax = plt.subplots(figsize=(16,6))

pd.pivot_table(dat, index=[dat.Date.dt.year, dat.Region], aggfunc="size").unstack(

                                            level=1).plot(kind='bar', stacked=True, ax=ax)

ax.set_title("Annual Executions by Region in the US (1976-2016)", fontweight='bold', fontsize=20)

ax.set_xlabel("")

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.xaxis.grid(False)

ax.set_ylabel("");
f,ax = plt.subplots(figsize=(16,6))

pd.pivot_table(data= dat.loc[dat.Date.dt.year>=1984, :], index=[dat.Date.dt.year, dat.State],

               aggfunc='size').sort_values(ascending=False).head(30).unstack(

               level=1).plot(kind='bar', stacked=True, ax=ax)

ax.set_title("States with 10 or more executions in a year", fontweight='bold', fontsize=20)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.xaxis.grid(False)

ax.set_xlabel("")

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=360)

ax.legend();
f, ax = plt.subplots()

dat.groupby(dat.Date.dt.year).Age.mean().plot(ax=ax)

ax.set_ylim(bottom=0, top=50)

ax.set_xlim(left=1977, right=2016)

ax.set_title("Average age of executed prisoners by year (1976-2016)", fontweight='bold', size=14)

ax.set_xlabel("Year", fontweight='bold')

ax.set_ylabel("Age", fontweight='bold')

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.xaxis.grid(False)

ax.fill_between(x=dat.groupby(dat.Date.dt.year).Age.mean().index,

                y1=dat.groupby(dat.Date.dt.year).Age.mean().values, color='#6BA3C3');
f, ax = plt.subplots(figsize=(16,6))

pd.pivot_table(dat, index=[dat.Date.dt.year, dat.Method], aggfunc="size").unstack(

                                            level=1).plot(kind='bar', stacked=True, ax=ax)

ax.set_title("Annual Executions by Method in the US (1976-2016)", fontweight='bold', fontsize=20)

ax.set_xlabel("")

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.xaxis.grid(False)

ax.set_ylabel("");