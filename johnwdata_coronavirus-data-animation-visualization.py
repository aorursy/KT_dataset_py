import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import matplotlib.ticker as ticker

from IPython.display import HTML

from matplotlib import ticker



import random



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/coronavirus-covid19-cases-by-us-state/covid_by_state.csv")
df.shape
df.head()
# Add column for percent change of cases by state

df['pct change cases'] = round(df.groupby(['state'])['cases'].pct_change() * 100, 1)

# Add column for number of new cases by state

df['new cases'] = df.groupby('fips')['cases'].diff()

# Add column for percent change of deaths by state

df['pct change deaths'] = round(df.groupby(['state'])['deaths'].pct_change() * 100, 1)

# Add column for total cases

df["total usa cases"] = df.groupby("date")["cases"].transform("sum")

# Add column for total deaths

df["total usa deaths"] = df.groupby("date")["deaths"].transform("sum")

df.fillna(0, inplace=True)
# get 7 day average for each state

dates = df.date.unique()

states = df.state.unique()

for i in range(len(dates)):

  currDate = dates[len(dates) - (i + 1)]

  for state in states:

    df.loc[(df['state'] == state) & (df.date == currDate), '7 day ave % change cases'] = round(df[(df.state == state) & (df.date.isin(dates[-(7+i):len(dates) - i]))]['pct change cases'].mean(), 2)

    df.loc[(df['state'] == state) & (df.date == currDate), '7 day ave new cases'] = round(df[(df.state == state) & (df.date.isin(dates[-(7+i):len(dates) - i]))]['new cases'].mean(), 1)
# Create a list of dates one month after first cases were reported in the US. Cases were limited early on so I skipped ahead one month.

dates_month_skipped = dates[30:]

# Create a list of colors, then randomly assign each state a color

colors = ['lightcoral','chocolate', 'y', 'silver', 'moccasin', 'coral', 'mediumturquoise', 'deepskyblue', 'darkseagreen', 'mediumslateblue', 'plum' ]

state_colors = dict()

for state in states:

    state_colors[state] = random.choice(colors)



fig, ax = plt.subplots(figsize=(15,10))

plt.box(False)

plt.close()



def update_barchart(date):

    df_top10 = df[df.date == date].sort_values('cases', ascending=False).head(10)[::-1]

    ax.clear()

    ax.barh(df_top10.state, df_top10.cases, color=[state_colors[state] for state in df_top10.state])

    

    # iterate over values to plot labels and values

    for i, (val, name) in enumerate(zip(df_top10['cases'], df_top10['state'])):

        ax.text(val, i, name, size=14,weight=600, ha = 'right', va="bottom")

        ax.text(val, i, f'{val:,d}' , size=14, ha = 'left', va='center')

    

    # clear state names from yticks since they are already being shown

    ax.set_yticks([])

    # move xticks to the top

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors="#777777", labelsize=16)

    # add grid lines

    ax.grid(which='major', axis='x', linestyle='--')

    # add date to right side of the canvas

    ax.text(1, 0.4, date, transform=ax.transAxes, color="#777777", size=48, ha="right", weight=600)

    ax.text(0, 1.12, "States With Most COVID-19 Cases By Date"  ,color ="#777777", transform=ax.transAxes, size=26, ha="left", weight=600)



animator = FuncAnimation(fig, update_barchart, frames = dates_month_skipped, interval=400)

HTML(animator.to_jshtml())
fig, ax = plt.subplots(figsize=(8,8))

plt.close()

data = dict()

labels=["NY-NJ", "Rest Of US"]

colors = ['#adb0ff', '#e48381']

ny_nj = ["New York", "New Jersey"]



for date in dates:

    new_york_cases = 0

    total_cases = df['total usa cases'][df.date==date].values[0]

    for state in ny_nj:

        try:

          new_york_cases += df.cases[(df.date==date) & (df.state == state)].values[0]

        except:

          new_york_cases += 0

    percent_ny_nj = (new_york_cases / total_cases) * 100

    data[date] = [percent_ny_nj, 100-percent_ny_nj]



def update(date):

    ax.clear()

    ax.pie(data[date],labels=labels, autopct='%1.1f%%', colors=colors)

    title = date + "\n" + f'{int(df["total usa cases"][df.date==date].values[0]):,.0f}' + " US cases" 

    ax.set_title(title)

    

mpl.rcParams['font.size'] = 18.0

animator = FuncAnimation(fig, update, frames = dates_month_skipped, interval=350)

HTML(animator.to_jshtml())
states = sorted(states)

fig, ax = plt.subplots(nrows=11, ncols=5, figsize=(40, 60))



index = 0

for row in ax:

  for col in row:

    X = np.arange(len(df[df.state == states[index]]['new cases']))    

    col.bar(X, df[df.state == states[index]]['new cases'])

    col.plot(X, df[df.state == states[index]]['7 day ave new cases'], color='red')

    col.set_title(states[index], loc='left', va = 'top')

    index += 1

plt.show()