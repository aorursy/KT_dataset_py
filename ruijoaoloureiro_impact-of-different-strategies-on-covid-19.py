# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
who_data = pd.read_csv("/kaggle/input/uncover/WHO/world-health-organization-who-situation-reports.csv", index_col=0)
interventions_data = pd.read_csv("/kaggle/input/uncover/HDE/acaps-covid-19-government-measures-dataset.csv", index_col=0)
n_cases = who_data.groupby('location').new_cases.sum()
new_cases = who_data.groupby('date').new_cases.sum()
n_deaths = who_data.groupby('location').new_deaths.sum()
new_deaths = who_data.groupby('date').new_deaths.sum()
n_interventions = interventions_data.groupby('country').country.count()
n_interventions_date = interventions_data.groupby('date_implemented').date_implemented.count()
plt.figure(figsize=(10,5))

interventions_vs_cases = sns.scatterplot(x=n_interventions, y=n_cases)

interventions_vs_cases.set(xlabel='nº of interventions', ylabel='nº of cases')

plt.show(interventions_vs_cases)
plt.figure(figsize=(10,5))

interventions_vs_deaths = sns.scatterplot(x=n_interventions, y=n_deaths)

interventions_vs_deaths.set(xlabel='nº of interventions', ylabel='nº of deaths')

plt.show(interventions_vs_deaths)
plt.figure(figsize=(10,5))

plt.title("Evolution of the number of new Covid-19 cases vs number of interventions")

ax = new_cases.plot(x="date", y="new_cases", legend=True, color="b")

ax2 = ax.twinx()

n_interventions_date.plot(x="date", y="n_interventions_date", ax=ax2, legend=False, color="y")

plt.xlabel("Date")
plt.figure(figsize=(10,5))

plt.title("Evolution of the number of new Covid-19 deaths vs number of interventions")

ax = new_deaths.plot(x="date", y="new_deaths", legend=True, color="b")

ax2 = ax.twinx()

n_interventions_date.plot(x="date", y="n_interventions_date", ax=ax2, legend=False, color="y")

plt.xlabel("Date")