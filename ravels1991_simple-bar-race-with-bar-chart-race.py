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

import time
start_time = time.time()
!pip install bar_chart_race
# import the package
import bar_chart_race as bcr
from IPython.display import HTML
#reading the csv file
df = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')
df.head()
cases = df.pivot_table(index='date', columns='state', values='cases')
deaths = df.pivot_table(index='date', columns='state', values='deaths')
region_cases = df.pivot_table(index='date', columns='region', values='cases')
region_deaths = df.pivot_table(index='date', columns='region', values='deaths')
cases_bcr_html = bcr.bar_chart_race(
    df=cases,
    filename=None,
    figsize=(7, 4),
    title='Casos de COVID-19 por Estados.',
    cmap='tab20c')
HTML(cases_bcr_html)
deaths_bcr_html = bcr.bar_chart_race(
    df=deaths,
    filename=None,
    figsize=(7, 4),
    title='Mortes COVID-19 por Estados.',
    cmap='tab20c')
HTML(deaths_bcr_html)
region_cases_bcr_html = bcr.bar_chart_race(
    df=region_cases,
    orientation='v',
    filename=None,
    figsize=(7, 4),
    title='Casos de COVID-19 por região.',
    cmap='tab20c')
HTML(region_cases_bcr_html)
region_deaths_bcr_html = bcr.bar_chart_race(
    df=region_deaths,
    orientation='v',
    filename=None,
    figsize=(7, 4),
    title='Mortes de COVID-19 por região.',
    cmap='tab20c')
HTML(region_deaths_bcr_html)
print(f"This kernel took {(time.time() - start_time)/60:.2f} minutes to run")