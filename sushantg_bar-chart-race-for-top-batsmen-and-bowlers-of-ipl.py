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
bcrbowlers = pd.read_csv('/kaggle/input/top-run-scorers-and-wicket-takers-in-ipl/iplbowlers.csv', index_col = 'PLAYER')

bcrbatsmen = pd.read_csv('/kaggle/input/top-run-scorers-and-wicket-takers-in-ipl/iplbatsmen.csv', index_col = 'PLAYER')

bcrbowlers
# Calcuate cumulative sum along the seasons and sort dfs as per 2019 (total across all seasons) values

bcrbowlers = bcrbowlers.cumsum(axis=1).sort_values(by='2019', ascending = False)

bcrbatsmen = bcrbatsmen.cumsum(axis=1).sort_values(by='2019', ascending = False)



# Selecting the top 50 bowlers/batsmen and transposing dfs to prepare it for BCR (wideformat)

bcrbowlers = bcrbowlers[:50].T

bcrbatsmen = bcrbatsmen[:50].T
!pip install bar_chart_race

import bar_chart_race as bcr
bcr.bar_chart_race(bcrbowlers, n_bars=10, fixed_max=True, steps_per_period=5,

                   period_length= 1500, filter_column_colors = True, cmap = 'Plotly', 

                   title = 'Total Wickets Taken by the Top 10 Bowlers in IPL Over Different Seasons ')
bcr.bar_chart_race(bcrbatsmen, n_bars=10, fixed_max=True, steps_per_period=5, 

                   period_length= 1500,filter_column_colors = True, cmap = 'Plotly',

                   title = 'Total Runs Scored by the Top 10 Batsmen in IPL over Different Seasons')