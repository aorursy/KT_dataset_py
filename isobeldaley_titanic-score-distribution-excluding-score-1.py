# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 110) # Set max rows to display to 110

import plotly.express as px # Data Visualisation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import public leaderboard (date of download 24/12/2019)

df = pd.read_excel('/kaggle/input/titanic-publicleaderboard.xlsx')

df.head()
# Drop all records for which Score=1

df = df.loc[df['Score']!=1]

df.head()
# Calculate score percentiles (1% intervals)

percentiles = np.percentile(df['Score'], np.arange(0, 100, 1)) 



# Create labels for each percentile (1% intervals)

labels = np.arange(0,100,1)
# Put percentiles and percentile labels into dataframe

percentiles_df = pd.DataFrame({'Percentile(%)':labels, 'Percentile_Score':percentiles})



# Display percentile DataFrame

percentiles_df
# Plot percentiles on line chart

fig = px.line(percentiles_df, x="Percentile(%)", y="Percentile_Score", title='Distribution of Scores: Titanic ML Competition')

fig.show()