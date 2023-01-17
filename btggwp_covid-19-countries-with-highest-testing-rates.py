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
df = pd.read_csv('/kaggle/input/covid19-testing-rate-all-countries/full-list-total-tests-for-covid-19.csv')
df.head(2)
df = df.drop_duplicates(subset=['Entity'], keep='last')
df.reset_index(drop=True, inplace=True)
df = df.sort_values(by=['Total tests'], ascending=False)
df = df[df['Entity'] != 'India, people tested']
df = df[df['Entity'] != 'United States, specimens tested (CDC)']
df.reset_index(drop=True, inplace=True)
df = df.head(20)
df
import plotly.express as px

fig = px.bar(df, x='Entity', y='Total tests', color='Entity', height=800, title='Top 20 Countries with maximum Testings done as of April 19, 2020')
fig.show()
