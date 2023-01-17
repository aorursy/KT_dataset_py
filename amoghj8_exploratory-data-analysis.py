# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH = "../input/"
data = pd.read_csv(f'{PATH}athlete_events.csv')
data.head()
data[(data.Sex=='F') & (data.Year==1996)].Age.sort_values().head(1)
data[(data.Sex=='M') & (data.Year==1996)].Age.sort_values(ascending=True).head(1)
non_gym = data[(data.Sex=='M') & (data.Year==2000) & (data.Sport!='Gymnastics')]['Name'].unique().shape[0]
gym = data[(data.Sex=='M') & (data.Year==2000) & (data.Sport=='Gymnastics')]['Name'].unique().shape[0]
round(gym/non_gym * 100, 1)
data[(data.Sex=='F') & (data.Year==2000) & (data.Sport=='Basketball')]['Height'].describe()
data[(data.Year==2002) & (data.Weight==data[data.Year==2002]['Weight'].max())].loc[:,['Name','Sport']]
data[data.Name=='Pawe Abratkiewicz']['Year'].unique()
data[(data.Sport=='Tennis') & (data.Team=='Australia') & (data.Year==2000) & (data.Medal=='Silver')]
data[(data.Team=='Switzerland') & (data.Year==2016)].loc[:,['Medal']].count()
data[(data.Team=='Serbia') & (data.Year==2016)].loc[:,['Medal']].count()
data.loc[data.Year==2014, 'Age'].value_counts()
data[(data.Season=='Summer') & (data.City=='Lake Placid')]
data[(data.Season=='Winter') & (data.City=='Sankt Moritz')].head()
data[(data.Year==1996)]['Sport'].unique().size
data[(data.Year==2016)]['Sport'].unique().size