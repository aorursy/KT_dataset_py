# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
crime_df = pd.read_csv('../input/crime_and_weather.csv')
crime_df.head()
crime_df['DATE_OCCURED'] = crime_df['DATE_OCCURED'].astype('datetime64[ns]')
crime_df['DATE_OCCURED'] = crime_df['DATE_OCCURED'].dt.date

crime_df['date'] = crime_df['date'].astype('datetime64[ns]')
crime_df['date'] = crime_df['date'].dt.date

crime_df['high'] = crime_df['high'].astype('int32')
crime_df.shape
crime_df.loc[crime_df['CRIME_TYPE'] == 'ASSAULT']
crime_df.loc[(crime_df['CRIME_TYPE'] == 'ASSAULT') & (crime_df['high'] >= 80)]
assault_df = pd.read_csv('../input/assault_counts.csv')
assault_df.head()
assault_df.plot.scatter(x='high', y='NUMBER_ASSAULTS')
import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x="high", y="NUMBER_ASSAULTS", data=assault_df)
assault_df['high'].corr(assault_df['NUMBER_ASSAULTS'])