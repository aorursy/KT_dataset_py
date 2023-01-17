# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
!du -h ../input/*
df = pd.read_csv('../input/year_latest.csv')
df.info()
# The area and country codes

country_df = pd.read_csv('../input/country_eng.csv')



country_df.head()
country_df.info()
# Joined trade records with Country name

joined_df = pd.merge(df, country_df, on=['Country'])
joined_df.head()
joined_df.describe().T
grouped_by_area = joined_df[['Year', 'VY', 'Area']].groupby(['Area', 'Year'], as_index=False)

vys = grouped_by_area.aggregate(np.sum)
# Plot VY transition

def plot_vys(column_value, column_name):

    data = vys[vys[column_name] == column_value]

    plt.plot(data['Year'], data['VY'], label=column_value)

    

areas = np.unique(country_df['Area'].values)

    

plt.figure(figsize=(7.5, 8))

for area in areas:

    plot_vys(area, 'Area')

    

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()