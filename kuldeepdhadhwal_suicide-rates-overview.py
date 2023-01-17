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
data = pd.read_csv('../input/master.csv')
data.head()
data['generation'].unique()
df = data[['HDI for year']]
df.fillna(df.mean(), inplace=True)
age_against_suicides = data[['age','suicides/100k pop']]

age_against_suicides.head()

data.describe()
import matplotlib.pyplot as plt

plt.plot(data['year'],data['gdp_per_capita ($)'],color='green')

plt.show()

def plot_bar_x():

    # this is for plotting purpose

    plt.bar(data['year'], data['suicides_no'])

    plt.xlabel('year', fontsize=5)

    plt.ylabel('gdp', fontsize=5)

#     plt.xticks(index, label)

    plt.title('Suicides Per Year')

    plt.show()
plot_bar_x()
plt.plot(data['year'],data['suicides_no'],color='green')

plt.show()