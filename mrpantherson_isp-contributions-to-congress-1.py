# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

congress = pd.read_csv('../input/congress_isp_donations.csv')
by_price = congress[congress['donation'] > 0].sort_values(by='donation')

print(by_price[:1])

print(by_price[-1:])
sns.barplot(x=['democrat', 'republican'], y=[len(congress[congress['party'] == 'Democrat']), len(congress[congress['party'] == 'Republican'])])

plt.title('Number of Replicans vs Democrats')

plt.show()
house = congress[congress['chamber'] == 'house']

house_state = house.groupby(by='state')['donation'].agg(['sum']).sort_values(by='sum', ascending=False)

house_state = house_state[:30]

sns.barplot(y=house_state.index, x=house_state['sum'])

plt.title('House of Representative Total Bribe Per State')

plt.show()
house = congress[congress['chamber'] == 'house']

house_state = house.groupby(by='state')['donation'].agg(['mean']).sort_values(by='mean', ascending=False)

house_state = house_state[:30]

sns.barplot(y=house_state.index, x=house_state['mean'])

plt.title('House of Representative Avg Bribe Per State')

plt.show()
senate = congress[congress['chamber'] == 'senate']

senate_state = senate.groupby(by='state')['donation'].agg(['sum']).sort_values(by='sum', ascending=False)

senate_state = senate_state[:30]

sns.barplot(y=senate_state.index, x=senate_state['sum'])

plt.title('Senate Total Bribe Per State')

plt.show()
sns.barplot(x=['senate', 'house', 'congress'], y=[house['donation'].mean(), senate['donation'].mean(), congress['donation'].mean()])

plt.title('Average Bribe for each Chamber')

plt.show()