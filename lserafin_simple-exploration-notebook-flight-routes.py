# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

color = sns.color_palette()

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# get titanic & test csv files as a DataFrame

routes_df = pd.read_csv("../input/routes.csv")



# preview the data

routes_df.head()
print("----------------------------")

routes_df.info()
#check the which columns are available

routes_df.columns
# Plot Top 10 Airlines based on number of flights

cnt_srs = routes_df['airline'].value_counts().nlargest(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])

plt.xticks(rotation='vertical')

plt.xlabel('Airlines', fontsize=12)

plt.ylabel('Number of Flights', fontsize=12)

plt.show()
# Plot Top 10 Aircraft Types

cnt_srs = routes_df[' equipment'].value_counts().nlargest(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])

plt.xticks(rotation='vertical')

plt.xlabel('Aircraft Types', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
# Plot Top 10 Depature Aiports 

cnt_srs = routes_df[' source airport'].value_counts().nlargest(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])

plt.xticks(rotation='vertical')

plt.xlabel('Airport', fontsize=12)

plt.ylabel('Number of Flights', fontsize=12)

plt.show()
# Plot Top 10 Destination Aiports 

cnt_srs = routes_df[' destination apirport'].value_counts().nlargest(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Airport', fontsize=12)

plt.ylabel('Number of Flights', fontsize=12)

plt.show()
# Flights which have one stop

routes_df.loc[routes_df[' stops'] == 1]