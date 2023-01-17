# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import seaborn as sns



data_bar = pd.read_csv("../input/bar_locations.csv")

data_parties = pd.read_csv("../input/party_in_nyc.csv")

data_test_parties = pd.read_csv("../input/test_parties.csv")

data_train_parties = pd.read_csv("../input/train_parties.csv")

data_bar.head()
data_parties.head()
data_test_parties.head()
lat = data_parties['Latitude']

long = data_parties['Longitude']



max_lat = max(lat)

max_long = max(long)

min_lat = min(lat)

min_long = min(long)
import matplotlib.pyplot as plt

sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'white'})

ax = plt.scatter(data_parties['Longitude'].values, data_parties['Latitude'].values,

              color='yellow', s=0.5, alpha=0.5)

ax.axes.set_title('Location of parties')

ax.figure.set_size_inches(6,5)

plt.grid(False)

plt.ylim(min_lat,max_lat)

plt.xlim(min_long,max_long)

plt.show()
plt.figure(figsize=(15,15))

location=data_bar.groupby("City")['Incident Zip'].count().reset_index().sort_values(by='City',ascending=False).reset_index(drop=True)

sns.barplot(x='City',y='Incident Zip',data=location)

plt.xticks(rotation=90)
plt.figure(figsize=(10,10))

location_type=data_parties.groupby("Location Type")['City'].count().reset_index().sort_values(by='Location Type',ascending=False).reset_index(drop=True)

sns.barplot(x='Location Type',y='City',data=location_type)

plt.xticks(rotation=45)