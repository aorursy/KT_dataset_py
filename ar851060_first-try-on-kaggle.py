# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.optimize import curve_fit

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding="ISO-8859-1")

data.head()
data.shape
data.describe().T
data.info()
data.isnull().sum()
data.rename(columns = { "Unnamed: 0" : "id",

                        "Acousticness.." : "Acousticness",

                        "Track.Name" : "Track_Name" ,

                        "Valence." : "Valence",

                        "Length." : "Length",

                        "Loudness..dB.." : "Loudness_dB" ,

                        "Artist.Name" : "Artist_Name",

                        "Beats.Per.Minute" :"Beats_Per_Minute",

                        "Speechiness." : "Speechiness"},inplace = True)

data = data.sort_values(['id'])

data = data.reindex(data['id'])

data = data.drop('id', axis = 1)

data = data.loc[:49,:]

data.tail()
data.groupby('Genre').Genre.count()
data['General_Genre'] = ['hip hop' if each =='atl hip hop'

                      else 'hip hop' if each =='canadian hip hop'

                      else 'hip hop' if each == 'trap music'

                      else 'pop' if each == 'australian pop'

                      else 'pop' if each == 'boy band'

                      else 'pop' if each == 'canadian pop'

                      else 'pop' if each == 'dance pop'

                      else 'pop' if each == 'panamanian pop'

                      else 'pop' if each == 'pop'

                      else 'pop' if each == 'pop house'

                      else 'electronic' if each == 'big room'

                      else 'electronic' if each == 'brostep'

                      else 'electronic' if each == 'edm'

                      else 'electronic' if each == 'electropop'

                      else 'rap' if each == 'country rap'

                      else 'rap' if each == 'dfw rap'

                      else 'escape room' if each == 'hip hop'

                      else 'latin' if each == 'latin'

                      else 'r&b' if each == 'r&n en espanol'

                      else 'raggae' for each in data['Genre']]

data.head()
print(data.groupby('General_Genre').General_Genre.count())

sns.countplot(x = 'General_Genre', data = data)
data.groupby('Artist_Name').Artist_Name.count().reset_index(name='count').sort_values(['count'],ascending=False).head(5)

data[data.Artist_Name =="Ed Sheeran"]
data_sorted = data.sort_values(by = "Popularity", ascending = False)

data_sorted["Rank"] = data_sorted["Popularity"].rank(ascending = False) 

data_sorted.head()
plt.subplots(figsize=(7, 7))

sns.regplot(x=data_sorted["Rank"],y=data_sorted["Popularity"])

plt.subplots(figsize=(7, 7))

sns.regplot(x=data_sorted["Energy"],y=data_sorted["Popularity"])