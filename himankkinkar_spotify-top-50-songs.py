# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
songs = pd.read_csv(r"/kaggle/input/top50spotify2019/top50.csv", encoding = "ISO-8859-1",index_col=0)
songs.info()
songs.head()
songs.groupby('Genre').describe()['Popularity'].sort_values(by='mean',ascending=False)
sns.set_style('whitegrid')

sns.jointplot(x='count',y='mean',data=songs.groupby('Genre').describe()['Popularity'])
songs_mat = songs[['Beats.Per.Minute', 'Energy',

       'Danceability', 'Loudness..dB..', 'Liveness', 'Valence.', 'Length.',

       'Acousticness..', 'Speechiness.', 'Popularity']]
songs_mat.head()
plt.figure(figsize=(14,8))

sns.heatmap(songs_mat.corr(),annot=True)