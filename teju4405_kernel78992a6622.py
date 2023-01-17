# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
spotify = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='latin-1')

spotify.head()
spotify=spotify.drop(spotify.iloc[:,0:1],axis=1)

spotify.head()
spotify.describe()
spo=spotify.set_index(['Track.Name','Artist.Name'])

spo.head()
spo['Popularity'].nlargest(3)
from matplotlib import pyplot as plt

import seaborn as sns
plt.figure(figsize=(10,10))

track=spotify['Track.Name']

genre=spotify['Energy']

beats=spotify['Beats.Per.Minute']

popularity=spotify['Popularity']

plt.bar(track,beats,linewidth=2)

plt.bar(track,genre,linewidth=2,color='red')

ax=plt.gca()

for item in ax.xaxis.get_ticklabels():

    item.set_rotation(90)

plt.legend(['Beats.Per.Minute','Energy'],loc='best')

plt.box(on=None)

plt.title('Comparing Beats with Energy')
plt.figure(figsize=(10,10))

dance=spotify['Danceability']

length=spotify['Length.']



plt.bar(track,length,linewidth=2,color='yellow')

plt.bar(track,dance,linewidth=2,color='violet')

ax=plt.gca()

for item in ax.xaxis.get_ticklabels():

    item.set_rotation(90)

plt.legend(['Danceability','Length.'],loc='best')

plt.box(on=None)

plt.title('Comparing Beats with Energy')
plt.figure(figsize=(10,10))

acoust=spotify['Acousticness..']

speech=spotify['Speechiness.']

plt.plot(track,acoust,linewidth=2)

plt.plot(track,speech,linewidth=2)

ax=plt.gca()

for item in ax.xaxis.get_ticklabels():

    item.set_rotation(90)

plt.box(on=None)

plt.legend(['Acousticness','Speechiness'],loc='best',frameon=False)

plt.title('Comparing Speechiness and Acousticness')
plt.figure(figsize=(10,10))

loud=spotify['Loudness..dB..']

live=spotify['Liveness']

valence=spotify['Valence.']

plt.plot(track,loud,linewidth=2)

plt.plot(track,live,linewidth=2)

plt.plot(track,valence,linewidth=2)

ax=plt.gca()

for item in ax.xaxis.get_ticklabels():

    item.set_rotation(90)

plt.box(on=None)

plt.legend(['Loudness','Liveness','Valence'],loc='best',frameon=False)

plt.title('Comparing Loudness, Liveness and Valence')
plt.figure(figsize=(15,15))

bars=plt.bar(track,popularity,linewidth=22,color='green')

ax=plt.gca()

for item in ax.xaxis.get_ticklabels():

    item.set_rotation(90)

for bar in bars:

    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height(),str(int(bar.get_height()))+'%',ha='center',color='black',fontsize=11)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

plt.legend(['Popularity'],loc='best')

plt.box(on=None)