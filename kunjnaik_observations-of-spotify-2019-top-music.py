# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/top50spotify2019/top50.csv', encoding='latin1')
df
#We count each word in each song title not including words surrounded by brackets or following a dash
df['words_in_title'] = 0

for i in range(0,50):

    word_count = 0

    words = df['Track.Name'][i].split()

    for word in words:

            if '(' in word:

                break

            elif '-'in word:

                break

            else:

                word_count += 1

    df['words_in_title'][i] = word_count

df
df.columns
#These values are researched from Google and entered into 'val' manually

val = [6,4,9,9,5,9,1,8,1,1,4,7,1,6,10,15,11,5,1,1,1,1,1,1,1,6,1,1,9,15,5,9,8,7,12,7,6,9,15,1,6,14,12,3,3,5,6,5,5,9]

df['Mainstream'] = 0

for i in range(0,50):

    #val = input('How many years has ' + df['Artist.Name'][i] + ' been mainstream?')

    df['Mainstream'][i] = val[i]
df
ap = 0

ap = df.groupby('Artist.Name')['Unnamed: 0'].nunique()

print(ap)
#We manually enter the origin of each artist, researched on Google

origin = ['NA','LA','NA','EU','NA','EU','NA','EU','NA','NA','LA','EU','EU','LA','NA','NA','LA','NA','NA','NA','EU','NA','LA','LA','NA','NA','NA','NA','LA','LA','NA','NA','LA','NA','NA','EU','LA','EU','NA','NA','EU','NA','NA','NA','LA','NA','LA','NA','NA','EU']

df['Origin'] = 0

for i in range(0,50):

    #origin = input('Where is ' + df['Artist.Name'][i] + ' from?')

    df['Origin'][i] = origin[i]

df
na = 0

la = 0

eu = 0

for i in range(0,50):

    if(df['Origin'][i] == 'NA'):

        na+=1

    elif(df['Origin'][i] == 'LA'):

        la+=1

    else:

        eu+=1

slices = [na/50,la/50,eu/50]

lbl = ['North America','Latin America', 'Europe']



plt.pie(slices, labels=lbl)

print('North America: '+str(na/50))

print('Latin America: '+str(la/50))

print('Europe: '+str(eu/50))
maximum = 0

minimum = 500

full_range = 0

middle = 0



for i in range(0,50):

    if df['Beats.Per.Minute'][i] < minimum:

        minimum = int(df['Beats.Per.Minute'][i]) 

    elif df['Beats.Per.Minute'][i] > maximum:

        maximum = int(df['Beats.Per.Minute'][i])

full_range = maximum - minimum

middle = full_range//2 + minimum
fast = 0

slow = 0

for i in range(0,50):

    if df['Beats.Per.Minute'][i] > middle:

        fast += 1

    else:

        slow += 1

print('Maximum BPM: '+str(maximum))

print('Middle BPM: ' + str(middle))

print('Minimum BPM: '+str(minimum))

print('Proportion of fast BPM tracks (above 137BPM): ' + str(fast/50))

print('Proportion of slow BPM tracks (137BPM or less): ' + str(slow/50))
BPM = df['Beats.Per.Minute']

energy = df['Energy']
BPM_energy = np.column_stack((BPM,energy))

kmeans1 = KMeans(n_clusters = 3)

kmeans1.fit(BPM_energy)

y_kmeans1 = kmeans1.predict(BPM_energy)
plt.scatter(BPM,energy)

plt.xlabel('Beats per minute')

plt.ylabel('Energy')

centers = kmeans1.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
dncblt = df['Danceability']
bpm_dncblt = np.column_stack((BPM,dncblt))

kmeans2 = KMeans(n_clusters = 3)

kmeans2.fit(bpm_dncblt)

y_kmeans1 = kmeans2.predict(bpm_dncblt)
plt.scatter(BPM,dncblt)

plt.xlabel('Beats per minute')

plt.ylabel('Danceability')

centers2 = kmeans2.cluster_centers_

plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)
valence = df['Valence.']
bpm_valence = np.column_stack((BPM,valence))

kmeans3 = KMeans(n_clusters=2)

kmeans3.fit(bpm_valence)

y_kmeans3 = kmeans3.predict(bpm_valence)
plt.scatter(BPM,valence)

plt.xlabel('Beats per minute')

plt.ylabel('Valence')

centers3 = kmeans3.cluster_centers_

plt.scatter(centers3[:, 0], centers3[:, 1], c='black', s=200, alpha=0.5)
high_live = 0

low_live = 0

for i in range(0,50):

    if df['Liveness'][i] > 25:

        high_live+=1

    elif df['Liveness'][i] < 26:

        low_live+=1

print('Proportion of tracks likely recorded live is: ' +str(high_live/50))

print('Proportion of tracks likely not produced live is: ' +str(low_live/50))
df['Length.'].describe()
high_acou = 0

low_acou = 0

for i in range(0,50):

    if df['Acousticness..'][i] > 50:

        high_acou+=1

    elif df['Acousticness..'][i] < 51:

        low_acou+=1

print('Proportion of songs which we can consider acoustic: '+str(high_acou/50))

print('Proportion of songs which we cannot consider acoustic: '+str(low_acou/50))