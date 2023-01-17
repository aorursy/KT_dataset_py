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
data = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')

data.head()
data.rename(columns={"Beats.Per.Minute":"BPM",

                     "Valence.":"Valence",

                     "Acousticness..":"Acousticness",

                     "Loudness..dB..":"Loudness",

                     "Speechiness.":"Speechiness",

                     "Track.Name":"Track",

                     "Artist.Name":"Artist"},inplace=True)
data.drop(["Unnamed: 0","Length."], axis=1,inplace=True)



data.head()
import re 



for genre in data['Genre']:

    if re.search('pop', genre):

        data.loc[data.Genre == genre, 'Genre'] = 'pop'

        

data.head()

from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

  

data['Genre']= le.fit_transform(data['Genre']) 



data.head()
X = data.copy()

X.drop(["Track","Artist"], axis=1,inplace=True)

X.head()
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 29)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 29)

y_kmeans = kmeans.fit_predict(X)
data['ClusterID'] = y_kmeans

data.head()
data.sort_values(by=['ClusterID'], inplace=True)



for clusternumber in data.ClusterID.unique():

    print("\nPlaylist Number: " + str(clusternumber+1))

    print(data.loc[data.ClusterID == clusternumber, ['Track', 'Artist']])