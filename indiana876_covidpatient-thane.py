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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

import folium

from sklearn.cluster import KMeans

import matplotlib as mpl
data = pd.read_csv('/kaggle/input/COVIDPatient.csv')
data.head()
unneccesary_cols = ['State', 'City Name','Locally Acquired Yes/No', 'Acquired Overseas: Yes/No', 'Contact Determined']
data_ = data.drop(unneccesary_cols, axis=1)
data_.info()
age = data_["Age"]

for i in range(0,7054):

    if age[i] == 'Not Available' :

        age[i] = '0'

        

data_["Age"] = data_["Age"].astype(str).astype(float)



age = data_["Age"]

for i in range(0,7054):

    if age[i] == '0' :

        age[i] = age.mean()
data_.info()
sns.countplot(data['Gender'],hue = data['Current Status'])

data['Gender'].value_counts()
sns.countplot(data['Current Status'])

data['Current Status'].value_counts()
plt.figure(figsize=(20,10))



sns.countplot(data['Ward Name'])

plt.xticks(rotation=45)

data['Ward Name'].value_counts()
sns.catplot(x="Current Status", y="Age", kind="violin", data=data_)
import re

data_["Result Date"] = data_["Result Date"].replace(to_replace ='-', value = '/', regex = True) 
data_['Result Date'] = pd.to_datetime(data_['Result Date'],errors='coerce').dt.date
df = data_.sort_values('Result Date')
plt.figure(figsize=(15,10))

sns.countplot(df['Result Date'])

plt.xticks(rotation=90)
plt.figure(figsize=(15,10))

plt.plot(data['Latitude'], data['Longitude'],'.')

plt.title('Plot of patient')

plt.show()
def get_kmeans_score(data_, center):

    '''

    returns the kmeans score regarding SSE for points to centers

    INPUT:

        data - the dataset you want to fit kmeans to

        center - the number of centers you want (the k value)

    OUTPUT:

        score - the SSE score for the kmeans model fit to the data

    '''

  

    kmeans = KMeans(n_clusters=center)

    model = kmeans.fit(data_)

    score = np.abs(model.score(data_))

    

    return score



scores = []

centers = list(range(1,25))



for center in centers:

    scores.append(get_kmeans_score(data.loc[: ,['Longitude','Latitude']], center))

    

    

plt.plot(centers, scores, linestyle='--', marker='o', color='b');

plt.xlabel('Number of Centers');

plt.ylabel('SSE');

plt.title('SSE vs. Number');
kmeans = KMeans(n_clusters=4)



kmeans.fit(data.loc[: ,['Longitude','Latitude']]).cluster_centers_
m = folium.Map(location=[19.198741,72.977948])
folium.Marker([19.19784481, 72.98635878],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)

folium.Marker([19.17479723, 73.03017528],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)

folium.Marker([19.19837351, 72.95482727],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)

folium.Marker([19.2414287 , 72.97456634],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)
m