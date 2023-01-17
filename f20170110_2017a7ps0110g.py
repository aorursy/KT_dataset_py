# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing neccessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Reading the given dataset 'data.csv'

data = pd.read_csv('/kaggle/input/dmassign1/data.csv')
#Checking which columns have null values in them

data.columns[data.isnull().any()]
#Analysis

data.dtypes['Col30']
#After analyzing the problem with data

#If any entry contains '?' or '', we simply replace it with NULL values

data = data.replace(to_replace=['?',''], value=np.nan)
#We make every column  that we had an issue with to float

#Note: Ideally, a regex matching must have been done before blindly just converting it to float

#      however it works so :Peace

dropCols = [30,31,34,36,37,38,39,40,43,44,46,47,48,49,50,51,53,56,138,139,140,141,142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,159,160,161,162,173,174,175,179,180,181,182,183,184,185,186,187,188]

colDType = {}

for col in dropCols:

    colDType[data.columns[col]] = 'float'

data = data.astype(colDType)

#data.dtypes
#Analysing the discrete attributes

print('Col189', data['Col189'].unique())

print('Col190', data['Col190'].unique())

print('Col191', data['Col191'].unique())

print('Col192', data['Col192'].unique())

print('Col193', data['Col193'].unique())

print('Col194', data['Col194'].unique())

print('Col195', data['Col195'].unique())

print('Col196', data['Col196'].unique())

print('Col197', data['Col197'].unique())

print('Col198', data['Class'].unique())

data.columns[data.isnull().any()]
#Domain Knowledge

data = data.drop(['ID'], 1)
#Since col197 has la and LA which seem to be the same thing, we map them to the same string

def SegregateCol197(val):

    if val=='XL': return 'XL'

    if val=='SM' or val=='sm': return 'SM'

    if val=='ME' or val=='M.E.' or val=='me': return 'ME'

    if val=='LA' or val=='la': return 'LA'

    return np.nan

data['Col197'] = data['Col197'].apply(SegregateCol197)
print(data['Col192'].value_counts())

print(data['Col193'].value_counts())

print(data['Col194'].value_counts())

print(data['Col195'].value_counts())

print(data['Col196'].value_counts())

print(data['Col197'].value_counts())
#We try to replace nan values with the highest occuring entries

#Note: The frequncy of each entry has been found out using the command

# data[<colname>].value_counts()

data['Col192'] = data['Col192'].replace(to_replace=np.nan, value='p2')

data['Col193'] = data['Col193'].replace(to_replace=np.nan, value='F0')

data['Col194'] = data['Col194'].replace(to_replace=np.nan, value='ad')

data['Col195'] = data['Col195'].replace(to_replace=np.nan, value='Jb3')

data['Col196'] = data['Col196'].replace(to_replace=np.nan, value='H3')

data['Col197'] = data['Col197'].replace(to_replace=np.nan, value='ME')
#Analysing the discrete attributes

print('Col189', data['Col189'].unique())

print('Col190', data['Col190'].unique())

print('Col191', data['Col191'].unique())

print('Col192', data['Col192'].unique())

print('Col193', data['Col193'].unique())

print('Col194', data['Col194'].unique())

print('Col195', data['Col195'].unique())

print('Col196', data['Col196'].unique())

print('Col197', data['Col197'].unique())

print('Class', data['Class'].unique())
#For rest of the data, we simply replace the given data by its median

for col in data.columns[data.isnull().any()]:

    if col == 'Class':

        continue

    data[col] = data[col].fillna(data[col].mean())
#Applying One hot encoding for the discrete attributes

#data['Col189']

data = pd.get_dummies(data, columns=['Col189', 'Col190', 'Col191', 'Col192', 'Col193', 'Col194', 'Col195', 'Col196', 'Col197'])
data.columns[data.isnull().any()]
#Performing Min_Max Normalization

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data)

sData = pd.DataFrame(np_scaled, columns=data.columns)

sData.head()
sData = sData.drop(['Class'], 1)
#Performing PCA to reduce the number of attributes to 60

from sklearn.decomposition import PCA

model = PCA(n_components=60)

modelData = model.fit(sData).transform(sData)

modelData.shape
#Applying

from sklearn.cluster import KMeans

kmean = KMeans(n_clusters = 5, random_state = 42, max_iter=20000, n_init=50)

clusterLabels = kmean.fit_predict(modelData)

print(len(clusterLabels))

print(clusterLabels)

#dbscan = DBSCAN(eps=1, min_samples=9)

#pred = dbscan.fit_predict(modelData)

#clusterLabels = dbscan.labels_

#labels1, counts1 = np.unique(clusterLabels, return_counts=True)

freq0 = {1:0, 2:0, 3:0, 4:0, 5:0}

freq1 = {1:0, 2:0, 3:0, 4:0, 5:0}

freq2 = {1:0, 2:0, 3:0, 4:0, 5:0}

freq3 = {1:0, 2:0, 3:0, 4:0, 5:0}

freq4 = {1:0, 2:0, 3:0, 4:0, 5:0}

for i in range(1300):

    if clusterLabels[i] == 0:

        freq0[data['Class'][i]] = freq0[data['Class'][i]] + 1

    elif clusterLabels[i] == 1:

        freq1[data['Class'][i]] = freq1[data['Class'][i]] + 1

    elif clusterLabels[i] == 2:

        freq2[data['Class'][i]] = freq2[data['Class'][i]] + 1

    elif clusterLabels[i] == 3:

        freq3[data['Class'][i]] = freq3[data['Class'][i]] + 1

    else:

        freq4[data['Class'][i]] = freq4[data['Class'][i]] + 1

print(freq0, freq1, freq2, freq3, freq4)

freq = [freq0, freq1, freq2, freq3, freq4]

avSet = set([1,2,3,4,5])

assSet = set([0,1,2,3,4])

mp = {}

i = 0

while i < 5:

    #Keymax = max(freq[i], key=freq[i].get)

    #Find the index which has the max assigned

    ind = 0

    maxTillNow = 0

    for j in assSet:

        Keymax = max(freq[j], key=freq[j].get)

        if freq[j][Keymax] >= maxTillNow:

            ind = j

            maxTillNow = freq[j][Keymax]

    

    kM = max(freq[ind], key=freq[ind].get)

    if kM in avSet:

        mp[ind] = kM

        avSet.remove(kM)

        assSet.remove(ind)

        i=i+1

    else:

        freq[ind][kM] = 0

print(mp)
print(mp)

classL = []

indexL = []

for i in range(1300, 13000):

    st = 'id' + str(i)

    indexL.append(st)

    classL.append(mp[clusterLabels[i]])



sub = pd.DataFrame(columns=['ID', 'Class'])

sub['ID'] = np.array(indexL)

sub['Class'] = np.array(classL)

sub
score = 0

for i in range(1300):

    if mp[clusterLabels[i]] == data['Class'][i]:

        score = score +1

print(score/1300)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)