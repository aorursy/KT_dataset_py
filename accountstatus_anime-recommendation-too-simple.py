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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from scipy.spatial import distance

data=pd.read_csv('../input/anime-recommendations-database/anime.csv')
data=data[data['genre']!='Hentai']
data.head()
#Let's make our own algorithm for the dataset
# So we are gonna make an anime recommendation system on our own :)
# we want to know all the genres of the animes
genre=data['genre'].values
data.info()
lis=[]

for i in genre:

    i=str(i)

    for p in i.split(','):

        if p not in lis:

            lis.append(p)
len(lis)
dic={}

t=[]

for i in genre:

    for j in lis:

        dic[j]=0

    i=str(i)

    for p in i.split(','):

        dic[p]+=1

    t.append(list(dic.values()))

        
len(t[0])
X=np.array(t)
# Dynamic Recommendation System made by me
# input 1 is the anime name

# input 2 is the number of search result you want

s=input('Enter the anime you like and we will find 3 like those :')

num=int(input('Please enter the number of recommendations you want:'))

name=data['name'].values

for i in range(len(name)):

    name[i]=name[i].lower()

s=s.lower()

h=-1

for i in range(len(name)):

    if s in name[i]:

        h=i

        break

imp=[]

if h==-1:

    print('Sorry No match found :(')

else:

    for i in range(len(t)):

        if i==h:

            continue

        else:

            if len(imp)<num:

                imp.append([distance.euclidean(t[i], t[h]),t[i],i])

            else:

                imp.sort()

                if imp[num-1][0]>distance.euclidean(t[i],t[h]):

                    del imp[num-1]

                    imp.append([distance.euclidean(t[i],t[h]),t[i],i])

    name=data['name'].values

    print('The anime recommended for you are :')

    count=0

    for i in imp:

        count+=1

        print(count,'. ',name[i[2]])

                