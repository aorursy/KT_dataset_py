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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
%matplotlib inline

filepath='/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv'
df_o = pd.read_csv(filepath,encoding="unicode_escape")
df=df_o.rename(columns={"top genre":"top_genre"})


df.shape

#note: no missing values in any rows to deal with
#function which aggregates all the different hip hop versions together.
# I also aggregate all the factors apart from the the top ten genres into "other",which I then ignore

def applyFunc(s):
    '''
    Input s - string - a selected genre which is to be replaced
    Output - desc - string - new name for this genre
    
    The idea was to aggregate certain genres into one- Hip hop because there were several types, 
    and others because of the bar chart display of minor genres.
    '''
    if s == 'dance pop':
        return 'dance pop'
    elif s == 'pop':
        return 'pop'
    elif s == 'canadian pop':
        return 'canadian pop'
    elif s == 'boy band':
        return 'boy band'
    elif s == 'barbadian pop':
        return 'barbadian pop'
    elif s == 'electropop':
        return 'electropop'
    elif s == 'british soul':
        return 'british soul'
    elif s == 'big room':
        return 'big room'
    elif s == 'canadian contemporary r&b':
        return 'canadian contemporary r&b'
    elif s == 'neo mellow':
        return 'neo mellow'
    elif s == 'art pop':
        return 'art pop'
    elif s == 'hip pop':
        return 'hip pop'
    elif s == 'canadian hip hop':
        return 'hip hop'
    elif s == 'atl hip hop':
        return 'hip hop'
    elif s == 'detroit hip hop':
        return 'hip hop'
    elif s == 'hip hop':
        return 'hip hop'
    elif s == 'australian hip hop':
        return 'hip hop'
    elif s == 'chicago rap':
        return 'hip hop'
    elif s == 'electronic trap':
        return 'hip hop'
    return 'others'

df['agg_genre'] = df['top_genre'].apply(applyFunc)

df_no_others = df.drop(df[df['agg_genre'] == 'others'].index, inplace= True)

df.head()


df.shape
df.hist()
# number of each genre after aggregation of genres
df.groupby('agg_genre').top_genre.count().sort_values(ascending=False).head(10)

#dropping the 'others" as it would spoil the picture in the bar chart. Others are just the other genre aside from the main genres, which we are not interested in
df.drop(df[df['agg_genre'] == 'others'].index, inplace= True)
#display of main genres as abar chart
Barchart_Data = df.groupby('agg_genre').top_genre.count().sort_values(ascending=False).head(10)


Barchart_Data.plot.bar()
#creating the hip hop genre dataframe
hip_hop=df[df['agg_genre']=='hip hop']
#who has to most tracks in the hiphop genre in the top ten
hip_hop['artist'].value_counts().reset_index()
#an attempt to select only those with two tracks
hip_hop.groupby('artist').count()>1
#group by artist and year
hip_hop.groupby(['artist','year']).sum()
#group by artist,year and genre
hip_hop.groupby(['artist','year','top_genre']).sum()
#hip hop genre means grouped in years
hip_hop.groupby(['year']).mean()
#general hip hop means for all attributes
hip_hop.mean()
#correlation of all data
sns.heatmap(df.corr(), annot=True, robust= True)
#correlation of attributes which have relevant effects
interest_factors=df[['pop','nrgy','dnce','val','dB']]
sns.heatmap(interest_factors.corr(), annot=True, robust= True)
#further analysis of factors
sns.pairplot(interest_factors,height=1.5)
#better display of effective factors 
plt.figure(figsize=(8,8))
sns.heatmap(interest_factors.corr(), annot=True, robust= True,cmap='coolwarm')
#not used for blog: most popular artist in the last decade
df['artist'].value_counts().reset_index()
#not used in blog: correlation display of hip hop genre
sns.heatmap(hip_hop.corr(), annot=True, robust= True)

hip_hop_int=hip_hop[['year','pop','nrgy','dnce','live','val','dB']]
hip_hop_int.head()
sns.heatmap(hip_hop_int.corr(), annot=True, robust= True)
sns.pairplot(hip_hop_int,height=1.5)
hip_hop.mean()
sns.heatmap(hip_hop.corr(), annot=True, robust= True)
#most recent and popular hip hop artist
DJ_Snake=hip_hop[hip_hop['artist']=='DJ Snake']
DJ_Snake.head()
#not used in blog: correlation display of interesting attributes of hip hop genre
hiphop_interest_factors=df[['year','pop','spch','acous','nrgy','dnce','val','dB']]
sns.heatmap(hiphop_interest_factors.corr(), annot=True, robust= True)
