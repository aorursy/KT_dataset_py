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
members = pd.read_csv('../input/members.csv')

members.head()
members.shape
import seaborn as sns

import matplotlib.pyplot as plt



f,axarray = plt.subplots(1,1,figsize=(15,10))

agehist = members.groupby(['bd'],as_index=False).count()

sns.barplot(x=agehist['bd'],y=agehist['gender'])
f,axarray = plt.subplots(1,1,figsize=(15,10))

cityhist = members.groupby(['city'],as_index=False).count()

sns.barplot(x=cityhist['city'],y=cityhist['gender'])
songs = pd.read_csv('../input/songs.csv')

songs.head()
print ("There are:",len(songs),"songs;",len(songs['composer'].unique()),

       "composers for",len(songs['genre_ids'].unique()),"genres in",

      len(songs['language'].unique()),"languages")
train = pd.read_csv('../input/train.csv')

train.head()
df = train.merge(members,how='inner',on='msno')
df = df.merge(songs,how='inner',on='song_id')

df.head()
df.shape
cities = df['city'].unique()

ca = []

for c in cities:

    ages = []

    tmp = df[df['city']==c].groupby(['bd'],as_index=False).count()

    for i in range(60):

        if i in tmp['bd'].values:

            if i ==0:

                ages.append(0)

            else:

                ages.append(tmp[tmp['bd']==i].values[0][1])

        else:

            ages.append(0)

    ca.append(ages)

cadf = pd.DataFrame(ca)
f,axarray = plt.subplots(1,1,figsize=(13,8))

sns.heatmap(cadf)
fdf = df[np.abs(df['bd']-df['bd'].mean())<=(3*df['bd'].std())]
cities = fdf['city'].unique()

ca = []

for c in cities:

    ages = []

    tmp = fdf[fdf['city']==c]['bd'].values

    ages.append(tmp)

    ca.append(ages)

cadf = pd.DataFrame(ca)
f,axarray = plt.subplots(21,1,figsize=(20,38),sharex=True)

plt.xlim(10,60)

for i in range(21):

    axarray[i].set_title('Members Ages in City '+str(i))

    sns.distplot(ca[i], hist=False, color="purple", kde_kws={"shade": True},ax=axarray[i])
df['registration_init_time'] = pd.to_datetime(df['registration_init_time'],format="%Y%m%d")

df['expiration_date'] = pd.to_datetime(df['expiration_date'],format="%Y%m%d")
df.head()
days = df.expiration_date - df.registration_init_time

days = [d.days for d in days]

df['days']=days
np.max(days)
df.head()
fdf = df[np.abs(df['days']-df['days'].mean())<=(3*df['days'].std())]
dayshist = df.groupby(['days'],as_index=False).count()

dayshist = dayshist.drop(0,axis=0)
sns.distplot(dayshist['days'], hist=True, color="g", kde_kws={"shade": True})
cities = fdf['city'].unique()

cduration = []

for c in cities:

    duration = []

    tmp = fdf[fdf['city']==c]['days']

    cduration.append(tmp)
f,axarray = plt.subplots(21,1,figsize=(20,38),sharex=True)

for i in range(21):

    axarray[i].set_title('Subscription Durations in City '+str(i))

    sns.distplot(cduration[i], hist=False, color="g", kde_kws={"shade": True},ax=axarray[i])
malec = len(df[df['gender']=='male'])

femalec = len(df[df['gender']=='female'])
f,axarray = plt.subplots(1,1,figsize=(8,5))

sns.barplot(x=['male','female'],y=[malec,femalec])
len(df[pd.isnull(df['gender'])])/len(df)
len(df['genre_ids'].unique())
ghist = df.groupby(['genre_ids'],as_index=False).count()
f,axa = plt.subplots(1,1, figsize=(12,18))

tghist = ghist[ghist['msno']>1000]

sns.barplot(y=tghist['genre_ids'],x=tghist['msno'],orient='h')
test = pd.read_csv('../input/test.csv')
test.head()
test['msno'][0] in df['msno'].values
tdf = test.merge(members,how='inner',on='msno')

tdf.head()
df.head()
tmp= df.groupby(['msno'],as_index=False).count()['song_id']

tmp.describe()
f,axa = plt.subplots(1,1,figsize=(15,8))

sns.distplot(tmp.values)
songs_index = [songs[songs['song_id']==df['song_id'][k]].index[0] for k in range(len(df))]
df['song_index'] = songs_index
