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


train = pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

members=pd.read_csv("../input/members.csv")

songs=pd.read_csv("../input/songs.csv")

print('train',train.shape)

print('test',test.shape)

print('members',members.shape)

print('songs',songs.shape)
print('train')

print(train.head())



print('members')

print(members.head())

print('songs')

print(songs.head())

print(train.target.value_counts()*100/train.target.value_counts().sum())

print('How many unique songs ',len(train.song_id.unique()))
%matplotlib inline

from matplotlib import rcParams

rcParams['figure.figsize'] = (20, 6)

repeats=train[train.target==1]

song_repeats=repeats.groupby('song_id',as_index=False).msno.count()

song_repeats.columns=['song_id','count']



song_repeats=pd.DataFrame(song_repeats).merge(songs,left_on='song_id',right_on='song_id')

print('median length of songs repeated',song_repeats.sort_values(by='count',ascending=False)[:2000].song_length.median())

print('median length of songs repeated',songs.song_length.median())
print("Top 20 songs repeated")

repeats.song_id.value_counts()[:20]
import matplotlib.pyplot as plt    

from wordcloud import WordCloud



def displaywc(txt,title):

    txt=""

    for i in g:

        txt+=str(i)

    wordcloud = WordCloud(background_color='white').generate(txt)



    plt.figure()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title(title)

    plt.show()
g=song_repeats.sort_values(by='count',ascending=False)[:200].artist_name.tolist()

#TODO Investigate how to display chinese

txt=""

for i in g:

    txt+=str(i)

displaywc(txt,'Most common artists that people listen to ')
df=pd.DataFrame(song_repeats.sort_values(by='count',ascending=False))

df.language.value_counts().plot(kind='bar')

plt.title('Language of most common songs')

plt.ylabel('Count')

plt.xlabel('Language')
df.genre_ids.value_counts()[:35].plot(kind='bar')

plt.title('genre of most listened to songs')

plt.xlabel('genre')

plt.ylabel('count')
g=song_repeats.sort_values(by='count',ascending=False)[:200].composer.tolist()

#TODO Investigate how to display chinese

txt=""

for i in g:

    txt+=str(i)

displaywc(txt,'Most common composers that people listen to ')
g=song_repeats.sort_values(by='count',ascending=False)[:200].lyricist.tolist()

#TODO Investigate how to display chinese

txt=""

for i in g:

    txt+=str(i)

displaywc(txt,'Most common lyricist that people listen to ')
print('Users that listen to the same song again and again')

repeats.msno.value_counts()[:10]
repeats.source_system_tab.value_counts().plot(kind='bar')

#the name of the tab where the event was triggered. 

#System tabs are used to categorize KKBOX mobile apps functions.

#For example, tab my library contains functions to manipulate the local storage, 

#and tab search contains functions relating to search.

plt.title('Song repeats by system tabs')

plt.xlabel('system tabls')

plt.ylabel('count')
#source_screen_name: name of the layout a user sees. 

repeats.source_screen_name.value_counts().plot(kind='bar')

plt.title('Repeat songs by screen name')

plt.xlabel('screen names')

plt.ylabel('count')


#source_type: an entry point a user first plays music on mobile apps. 

#An entry point could be album, online-playlist, song .. etc. 

repeats.source_type.value_counts().plot(kind='bar')

plt.title('Where does the user play music from')

plt.xlabel('source type')

plt.ylabel('count')

user_repeats=repeats.merge(members,left_on='msno',right_on='msno',how='left')

print("users who repeat")

user_repeats.msno.value_counts()[:20]
user_repeats.gender.value_counts(dropna=False).plot(kind='bar')

plt.title('Users listening to same song by gender')

plt.xlabel('gender')

plt.ylabel("count")
user_repeats.city.value_counts().plot(kind='bar')

plt.title('repeat users by city')

plt.xlabel('city')

plt.ylabel('count')
user_repeats.registered_via.value_counts().plot(kind='bar')

plt.title('repeat users by registration mechanism')

plt.xlabel('registration mechanism')

plt.ylabel('count')