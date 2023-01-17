import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
lyrics = pd.read_csv('../input/songdata.csv')

lyrics.head(2)
lyrics = lyrics.drop(['link'], axis=1)                   # remove useless column

lyrics.head(2)
arts_list = lyrics['artist'].unique()

print(len(arts_list))                           # prints the number of artists 
my_arts=['Aerosmith','Bruce Springsteen','Depeche Mode','Jimi Hendrix','Leonard Cohen',

         'Metallica','Paul Simon','Prince','Queen','U2']        # my sample
text_lenght = [0]*10                                 # creating the subtable

my_lyrics = lyrics[lyrics.artist =='Aerosmith']

text_lenght[0] = len(my_lyrics)



for i in range(1,10):

    text_lenght[i]= len(lyrics[lyrics['artist']==my_arts[i]]['text'])

    my_lyrics = my_lyrics.append(lyrics[lyrics.artist ==my_arts[i]])

    i+=1



print(text_lenght)                               # Number of songs per chosen artist  

print(len(my_lyrics))                            # Combined number of songs
sns.set_context("notebook", font_scale=1.0)

sns.set_palette('cubehelix',4)         

plt.figure(figsize=(13,3))

plt.title('Number of Songs by Artist')

g = sns.countplot(my_lyrics['artist'])

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=10)                                        

my_lyrics.head()
warray = [['']]*10                     # Picking lists of most favorite words by artist

fav_words = [['']]*10

word_cnt = [0]*10

tfidf = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, stop_words='english')

i=0                                                    

for artist, songs in my_lyrics.groupby('artist'):

    my_texts = lyrics[lyrics['artist']==my_arts[i]]['text']               

    tfidf.fit_transform(my_texts)

    cnt = np.sum(tfidf.transform(songs['text']).toarray(), axis=0)

    warray[i] = tfidf.get_feature_names()

    word_cnt[i] = len(warray[i])

    sort_freq = np.argsort(cnt.flatten())[::-1]   

    fav_words[i] = [tfidf.get_feature_names()[idx] for idx in sort_freq.tolist()[:10]]

    print(my_arts[i], "has the most favorite words :",fav_words[i],

         'and the number of unique words in all songs is:', word_cnt[i])

    i+=1 
comb_texts = ['']*10            # concatenated texts of all songs by the artist

comb_length = [0]*10

i=0

j=0

mn=0

while (i!=len(my_lyrics)):

    comb_texts[j] += (my_lyrics.iloc[i]['text'])

    if (i == text_lenght[j]+ mn ):

        mn+=text_lenght[j]

        j+=1

    i+=1 

for l in range (0,10):

    comb_length[l] = len(comb_texts[l])    

print(comb_length)                         # lenghts of combined strings per artist 
artists_stats = pd.DataFrame()

artists_stats['artist'] = my_arts

artists_stats['songs'] = text_lenght

artists_stats['words'] = word_cnt

artists_stats['all_length'] = comb_length

artists_stats
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('Number of Words by Artist')

g = sns.barplot( x='artist', y='words', data=artists_stats )

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=10)
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('Lenght of combined songs by Artist')

g = sns.barplot( x='artist', y='all_length', data=artists_stats )

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=10)
topics = ['life','death','love','hate','dream','hell','light','dark','heart','kill']

topic_cnt = np.zeros((10,10), int)

i = 0                                    # setting short vocabulary positive/negative pairs

j = 0                                 

for text in comb_texts:

    for topic in topics:

        topic_cnt[i][j] = text.count(topic)

        j+=1

    j=0    

    i+=1

    

def score_count(a,b,c,d,e,f,g,h,k,l):

    return (100*(a+b+c+d+e)/(f+g+h+k+l))            # for better visualization to integer 
freq_df = pd.DataFrame(topic_cnt)

freq_df.rename(columns=lambda x: topics[int(x)], inplace=True)

freq_df['artist'] = my_arts

col_to_keep=['artist','life','death','love','hate','dream','hell','light','dark',

             'heart','kill']

freq_df = freq_df[col_to_keep]

freq_df['mood'] =score_count(freq_df.life,freq_df.love,freq_df.dream,freq_df.light,

                             freq_df.heart,freq_df.death,freq_df.hate,freq_df.hell,

                             freq_df.dark,freq_df.kill)

freq_df                      # table of word frequencies by artist
sns.set_context("notebook",font_scale=1.)

plt.figure(figsize=(13,4))

plt.title('Positive mood score by Artist')

g = sns.barplot( x='artist', y='mood', data=freq_df )

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=10)