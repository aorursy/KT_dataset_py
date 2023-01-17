import pandas as pd

df = pd.read_csv("../input/taylor_swift_lyrics.csv", encoding = 'latin-1')

df.head()
pd.options.display.max_colwidth = 5000
songs = df.groupby('track_title').agg({'lyric':lambda x: ' '.join(x),

                                       'year': 'mean'}).reset_index()

songs.head()
len(songs)
# Bag of Words

# tf-idf = term frequency-inverse document frequency

# tf x (penalty for common words)
import nltk

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = stop_words, min_df = 0.1)

stop_words.extend(['back','said','come','things','get','oh','one',\

'yeah','place','would','like','know','stay','go','let','cause',\

'could','wanna','would','gonna'])
tfidf = vectorizer.fit_transform(songs['lyric'])
# top modeling: Method1. LDA Method2. NMF 
from sklearn.decomposition import NMF
# 6 topics

nmf = NMF(n_components = 6)
topic_values = nmf.fit_transform(tfidf)
for topic_num, topic in enumerate(nmf.components_):

    message = 'Topc #{}:'.format(topic_num + 1)

    # top 10 words in each topic

    message += ' '.join([vectorizer.get_feature_names()[i] 

                         for i in topic.argsort()[:-11 :-1]])

    

    print(message)
topic_labels = ['love/beauty', 'growing up', 'home',\

                'bad/remorse', 'hope/better',  'party/dance']
df_topics = pd.DataFrame(topic_values, columns = topic_labels)

df_topics.head()
songs = songs.join(df_topics)

songs.head()
for topic in topic_labels:

    songs.loc[songs[topic] >= 0.1, topic] = 1

    songs.loc[songs[topic] < 0.1, topic] = 0
songs.head()
year_topics = songs.groupby('year').sum().reset_index()

year_topics
import matplotlib

import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 30, 'lines.linewidth': 8 })



plt.figure(figsize=(30,15))

plt.grid(True)

for topic in topic_labels:

    plt.plot(year_topics['year'], year_topics[topic], label = topic, linewidth=7.0)

plt.legend()  

plt.xlabel('year')

plt.ylabel('# of songs per topic')

plt.title('Topic modeling of Taylor Swift\'s lyrics')

plt.show()



# Year 2010 is around the time TS moved out of her family home into her first apartment: 

# home and growing up makes a drastic increase.

# The topic of growing up decreases since 2012, a sign that TS is more confident as a female artist.