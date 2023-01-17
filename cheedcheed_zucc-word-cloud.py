import pandas as pd

df = pd.read_csv('../input/mark.csv')
grp = df.groupby('Person')

speakers = grp['Text'].apply(' '.join).reset_index()
speakers = speakers[speakers['Person'].str.isupper()]  # CSV malformed
speakers['Text'] = speakers['Text'].str.replace('\r\n', ' ')
speakers['String Length'] = speakers['Text'].apply(len)

# The speakers dataframe contains all dialog spoken, grouped by each speaker
speakers
import wordcloud
import matplotlib.pyplot as plt

def word_cloud(corpus, title):
    words = corpus.lower()
    cloud = wordcloud.WordCloud(background_color='black',
                                max_font_size=200,
                                width=1600,
                                height=800,
                                max_words=300,
                                relative_scaling=.5).generate(words)
    plt.figure(figsize=(16,10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(cloud)
mask = df['Person'].str.contains('ZUCK')
senators = df[~mask]['Text'].str.replace('\r\n', ' ').values

word_cloud(' '.join(senators), 'All Senators')
for _, (title, text, _) in speakers.iterrows():
    word_cloud(text, title)