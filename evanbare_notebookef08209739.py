import pandas as pd

import numpy

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
debate_df = pd.read_csv('../input/debate.csv', encoding='cp1252')

debate_df.head()
clinton_df = debate_df[debate_df['Speaker'] == 'Clinton']

trump_df = debate_df[debate_df['Speaker'] == 'Trump']
trump_russia = trump_df[trump_df['Text'].str.contains('Russia')]

trump_russia_words = ' '.join(trump_russia['Text'])

russia_stopwords = STOPWORDS.union({'Russia', 'Russian', 'Russians'})

trump_russia_cloud = WordCloud(stopwords=russia_stopwords, background_color='black', width=1800, height=1400).generate(trump_russia_words)

plt.imshow(trump_russia_cloud)

plt.axis('off')

plt.savefig('./donald_russia.png')

plt.show()

clinton_russia = clinton_df[clinton_df['Text'].str.contains('Russia')]

clinton_russia_words = ' '.join(clinton_russia['Text'])

clinton_russia_cloud = WordCloud(stopwords=russia_stopwords, background_color='black', width=1800, height=1400).generate(clinton_russia_words)

plt.imshow(clinton_russia_cloud)

plt.axis('off')

plt.savefig('./clinton_russia.png')

plt.show()