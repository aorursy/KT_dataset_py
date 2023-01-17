import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud
df = pd.read_json('../input/scraps-from-the-loft.jl', lines=True)
df.head()
len(df)
tokens = []

tokens.extend([item for sublist in df.categories for item in sublist])

tokens.extend([item for sublist in df.tags for item in sublist])

tokens = pd.DataFrame(tokens, columns=['tokens'])

tokens.head()
tokens['count'] = 1

tokens.head()
count = tokens.groupby('tokens').count()

count = count.sort_values('count', ascending=False)

count.head()
count.head(50).plot.bar(figsize=(15,10))

plt.xticks(rotation=90);

plt.xlabel("Token");

plt.ylabel("Absolute Frequency");
wordcloud = WordCloud(width=800, height=600, max_font_size=80, background_color="white", colormap='Set1')

wordcloud = wordcloud.generate_from_frequencies(count.head(100).to_dict()['count'])

plt.figure(figsize=[14,21])

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()