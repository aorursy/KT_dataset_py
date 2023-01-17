import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
all_desc = reviews.description.str.split(' ')
rw = [ word for desc in all_desc for word in desc]
rw4 = [ w for w in rw if len(w) > 4]
rwSerie = pd.Series(rw4)
rwCounts = rwSerie.value_counts()
top20 = rwCounts[:20]
words = ''
for w in top20.index:
    for i in range(top20.loc[w] // 5000):
        words = words + w + ' '

plt.figure(figsize=(12,8))
wc = WordCloud(background_color='gray', max_font_size=200,
                            width=600,
                            height=400,
                            max_words=20,
                            relative_scaling=.2).generate(words)
plt.imshow(wc)
plt.title("Words at Wine Reviews", fontsize=24)
plt.axis("off");
