%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import time

from datetime import datetime

from collections import Counter

from wordcloud import WordCloud
df = pd.read_csv('../input/attacks.csv', encoding = "ISO-8859-1")
wordcloud = WordCloud(background_color="white")

wordcloud.generate(' '.join(df.Activity.dropna().values))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()