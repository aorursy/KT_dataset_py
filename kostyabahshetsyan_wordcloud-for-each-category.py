import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image

import requests

tech = pd.read_csv("../input/techcrunch_posts.csv")
tech['category']=tech['category'].apply(lambda x:['No category'] if str(x)=='NaN' else str(x))
tech.info()
stopwords = set(STOPWORDS)



# generate word cloud and show it

for x in tech.category.unique():

	wc = WordCloud(background_color="white", max_words=2000, stopwords=stopwords,

                   max_font_size=40, random_state=42)

	wc.generate(tech.content[(tech.content.notnull()) & (tech.category == x)].to_string())

	plt.imshow(wc)

	plt.title(x)

	plt.axis("off")

	plt.show()

tc_coloring = np.array(im.data)