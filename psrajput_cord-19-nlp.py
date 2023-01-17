# from sklearn.feature_extraction.text import CountVectorizer



# Data Processing

import numpy as np, pandas as pd, os, json, glob, re, nltk



# NLP

# nltk.download('stopwords')

# !pip install scispacy

# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz

# import en_core_sci_sm as biod

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# import scispacy

import spacy

from spacy import displacy



# Visualisation



%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt



# Everything Else



from tqdm.notebook import tqdm

# from PIL import Image
covid_19 = pd.read_csv("../input/covid-19/covid_19.csv")

covid_19.head()
corpus = []



stopwords = nltk.corpus.stopwords.words('english')

# stopwords.extend(custom_stopwords)



# Data Cleaning: Removing symbols, lowercasing, spliiting, reducing similar words

for i in tqdm(range(len(covid_19["Body_text"]))):

    body_text = re.sub('[^a-zA-Z]', ' ', covid_19['Body_text'][i])

    body_text = body_text.lower()

    body_text = body_text.split()

    body_text = [word for word in body_text if not word in set(stopwords)]

    body_text = ' '.join(body_text)

    corpus.append(body_text)



len(corpus)
# Merging corpus to a string



text = ""



for i in tqdm(corpus):

    text += i
# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=100, background_color="white").generate(text)



# Display the generated image:

plt.figure(figsize=(14,8))

plt.imshow(wordcloud, interpolation='nearest')

plt.axis("off")

plt.show()

# plt.savefig("wc1.png", dpi=900)