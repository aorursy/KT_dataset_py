# import all necessary libraries

import numpy as np

import pandas as pd

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



import re
# Read data from restaurantreviews.csv

df = pd.read_csv("/kaggle/input/mydata/restaurantreviews.csv", sep="\t", names=['Review','Liked'], encoding="latin-1")

df
# init the corpus

corpus = []

wordnet = WordNetLemmatizer()

for i in range(1,len(df)):

    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])

    review = review.lower()

    review = review.split()

    review = [wordnet.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)

#corpus
# Extract the feature (BOW) 

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x = cv.fit_transform(corpus)

#print(x)
# Extract the feature (Tf-IDF)

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(max_features=2500)

x = tf.fit_transform(corpus)

print(x)