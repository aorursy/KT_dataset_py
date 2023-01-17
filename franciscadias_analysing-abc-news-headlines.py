import numpy as np 

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/abcnews-date-text.csv")

df.head()
df.dtypes
df["year"] = df["publish_date"].astype(str).str[:4].astype(np.int64)

df.head()
df["month"] = df["publish_date"].astype(str).str[4:6].astype(np.int64)

df.head()
df.year.unique()
df.month.unique()
df["word_count"] = df["headline_text"].str.len()

df.head()
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns



with sns.color_palette("GnBu_d", 10):

    ax= sns.countplot(y="year",data=df)

    ax.set(xlabel='Number of Articles', ylabel='Year')

plt.title("Number of Articles per Year")
df["headline_text"][0]
from nltk.corpus import stopwords

stopwords.words("english")
# Remove stop words from "words"



words = [w for w in words if not w in stopwords.words("english")]

words
df.shape
from sklearn.feature_extraction.text import CountVectorizer





vectorizer = CountVectorizer(analyzer = "word",   

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = "english",   

                             max_features = 30)



news_array = vectorizer.fit_transform(df["headline_text"])



# Numpy arrays are easy to work with, so convert the result to an array

news_array = news_array.toarray()



# Lets take a look at the words in the vocabulary and  print the counts of each word in the vocabulary:

vocab = vectorizer.get_feature_names()



# Sum up the counts of each vocabulary word

dist = np.sum(news_array, axis=0)



# For each, print the vocabulary word and the number of times it appears in the training set

for tag, count in zip(vocab, dist):

    print (count, tag)