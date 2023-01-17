# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



file = '/kaggle/input/beers-breweries-and-beer-reviews/reviews.csv'
# Visualization

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

import seaborn as sns



# sklearn for feature extraction & modeling

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

# Iteratively read files

import glob

import os



# For displaying images in ipython

import seaborn as sns

sns.set(color_codes = True)

%matplotlib inline
df = pd.read_csv(file, engine = "c", nrows = 3000)

#df = pd.read_csv(file, sep=",", dtype={'reviews.rating': int})

df = df[df["text"].str.strip().str.contains(' ') == True]



train, test = train_test_split(df, test_size = 0.1, random_state=42)
from nltk.corpus import wordnet



def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN

    

import string

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer



# função pra limpar os dados

def clean_text(text):

    # lower text

    text = text.lower()

    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    stop = stopwords.words('english')

    text = [x for x in text if x not in stop]

    # remove empty tokens

    text = [t for t in text if len(t) > 0]

    # pos tag text

    pos_tags = pos_tag(text)

    # lemmatize text

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]



    text = [t for t in text if len(t) > 1]



    text = " ".join(text)

    return(text)
df["text"] = df["text"].apply(lambda x: clean_text(x))
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



heading_1 = train[train["score"]==1]["text"]

collapsed_heading_1 = heading_1.str.cat(sep=' ')



heading_2 = train[train["score"]==2]["text"]

collapsed_heading_2 = heading_2.str.cat(sep=' ')



heading_3 = train[train["score"]==3]["text"]

collapsed_heading_3 = heading_3.str.cat(sep=' ')



heading_4 = train[train["score"]==4]["text"]

collapsed_heading_4 = heading_4.str.cat(sep=' ')



heading_5 = train[train["score"]==5]["text"]

collapsed_heading_5 = heading_5.str.cat(sep=' ')
stopwords = set(STOPWORDS)

stopwords.update(["beer"])



print("Nota 1")

wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=40).generate(collapsed_heading_1)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



print("\nNota 2")

wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=40).generate(collapsed_heading_2)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



print("\nNota 3")

wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=40).generate(collapsed_heading_3)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



print("\nNota 4")



wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=40).generate(collapsed_heading_4)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()

     

print("\nNota 5")

wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=40).generate(collapsed_heading_5)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from textblob import TextBlob

from textblob.classifiers import NaiveBayesClassifier



model = NaiveBayesClassifier(train[['text','score']].values)



right = 0



for row in test.values:

    value = model.classify(row[3])

    

    if (value == round(row[9])):

        right += 1



print("reviews: " +  str(len(test)))

print("certo: " + str(right))