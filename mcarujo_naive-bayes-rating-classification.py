! pip install nb_black -q
%load_ext nb_black
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="ticks", color_codes=True)
data = pd.read_csv("../input/yelp-reviews-dataset/yelp.csv")

data["length"] = data.text.apply(len)

data.drop(["business_id", "review_id", "user_id", "type"], axis=1, inplace=True)

data.head()
data.describe()
data.info()
sns.countplot(y="stars", data=data)

plt.title("Counting by stars")
g = sns.FacetGrid(data=data, col="stars", col_wrap=5)

g.fig.suptitle("Dists by stars")

g.map(plt.hist, "length", bins=10)

filter_bad_good = data.stars.isin([1, 5])

data_bad_good = data[filter_bad_good]
sns.countplot(data_bad_good["stars"])

plt.title("Count by stars again...")
import string

from nltk.corpus import stopwords

import nltk



stemmer = nltk.RSLPStemmer()

stopwords = list(stopwords.words("english"))

punctuation = [word for word in string.punctuation]

punctuation += ['...', '  ', '\n']







def remove_punctuation(serie, stopwords):

    aux = list()

    for el in serie:

        for word in stopwords:

            el = el.replace(word,' ')

        aux.append(el)

    return aux



def remove_stopwords(serie, stopwords):

    tokenizer = nltk.WordPunctTokenizer()



    result_serie= list()

    for row in serie:

        aux = list()

        text_row = tokenizer.tokenize(row.lower())

        for word in text_row:

            if word not in stopwords: # stopwords

                aux.append(word)

        result_serie.append(' '.join(aux))

    return result_serie

data_bad_good.text = data_bad_good.text.str.lower()

data_bad_good.text = remove_stopwords(data_bad_good.text, punctuation)

data_bad_good.text = remove_stopwords(data_bad_good.text, stopwords)
from sklearn.feature_extraction.text import CountVectorizer



vectorize = CountVectorizer()



X = vectorize.fit_transform(data_bad_good.text)

Y = data_bad_good.stars.map({5: 1, 1: 0}).values

print("How many features (bag of words): ", len(vectorize.get_feature_names()))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(

    X, Y, test_size=0.2, random_state=42

)



model = MultinomialNB()



model.fit(X_train, y_train)





# Let's see the results

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100

print("The accuracy was %.2f%%" % accuracy)
from sklearn.metrics import confusion_matrix



m_c = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5, 4))

sns.heatmap(m_c, annot=True, cmap="Reds", fmt="d").set(xlabel="Predict", ylabel="Real")