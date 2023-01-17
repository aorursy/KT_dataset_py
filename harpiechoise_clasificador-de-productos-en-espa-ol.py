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
df = pd.read_csv('/kaggle/input/flipkart-products/flipkart_com-ecommerce_sample.csv')
df.head()
df.info()
df = df[['description', 'product_category_tree']]
# Challenging

df.head()
df['product_category_tree'] = df.product_category_tree.apply(lambda x: x.replace('[', '').replace(']', '').replace('"',''))
x = df['product_category_tree'].apply(lambda x: x.split('>>'))
x = x.apply(lambda x: x[0])
df['category'] = x
#df = df.drop('product_category_tree', axis=1)

df.head()
counts = df.category.value_counts()
most_frequent_category_names = counts[:10].index.tolist()

most_frequent_category_counts = counts[:10].values.tolist()

import plotly.graph_objects as go

fig = go.Figure(go.Bar(x=most_frequent_category_names, y=most_frequent_category_counts))

fig.show()
import spacy

import nltk

nlp = spacy.load("en_core_web_sm")

def count_words(matrix):

    words = []

    words_freq = []

    words_vals = []

    for desc in matrix:

        doc = nlp(desc)

        for token in doc:

            if not token.is_stop:

                words.append(doc)

    f = nltk.FreqDist(words)

    for x, v in f.most_common(10):

        words_freq.append(x)

        words_vals.append(v)

    return words, words_vals, words_freq
words, words_vals, words_freq = count_words(df.category.values)
from wordcloud import WordCloud
im = WordCloud(width=640, height=800).generate(",".join([i.text for i in words]))
import matplotlib.pyplot as plt

f, a = plt.subplots(figsize=(8, 12))

a.imshow(im, interpolation='nearest', aspect='auto')
# Modelando

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.svm import SVC

cv = CountVectorizer(stop_words='english')

tf = TfidfTransformer()
df.dropna(inplace=True)

X = cv.fit_transform(df.description.values)
X = tf.fit_transform(X)

y = df.category.values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV

import numpy as np

param_grid = {'alpha':np.linspace(0, 1, 10), 'fit_prior':[True, False]}

grid = GridSearchCV(MultinomialNB(), param_grid=param_grid, cv=4)
grid.fit(X_train, y_train)
model = grid.best_estimator_

model.score(X_test, y_test)
from textblob import TextBlob



def clasificador_español(model, cv, tf, producto='Zapatillas deportivas'):

    blob = TextBlob(producto)

    ingles = blob.translate(to='en')

    vect = cv.transform([ingles.raw])

    idf = tf.transform(vect)

    ingles = model.predict(idf)[0]

    blob = TextBlob(ingles)

    

    return blob.translate(to='es').raw
clasificador_español(model, cv, tf)
clasificador_español(model, cv, tf, 'Desatornillador Electrico')
clasificador_español(model, cv, tf, 'Laptop Acer')
clasificador_español(model, cv, tf,'Collar Swarovski Elements Corazón 14mm Cadena Plata')