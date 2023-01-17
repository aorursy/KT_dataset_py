import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import stats

from nltk.stem.wordnet import WordNetLemmatizer



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
play = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
playreview = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")
play.head()
play.shape
play.isnull().sum()
play.Rating.value_counts()
sns.boxplot(play.Rating)
playreview.dropna(inplace=True)
playreview.isnull().sum()
playreview['Translated_Review'] = playreview.Translated_Review.str.replace("[^a-zA-Z#]", " ")
playreview.head()
playreview.head()
playreview['Translated_Review'] = playreview['Translated_Review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
playreview.head()
playreview = playreview.reset_index().drop('index',axis=1)
tokens = playreview['Translated_Review'].apply(lambda x: x.split())

tokens.head()
wnl = WordNetLemmatizer()
tokens = tokens.apply(lambda x: [wnl.lemmatize(i) for i in x])
tokens
for i in range(len(tokens)):

    tokens[i] = ' '.join(tokens[i])



playreview['Translated_Review'] = tokens
playreview['Translated_Review'].apply(lambda x: '')
all_words = ' '.join([text for text in playreview['Translated_Review']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
positive_words = ' '.join([text for text in playreview['Translated_Review'][playreview['Sentiment']=='Positive']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
negative_words = ' '.join([text for text in playreview['Translated_Review'][playreview['Sentiment']=='Negative']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
playreview.Sentiment.value_counts()
sns.countplot(playreview.Sentiment)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
x = playreview.Translated_Review

y = playreview.Sentiment
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.3)
# Logistic

review_lr = Pipeline([('tfidf', TfidfVectorizer()),

                     ('review', LogisticRegression()),

])



# Naïve Bayes:

review_nb = Pipeline([('tfidf', TfidfVectorizer()),

                     ('review', MultinomialNB()),

])



# Linear SVC:

review_svc = Pipeline([('tfidf', TfidfVectorizer()),

                     ('review', LinearSVC()),

])
def model(obj,name):

    ypred = obj.fit(xtrain,ytrain).predict(xtest)

    return print(name,"\n\n",

                "Accuracy Score:- ",accuracy_score(ytest,ypred),"\n\n Confusion Matrix:- \n",confusion_matrix(ytest,ypred),

                "\n\n Classification Report:- \n",classification_report(ytest,ypred))
model(review_lr,"Logistic Regression")
model(review_nb,"Naive Bayes")
model(review_svc,"Support Vector Classifier")