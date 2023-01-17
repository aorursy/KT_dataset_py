import re
import numpy as np 
import pandas as pd 
ag_train = pd.read_csv('ag_train.csv', header=None)
ag_test = pd.read_csv('ag_test.csv', header=None)
ag_train.columns = ['Topic', 'Title', 'Article']
ag_test.columns = ['Topic', 'Title', 'Article']
ag_train.head()
(set(ag_train['Topic']))
print("There are {} articles and 4 Topic in this dataset.".format(ag_train.shape[0]))
ag_test.head()
(set(ag_test['Topic']))
print("There are {} observations articles and 4 Topic in this dataset.".format(ag_test.shape[0]))
ag_news = pd.concat([ag_train, ag_test])
ag_news.head(10)
print("There are {} observations (Articles) and 4 Topic in this dataset.".format(ag_news.shape[0]))
# using headlines and short_description as input X

ag_news['text'] = ag_news.Title + " " + ag_news.Article

ag_news.head(5)
data = ag_news[["Topic", "Title"]]
data.Topic.value_counts().plot.bar(figsize = (10,5))
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
stopwords = nltk.corpus.stopwords.words("english")
stop_words = stopwords
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
def create_wordcloud(Topic):
    text = " ".join(desc for desc in ag_train[ag_train['Topic'] == Topic].Article)
    wordcloud = WordCloud(width=1500, height=800, max_font_size=200, background_color = 'white', stopwords = stopwords).generate(text)
    plt.figure(figsize=(20,15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
create_wordcloud(1)
create_wordcloud(2)
create_wordcloud(3)
create_wordcloud(4)

# Text Normalizing function

def clean_text(text):
    
    # Remove puncuation
    text = text.translate(string.punctuation)
    
    # Convert words to lower case and split them
    text = text.lower().split()
    
    # Remove stop words
    stops = stopwords
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    
    return text
ag_news['text'] = ag_news['text'].map(lambda x: clean_text(x))
ag_news.head(10)
from sklearn.model_selection import train_test_split

X = ag_news['text']
y = ag_news['Topic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer

tokenizer = RegexpTokenizer(r"\w+")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stopwords)
x_train_vec = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
x_train = tfidf_transformer.fit_transform(x_train_vec)

# convert topic occurrences in model-understandable numerical data
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

x_test = vectorizer.transform(X_test)
y_test = encoder.transform(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(x_train, y_train)
print("Accuracy: {}".format(nb.score(x_test, y_test)))
x_test_pred_nb = nb.predict(x_test)
print(classification_report(y_test, x_test_pred_nb))
#NOT MANDATORY

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
#import warnings filter to avoid showing of warning for reaching number of iterations
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

onevsrest = OneVsRestClassifier(LogisticRegression())
onevsrest.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(onevsrest.score(x_test, y_test)))
x_test_pred_ovr = onevsrest.predict(x_test)
print(classification_report(y_test, x_test_pred_ovr))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
# Fit the classifier to the training data
rf.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(rf.score(x_test, y_test)))
from sklearn.metrics import confusion_matrix
x_test_pred_rf = rf.predict(x_test)
print(classification_report(y_test, x_test_pred_rf))
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(dtc.score(x_test, y_test)))
from sklearn.metrics import confusion_matrix
x_test_pred_dtc = dtc.predict(x_test)
print(classification_report(y_test, x_test_pred_dtc))
print("Accuracies comparison")
print("")
print("")
print("Accuracy of Multinomial Naive Bayes Classifier is: {}".format(nb.score(x_test, y_test)))
print("")
print("Accuracy of OneVsRest Classifier is: {}".format(onevsrest.score(x_test, y_test)))
print("")
print("Accuracy of Random Forest Classifier is: {}".format(rf.score(x_test, y_test)))
print("")
print("Accuracy of Decision Trees Classifier is: {}".format(dtc.score(x_test, y_test)))