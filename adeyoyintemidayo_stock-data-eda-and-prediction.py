# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from nltk.stem.porter import PorterStemmer

from nltk import word_tokenize, WordNetLemmatizer

import nltk

import re 

nltk.download('wordnet')

  
## Load in the data

data = pd.read_csv("../input/stockmarket-sentiment-dataset/stock_data.csv")
## Read the data

data.head()
data.shape
data.info()
data.describe()
## Sentiment Value count 

data["Sentiment"].value_counts()
## Plot the Sentiment value count 

sns.countplot(data["Sentiment"])
## Lenght of the Text using KDEplot

lenght = data["Text"].str.len()

sns.kdeplot(lenght)
## Checking for stopwords

from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))

print(stop_words)
word_list = list()

for i in range(len(data)):

    lip = data.Text[i].split()

    for k in lip:

        word_list.append(k)
from collections import Counter 

wordCounter = Counter(word_list)

countedWordDict = dict(wordCounter)

sortedWordDict = sorted(countedWordDict.items(),key = lambda x : x[1],reverse=True)

sortedWordDict[0:20]
from wordcloud import WordCloud

wordList2 = " ".join(word_list)

stop_word_Cloud = set(stopwords.words("english"))

wordcloud = WordCloud(stopwords=stop_word_Cloud,max_words=2000,background_color="white",min_font_size=3).generate_from_frequencies(countedWordDict)

plt.figure(figsize=[20,10])

plt.axis("off")

plt.imshow(wordcloud)

plt.show()
## Replacing the negative one with zero so our model can predict well

data["Sentiment"] = data["Sentiment"].replace(-1,0)
## Lets check our data again

data["Sentiment"].value_counts()
data.shape
## NlP Processing

ps = PorterStemmer()

lemma = WordNetLemmatizer()

stopwordSet = set(stopwords.words("english"))
## Clean the text 

text_reviews = list()

for i in range(len(data)):

    text = re.sub('[^a-zA-Z]'," ",data['Text'][i])

    text = text.lower()

    text = word_tokenize(text,language="english")

    text = [lemma.lemmatize(word) for word in text if(word) not in stopwordSet]

    text = " ".join(text)

    text_reviews.append(text)
## Create the (B.O.W) bag of word model

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(text_reviews).toarray()

y= data['Sentiment']



## Split the dataset into Training and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state = 0)
## Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)
print(classification_report(y_test, Y_pred))
print(confusion_matrix(y_test, Y_pred))
## Naives baye multinomial

clf = MultinomialNB()

clf.fit(X_train, y_train)

Y_pred = clf.predict(X_test)
print(classification_report(y_test, Y_pred))
print(confusion_matrix(y_test, Y_pred))
## Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)
print(classification_report(y_test, Y_pred))
print(confusion_matrix(y_test, Y_pred))