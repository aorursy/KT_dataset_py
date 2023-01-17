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
data=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

data
for i in data:

    print(data[i].unique())
data=data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

list1 = data['v1'].unique()

list1.sort()



dict1 = dict(zip(list1, range(len(list1))))

data['v1'].replace(dict1, inplace=True)

data
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

spam = data[data["v1"] == 1]["v2"]

ham = data[data["v1"] == 0]["v2"]



spam_words = []

ham_words = []





def extractSpamWords(spamMessages):

    global spam_words

    words = [word.lower() for word in word_tokenize(spamMessages) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

    spam_words = spam_words + words

    

def extractHamWords(hamMessages):

    global ham_words

    words = [word.lower() for word in word_tokenize(hamMessages) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

    ham_words = ham_words + words



spam.apply(extractSpamWords)

ham.apply(extractHamWords)
import matplotlib.pyplot as plt 
from wordcloud import WordCloud

spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam_words))

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()

import string

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



def cleanText(message):

    

    message = message.translate(str.maketrans('', '', string.punctuation))

    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]

    

    return " ".join(words)



data["v2"] = data["v2"].apply(cleanText)

data.head(n = 10) 
data.iloc[5,1]
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

features = vec.fit_transform(data['v2'])

print(features.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, data["v1"], stratify = data["v1"], test_size = 0.2)
X_train.shape
from sklearn.metrics import accuracy_score



from sklearn.naive_bayes import MultinomialNB

gaussianNb = MultinomialNB()

gaussianNb.fit(X_train, y_train)



y_pred = gaussianNb.predict(X_test)



print(accuracy_score(y_test, y_pred, beta = 0.5))
#print(gaussianNb.predict(features))