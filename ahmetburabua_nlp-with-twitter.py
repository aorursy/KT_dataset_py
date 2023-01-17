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
import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from textblob import TextBlob

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

from sklearn.model_selection import train_test_split



from warnings import filterwarnings

filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train.head()
train.isnull().sum()
train.drop(["keyword","location"],axis = 1,inplace = True)
train["text"] = train["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
train["text"]
train["text"] = train["text"].str.replace("[\d]","")
train["text"]
train["text"] = train["text"].str.replace("[^\w\s]","")
train["text"]
import nltk

from nltk import WordNetLemmatizer

from nltk.corpus import stopwords



sw = stopwords.words("english")
sw.append("u")

sw.append("im")
train["text"] = train["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
train["text"]
from textblob import Word

nltk.download("wordnet")
train["text"] = train["text"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
train["text"]
train["text"] = train["text"].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"")

train["text"] = train["text"].str.replace(r"We're", "We are")

train["text"] = train["text"].str.replace(r"That's", "That is")

train["text"] = train["text"].str.replace(r"won't", "will not")

train["text"] = train["text"].str.replace(r"they're", "they are")

train["text"] = train["text"].str.replace(r"Can't", "Cannot")

train["text"] = train["text"].str.replace(r"wasn't", "was not")

train["text"] = train["text"].str.replace(r"don\x89Ûªt", "do not")

train["text"] = train["text"].str.replace(r"aren't", "are not")

train["text"] = train["text"].str.replace(r"isn't", "is not")

train["text"] = train["text"].str.replace(r"You're", "You are")

train["text"] = train["text"].str.replace(r"I'M", "I am")

train["text"] = train["text"].str.replace(r"shouldn't", "should not")

train["text"] = train["text"].str.replace(r"wouldn't", "would not")

train["text"] = train["text"].str.replace(r"i'm", "I am")

train["text"] = train["text"].str.replace(r"We've", "We have")

train["text"] = train["text"].str.replace(r"Didn't", "Did not")

train["text"] = train["text"].str.replace(r"it's", "it is")

train["text"] = train["text"].str.replace(r"can't", "cannot")

train["text"] = train["text"].str.replace(r"don't", "do not")

train["text"] = train["text"].str.replace(r"you're", "you are")

train["text"] = train["text"].str.replace(r"I've", "I have")

train["text"] = train["text"].str.replace(r"Don't", "do not")

train["text"] = train["text"].str.replace(r"I'll", "I will")

train["text"] = train["text"].str.replace(r"Let's", "Let us")

train["text"] = train["text"].str.replace(r"Could've", "Could have")

train["text"] = train["text"].str.replace(r"youve", "you have")

train["text"] = train["text"].str.replace(r"It's", "It is")
train["text"].head()
from sklearn.model_selection import train_test_split
tf1 = (train["text"][0:1000]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ["words","tf"]
tf
x = tf1[tf1["tf"] > 30].sort_values(by = "tf" , ascending = False)
x.plot.bar( x = "words", y = "tf", color = "green");
from wordcloud import WordCloud
for i in range(1,5):

    text = train["text"][i]

    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()

    
x_train,x_test,y_train,y_test = train_test_split(train["text"],train["target"],

                                                test_size = 0.3,

                                                random_state = 18)
vectorizer = CountVectorizer()

vectorizer.fit(x_train)
x_train_count = vectorizer.transform(x_train)

x_test_count = vectorizer.transform(x_test)
lr = linear_model.LogisticRegression()

lr_model = lr.fit(x_train_count,y_train)

y_pred = lr_model.predict(x_test_count)



accuracy_score(y_test,y_pred)
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test.info()
test.drop(["keyword","location"], axis = 1,inplace = True)
test.info()
def prep(test):

    test["text"] = test["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

    

    test["text"] = test["text"].str.replace("[\d]","")

    

    test["text"] = test ["text"].str.replace("[^\w\s]","")

    

    test["text"] = test["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

    

    test["text"] = test["text"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

    

    test["text"] = test["text"].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"")

    

    test["text"] = test["text"].str.replace(r'(((http)(s)?|www(.)?)(://)?\S+)',"")

    

    test["text"] = test["text"].str.replace(r"\x89ÛÓ", "")

    test["text"] = test["text"].str.replace(r"he's", "he is")

    test["text"] = test["text"].str.replace(r"there's", "there is")

    test["text"] = test["text"].str.replace(r"We're", "We are")

    test["text"] = test["text"].str.replace(r"That's", "That is")

    test["text"] = test["text"].str.replace(r"won't", "will not")

    test["text"] = test["text"].str.replace(r"they're", "they are")

    test["text"] = test["text"].str.replace(r"Can't", "Cannot")

    test["text"] = test["text"].str.replace(r"wasn't", "was not")

    test["text"] = test["text"].str.replace(r"don\x89Ûªt", "do not")

    test["text"] = test["text"].str.replace(r"aren't", "are not")

    test["text"] = test["text"].str.replace(r"isn't", "is not")

    test["text"] = test["text"].str.replace(r"You're", "You are")

    test["text"] = test["text"].str.replace(r"I'M", "I am")

    test["text"] = test["text"].str.replace(r"shouldn't", "should not")

    test["text"] = test["text"].str.replace(r"wouldn't", "would not")

    test["text"] = test["text"].str.replace(r"i'm", "I am")

    test["text"] = test["text"].str.replace(r"We've", "We have")

    test["text"] = test["text"].str.replace(r"Didn't", "Did not")

    test["text"] = test["text"].str.replace(r"it's", "it is")

    test["text"] = test["text"].str.replace(r"can't", "cannot")

    test["text"] = test["text"].str.replace(r"don't", "do not")

    test["text"] = test["text"].str.replace(r"you're", "you are")

    test["text"] = test["text"].str.replace(r"I've", "I have")

    test["text"] = test["text"].str.replace(r"Don't", "do not")

    test["text"] = test["text"].str.replace(r"I'll", "I will")

    test["text"] = test["text"].str.replace(r"Let's", "Let us")

    test["text"] = test["text"].str.replace(r"Could've", "Could have")

    test["text"] = test["text"].str.replace(r"youve", "you have")

    test["text"] = test["text"].str.replace(r"It's", "It is")

    

    return test

    
df = prep(test)
test_x = df["text"] 
vectorizer = CountVectorizer()

vectorizer.fit(test_x)

vectorizer.transform(test_x)
x_test_vec = vectorizer.transform(test_x)
x_train_vec = vectorizer.transform(x_train)
nb = naive_bayes.MultinomialNB()

nb_model = nb.fit(x_train_vec,y_train)

y_pred = nb_model.predict(x_test_vec)

y_pred

dictionary_ = {}

dictionary_["id"] = test.id

dictionary_["target"] = y_pred

submission = pd.DataFrame(dictionary_)
submission
submission.to_csv("submission_1.csv" , index = None)