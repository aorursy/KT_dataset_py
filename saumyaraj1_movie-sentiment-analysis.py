import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


train = pd.read_csv("../input/train.tsv",sep = "\t")
test = pd.read_csv("../input/test.tsv",sep = "\t")
train
test.head()
sns.countplot(train["Sentiment"])
train["Sentiment"].value_counts(normalize = True)*100
train["number of words"] = train["Phrase"].apply(lambda x: len(x.split(" ")))
test["number of words"] = test["Phrase"].apply(lambda x: len(x.split(" ")))
test[test.Phrase == " "]
test.Phrase =  test.Phrase.replace(" ","no value")
train.Phrase =  train.Phrase.replace(" ","no value")

sns.distplot(train["number of words"])
sns.distplot(test["number of words"])
sns.violinplot("Sentiment","number of words",data = train)
train["number of characters"] = train["Phrase"].apply(lambda x: sum(len(x) for x in x.split(" ")))
test["number of characters"] = test["Phrase"].apply(lambda x: sum(len(x) for x in x.split(" ")))
sns.violinplot("Sentiment","number of characters",data = train)
def avg_word_len(x):
    words = x.split(" ")
    avg_length = sum(len(word) for word in words)/len(words)
    return avg_length
train["avg word length"] = train["Phrase"].apply(lambda x: avg_word_len(x))
test["avg word length"] = test["Phrase"].apply(lambda x: avg_word_len(x))
sns.violinplot("Sentiment","avg word length",data = train)
sns.distplot(train["avg word length"])
plt.show()
sns.distplot(test["avg word length"])
plt.show()
#avg word length should be ommited ,giving distinction only after 25
from nltk.corpus import stopwords
stop = stopwords.words("english")
train["Phrase"] = train["Phrase"].apply(lambda x : x.lower())
test["Phrase"] = test["Phrase"].apply(lambda x : x.lower())
def num_stop(x):
    words = x.split(" ")
    t = []
    for word in words:
        if word in stop:
            t.append(word)
    return len(t)
train["number of stopwords"] = train["Phrase"].apply(lambda x : num_stop(x))
test["number of stopwords"] = test["Phrase"].apply(lambda x : num_stop(x))
sns.violinplot("Sentiment","number of stopwords",data = train)
sns.distplot(train["number of stopwords"])
plt.show()
sns.distplot(test["number of stopwords"])
plt.show()
from nltk.stem import PorterStemmer
st = PorterStemmer()
train["Phrase"] = train["Phrase"].apply(lambda x: " ".join([st.stem(word) for  word in x.split()]))
test["Phrase"] = test["Phrase"].apply(lambda x: " ".join([st.stem(word) for  word in x.split()]))
train.head()
train["Phrase"] = train["Phrase"].str.replace("[^\w\s]","")
test["Phrase"] = test["Phrase"].str.replace("[^\w\s]","")
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(stop_words = "english")

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators = 200,max_depth = 100)

data = tf.fit_transform(train.Phrase)
xgb.fit(data,train.Sentiment)
new = tf.transform(test.Phrase)
test["Sentiment"] = xgb.predict(new)
test[["PhraseId","Sentiment"]].to_csv("sample_submission.csv",index = False)
