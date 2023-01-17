import re
import tensorflow as tf
import torch
from torch import nn
from torch import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
stemming = PorterStemmer()
stops = set(stopwords.words("english"))
from sklearn.linear_model import SGDClassifier
df = pd.read_csv('train.csv')
df.rename(columns={' Review': 'Review', ' Prediction':
                   'Prediction'}, inplace=True)
df = df.dropna()
feature_columns = 'Review'
target_column = 'Prediction'

df1 = pd.read_csv('test.csv')

def apply_cleaning_function_to_list(text_to_clean):
    cleaned_text = []
    for raw_text in text_to_clean:
        cleaned_text.append(clean_text(raw_text))
    return cleaned_text


def clean_text(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    phrase_list = [w.lower() for w in tokens if w.isalpha()]
    meaningful_words = [w for w in phrase_list if not w in stops]
    stemmed_words = [stemming.stem(w) for w in meaningful_words]
    phrase = ""
    for words in stemmed_words:
      phrase += words + ' '
    return phrase
text_to_clean = list(df['Review'])
cleaned_text = apply_cleaning_function_to_list(text_to_clean)
df['Review'] = cleaned_text
df['Review'].replace('', np.nan, inplace=True)
df = df.dropna()
df = df.loc[(df['Review'].str.len() != 0), :]
del cleaned_text
del text_to_clean

test_text_to_clean = list(df1['Review'])
test_cleaned_text = apply_cleaning_function_to_list(test_text_to_clean)
df1['Review'] = test_cleaned_text
df1['Review'].replace('bad', np.nan, inplace=True)
test_text = list(df1['Review'])
for i in range(len(test_text)):
  if len(test_text[i]) == 0:
    test_text[i] = "good"
df1['Review'] = test_text
#df1.to_csv(r'/content/df1.csv')
del test_cleaned_text
del test_text_to_clean
del test_text

y = list(df['Prediction'])
my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])
vectorizer = TfidfVectorizer(stop_words=my_stop_words, ngram_range=(1, 2),max_features = 10000)


X_test_train = vectorizer.fit_transform(list(df['Review']) + list(df1['Review'])).toarray()
X = X_test_train[:len(list(df['Review']))]
test_X = X_test_train[len(list(df['Review'])):]
X = csr_matrix(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.25, )
model = LogisticRegression(penalty="l2", solver="newton-cg", max_iter=100)

model.fit(X_train,y_train)
score = model.score(X_test, y_test)
print(score)
Predictions = model.predict(test_X)
csvfilelist = [['Id', 'Prediction']]
for k in range(len(Predictions)):
  mylist = []
  mylist.append(k+1)
  mylist.append(Predictions[k])
  csvfilelist.append(mylist)
mylist = []
for k in range(len(Predictions)):
  mylist.append(k+1)
"""preddf = pd.DataFrame(csvfilelist)
print(preddf)"""
preddf = pd.DataFrame(data={"Id": mylist, "Prediction": Predictions})
preddf.to_csv("prediction.csv", sep=',',index=False)
#preddf.to_csv('prediction.csv', index=False)
print(preddf)