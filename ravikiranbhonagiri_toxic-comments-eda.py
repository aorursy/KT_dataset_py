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
import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation
df_train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

df_test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

df_test_labels = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip")

df_sub = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")
df_train.head()
df_test.head()
df_test_labels.head()
df_sub.head()
df_train.shape
df_train.describe()
df_train.dtypes
df_train.isna().sum()
feature_cols = ['comment_text']

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

total_cols = feature_cols + label_cols


value_counts_1 = []



for col in label_cols:

  plt.figure(figsize=(6,4))

  df_train[col].value_counts().plot.bar()

  print(df_train[col].value_counts())

  value_counts_1.append( df_train[col].value_counts()[1] )

  plt.xlabel(col)

  plt.ylabel("Count")

  plt.show()



plt.figure(figsize=(8,6))

plt.barh(label_cols, value_counts_1)

plt.xlabel("Labels")

plt.ylabel("Count")

#plt.xticks(label_cols, range(len(label_cols)))

#plt.show()



for index, value in enumerate(value_counts_1):

  plt.text(value, index, str(value))



#print(label_cols)

#print(value_counts_1)
df_train["is_labeled"] = df_train[label_cols].sum(axis=1)
df_train.head()
plt.figure(figsize=(8,6))

df_train["is_labeled"].value_counts().plot.barh()

print(df_train["is_labeled"].value_counts())

for index in range(7):

  plt.text(df_train["is_labeled"].value_counts()[index], index, str(df_train["is_labeled"].value_counts()[index]))
import nltk

from nltk import word_tokenize

import re

nltk.download('punkt')

nltk.download('stopwords')

from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

#stemmer = PorterStemmer("english")

stemmer = SnowballStemmer("english")
def removeHTMLTags(input_str):

  regTag = re.compile('<.*?>')

  cleantext = re.sub(regTag, ' ', str(input_str))

  return cleantext
def removePunctChars(input_str):

  processed_str = re.sub(r'[?|!|\'|"|#]',r'',input_str)

  processed_str = re.sub(r'[.|,|)|(|\|/]',r' ',processed_str)

  processed_str = processed_str.strip()

  processed_str = processed_str.replace("\n"," ")

  return processed_str
def removeOtherSpecialChars(input_str):

  modified_str = ""

  for word in input_str.split():

    mod_word = re.sub('[^a-zA-Z ]+', ' ', word)

    modified_str += mod_word 

    modified_str += " "

  return modified_str.strip()

def makeLower(input_str):

  return input_str.lower()
def removeStopWords(input_str):

  modified_str = ""

  for word in input_str.strip().split():

    if word not in stop_words:

      modified_str += word

      modified_str += " "

  return modified_str.strip()
def stemmingWords(input_str):

  modified_str = ""

  for word in input_str.strip().split():

    modified_word = stemmer.stem(word)

    modified_str += modified_word

    modified_str += " "

  return modified_str.strip()
df_train["comment_text"].head(10)
df_train["comment_text"] = df_train["comment_text"].apply(makeLower)

df_train["comment_text"] = df_train["comment_text"].apply(removeHTMLTags)

df_train["comment_text"] = df_train["comment_text"].apply(removePunctChars)

df_train["comment_text"] = df_train["comment_text"].apply(removeOtherSpecialChars)

df_train["comment_text"] = df_train["comment_text"].apply(removeStopWords)

df_train["comment_text"] = df_train["comment_text"].apply(stemmingWords)
df_train["comment_text"].head(10)
comment_text_length = [len(m_str.split()) for m_str in df_train["comment_text"].tolist()]
plt.plot(comment_text_length)
print( "Max Length ", max(comment_text_length))

print( "Min Length ", min(comment_text_length))

from wordcloud import WordCloud,STOPWORDS



def showWordCloud(df, label):

  plt.figure(figsize=(15,12))



  text = df[df[label]==1]["comment_text"].tolist()

  label_img = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=1200,

                          height=800,

                         ).generate(" ".join(text))



  plt.title(label,fontsize=40)

  plt.imshow(label_img)

showWordCloud(df_train, "toxic")
showWordCloud(df_train, "severe_toxic")

showWordCloud(df_train, "obscene")

showWordCloud(df_train, "threat")



showWordCloud(df_train, "insult")

showWordCloud(df_train, "identity_hate")
df_train["comment_text"].head()
from sklearn.model_selection import train_test_split



train, test = train_test_split(df_train, random_state=45, test_size=0.25, shuffle=True)



print(train.shape)

print(test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')

vectorizer.fit(train["comment_text"])

vectorizer.fit(test["comment_text"])
x_train = vectorizer.transform(train["comment_text"])

y_train = train.drop(labels = ['id','comment_text'], axis=1)



x_test = vectorizer.transform(test["comment_text"])

y_test = test.drop(labels = ['id','comment_text'], axis=1)
# print(x_train[:5])

print(y_train[:5])



# print(x_test[:5])

print(y_test[:5])
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.multiclass import OneVsRestClassifier

import pickle


# Using pipeline for applying logistic regression and one vs rest classifier

LogReg_pipeline = Pipeline([

                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),

            ])



for category in label_cols:

    print('Processing {} comments'.format(category))

    

    # Training logistic regression model on train data

    LogReg_pipeline.fit(x_train, train[category])

    

    # calculating test accuracy

    prediction = LogReg_pipeline.predict(x_test)

    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

    print("\n")