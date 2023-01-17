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
df = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter="\t")
df = df.drop(['id'], axis=1)
df.head()
df.info()
df.sentiment.value_counts()
df1=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip",delimiter= "\t")
df1.head()
train_len=df['review'].apply(len)
train_len.describe()

test_len=df['review'].apply(len)
test_len.describe()
import matplotlib.pyplot as plt
import seaborn as sns
fig=plt.figure(figsize=(14,8))
fig.add_subplot(1,2,1)
sns.distplot((train_len),color='red')

fig.add_subplot(1,2,2)
sns.distplot((test_len),color='blue')
df['word_n'] = df['review'].apply(lambda x : len(x.split(' ')))
df1["word_n"]=df1["review"].apply(lambda x : len(x.split(" ")))
fig=plt.figure(figsize=(14,6))
fig.add_subplot(1,2,1)
sns.distplot(df['word_n'],color='red')

fig.add_subplot(1,2,2)
sns.distplot(df1['word_n'],color='blue')
sns.countplot(df['sentiment'])
from wordcloud import WordCloud
cloud=WordCloud(width=800, height=600).generate(" ".join(df['review'])) 
# join function can help merge all words into one string. " " means space can be a seperator between words.
plt.figure(figsize=(16,10))
plt.imshow(cloud)
plt.axis('off')
import re
import json

TAG_RE = re.compile(r'<[^>]+>')
df['review']=df['review'].apply(lambda x:TAG_RE.sub('', x))
df1['review']=df1['review'].apply(lambda x: TAG_RE.sub('', x))
from wordcloud import WordCloud
cloud=WordCloud(width=800, height=600).generate(" ".join(df['review'])) 
# join function can help merge all words into one string. " " means space can be a seperator between words.
plt.figure(figsize=(16,10))
plt.imshow(cloud)
plt.axis('off')
df['review']=df['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
df1['review']=df1['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
df1.sample(4)
df["review"].str.find("?").value_counts()
df['word_n_2'] = df['review'].apply(lambda x : len(x.split(' ')))
df1['word_n_2'] = df1['review'].apply(lambda x : len(x.split(' ')))

fig, axe = plt.subplots(1,1, figsize=(7,5))
sns.boxenplot(x=df['sentiment'], y=df['word_n_2'], data=df)
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
# lemmatizer = WordNetLemmatizer()
df["review"]=df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df1["review"]=df1['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
test=df1.drop(["word_n","word_n_2","id"],axis=1)
X=df.drop(["word_n","word_n_2","sentiment"],axis=1)
X.head(3)
Y=df.drop(["word_n","word_n_2","review"],axis=1)
Y.head(3)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df["review"])
text_counts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, Y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['review'])
text_tf
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_tf,Y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
