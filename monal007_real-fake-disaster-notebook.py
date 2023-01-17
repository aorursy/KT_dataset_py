import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams, bigrams , trigrams
from nltk.corpus import stopwords
from nltk import ne_chunk

df = pd.read_csv(r"../input/disaster-dataset/train.csv")
df1 = pd.read_csv(r"../input/disaster-dataset/test.csv")
sub = pd.read_csv(r"../input/disaster-dataset/sample_submission.csv")

df
import missingno as msg
msg.matrix(df)
import missingno as msg
msg.matrix(df1)
df.head()
df = df.drop( ['id'] , axis = 1)
df1 = df1.drop(['id'] , axis = 1)

df.head()
df['keyword'].value_counts().sort_values(ascending = False)
sb.countplot(df.keyword)
df.keyword.unique()
df = df.drop( ['keyword'] , axis = 1)
df1 = df1.drop(['keyword'] , axis = 1)

df
df['location'].value_counts().sort_values(ascending = False)
df.location.unique()
df = df.drop( ['location'] , axis = 1)
df1 = df1.drop(['location'] , axis = 1)

df
df.iloc[0 ,0]
import re
def clean_data(tweet):
    tweet = re.sub("RT @[\w]*:", "", tweet)
    tweet = re.sub("@[\w]*", "", tweet)
    tweet = re.sub("https://[A-Za-z0-9./]", "", tweet)
    tweet = re.sub("\n", "", tweet)
    tweet = re.sub("&amp", "", tweet)
    tweet = re.sub("#", "", tweet)
    tweet = re.sub(r"[^\w]", ' ', tweet )
    return tweet
df1
df['text'] = df['text'].apply(lambda x: clean_data(x))
df1['text'] = df1['text'].apply(lambda x: clean_data(x))

df
df = df.sample(frac =1).reset_index(drop = True)
df['text'] = df['text'].apply(lambda x : x.lower())
df1['text'] = df1['text'].apply(lambda x : x.lower())

df
df1
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df
z = df1.text.values
z
stop = stopwords.words('english')
df1['text'] = df1['text'].apply(lambda z: ' '.join([word for word in z.split() if word not in (stop)]))

df1
import matplotlib.pyplot as plt
from wordcloud import WordCloud
fake_data = df[df["target"] == 0]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
from wordcloud import WordCloud
real_data = df[df["target"] == 1]
all_words = ' '.join([text for text in real_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
df
x,y = df.text , df.target
x = x.values
y = y.values
cvt = TfidfVectorizer(tokenizer=word_tokenize , ngram_range=(1,2))
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state =100)
cvt.fit(x_train)
x_train_trans = cvt.transform(x_train)

x_test_trans = cvt.transform(x_test)

x_train_trans
x_test_trans
#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf = TfidfTransformer()
#tfidf.fit(x_train_trans)
#x_traintf = tfidf.transform(x_train_trans)

#x_testtf = tfidf.transform(x_test_trans)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_trans , y_train)
pred = model.predict(x_test_trans)
from sklearn.metrics import accuracy_score
accuracy_score(pred , y_test)
df1
df1_trans = cvt.transform(df1.text)
sub
sub.target = model.predict(df1_trans)
sub
sub.target.value_counts()
#Now save the file and so the submission
# Upvote, If you learnt something new :)