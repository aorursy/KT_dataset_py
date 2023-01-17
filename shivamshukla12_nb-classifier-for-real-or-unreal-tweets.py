print('Tweets Real / Fake Classification......')
import os
import pandas as pd
lv_path = r'../input/nlp-getting-started/'
print(os.listdir(lv_path))
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
train_df = pd.read_csv(r'../input/nlp-getting-started/train.csv')
train_df.head(5)
print('training data shape: ', train_df.shape)
sub_df = pd.read_csv(r'../input/nlp-getting-started/sample_submission.csv')
sub_df.head(5)
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
test_df.head(5)
train_df['target'].value_counts()
import seaborn as sns
sns.barplot(train_df['target'].value_counts().index,train_df['target'].value_counts(),palette='rocket')
# A disaster tweet
disaster_tweets = train_df[train_df['target']==1]['text']
disaster_tweets.values[1]
#not a disaster tweet
non_disaster_tweets = train_df[train_df['target']==0]['text']
non_disaster_tweets.values[1]
train_df.loc[train_df['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
# A quick glance over the existing data
train_df['text'][:5]
import re
print(train_df.text[3])
re.sub("RT @[\w]*:", "" , train_df.text[3])
def clean_data(tweet):
    tweet = re.sub("RT @[\w]*:", "", tweet)
    tweet = re.sub("@[\w]*", "", tweet)
    tweet = re.sub("https://[A-Za-z0-9./]", "", tweet)
    tweet = re.sub("\n", "", tweet)
    tweet = re.sub("&amp", "", tweet)
    tweet = re.sub("#", "", tweet)
    tweet = re.sub(r"[^\w]", ' ', tweet )
    return tweet
train_df['text'] = train_df['text'].apply(lambda x: clean_data(x))
test_df['text'] = test_df['text'].apply(lambda x: clean_data(x))
train_df.head()
test_df.head()
train_df['text'] = train_df['text'].apply(lambda x: x.lower())
train_df['text']
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
train_df['text'] = train_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train_df['text']
import matplotlib.pyplot as plt
from wordcloud import WordCloud
fake_data = train_df[train_df["target"] == 0]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

from wordcloud import WordCloud
fake_data = train_df[train_df["target"] == 1]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
from sklearn.feature_extraction.text import CountVectorizer
X, y = train_df['text'], train_df['target']
# #DataFlair - Split the dataset
# x_train,x_test,y_train,y_test=train_test_split(train_df['target'], labels, test_size=0.2, random_state=7)
count_vect = CountVectorizer()
x_train_df = count_vect.fit_transform(X)
# x_train_tr = count_vect.fit_transform(train_df.target)
# x_train_df
x_train_df.shape, train_df.shape
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
x_traintf = tfidf.fit_transform(x_train_df)
x_traintf.shape
# labels = train_df.target
# labels
train_df.head(2)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_traintf, y)
test_in = [
#     'deeds reason earthquake may allah forgive us',
#     'summer lovely',
#     'damage school bus 80 multi car crash breaking',
#     'man'
'The U.S. Army released new guidelines for optimal soldier performance — and they include strategic and aggressive napping']
X_test_in = count_vect.transform(test_df.text) #(test_in)
x_test_tf = tfidf.transform(X_test_in)
x_test_tf
pred = clf.predict(x_test_tf)
pred
train_df.head(20)
sub_df['target'] = pred.round().astype(int)
sub_df.to_csv(r'Your System Path', index=False)

