import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
business=pd.read_csv("../input/yelp_business.csv")
business.head()
x = business['stars'].value_counts().index
y = business['stars'].value_counts().values
plt.figure(figsize=(9,6))
ax= sns.barplot(x, y,data= business ,alpha=0.8 )
plt.title("Ratings Distribution")
plt.xlabel('Ratings ', fontsize=12)
plt.figure(figsize=(9,6))
ax= sns.barplot(x = 'stars', y='review_count',data= business ,alpha=0.8 )
plt.title("Ratings Distribution")
plt.xlabel('Ratings ', fontsize=12)
business['categories'].head()
business_cat=' '.join(business['categories'])
categry=pd.DataFrame(business_cat.split(';'),columns=['category'])
x = categry.category.value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:20]
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()
x = business['city'].value_counts().sort_values(ascending = False)
x=x.iloc[0:25]
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()
x = business['name'].value_counts().sort_values(ascending = False)

x=x.iloc[0:25]
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()
busi_attr = pd.read_csv('../input/yelp_review.csv') 
busi_attr = busi_attr[:100000]
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
nltk.download('punkt')
a = busi_attr['text'].str.lower().str.cat(sep=' ')
import re
b = re.sub('[^A-Za-z]+', ' ', a)
b[:1000]
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)
word_tokens = word_tokenize(b)
len(word_tokens)
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
len(filtered_sentence)
# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]   
top_N = 100
word_dist = nltk.FreqDist(cleaned_data_title)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))
from wordcloud import WordCloud, STOPWORDS
def wc(data,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
wc(cleaned_data_title,'black','Most Used Words')
from textblob import TextBlob

bloblist_desc = list()

df_review_str=busi_attr['text'].astype(str)
for row in df_review_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])
df_polarity_desc.head()
def f(df_polarity_desc):
    if df_polarity_desc['sentiment'] > 0:
        val = "Positive Review"
    elif df_polarity_desc['sentiment'] == 0:
        val = "Neutral Review"
    else:
        val = "Negative Review"
    return val
df_polarity_desc['Sentiment_Type'] = df_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_polarity_desc)
positive_reviews=df_polarity_desc[df_polarity_desc['Sentiment_Type']=='Positive Review']
negative_reviews=df_polarity_desc[df_polarity_desc['Sentiment_Type']=='Negative Review']
negative_reviews.head()
wc(positive_reviews['Review'],'black','Most Used Words')
wc(negative_reviews['Review'],'black','Most Used Words')
busi_attr=busi_attr.dropna(axis=0,how='any')
rating_class = busi_attr[(busi_attr['stars'] == 1) | (busi_attr['stars'] == 5)]
X_review=rating_class['text']
y=rating_class['stars']
import string
def text_process(review):
    nopunc=[word for word in review if word not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_review)
X_review = bow_transformer.transform(X_review)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.3, random_state=101)
X_train
from sklearn.svm import SVC
sv_model = SVC()
sv_model.fit(X_train, y_train)
Y_pred = sv_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, Y_pred))
print('\n Accuracy:')
print(accuracy_score(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)
Y_pred = lg_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, Y_pred))
print('\n Accuracy:')
print(accuracy_score(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
predict=nb.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, predict))
print('\n Accuracy:')
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
