# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Lib for ploting interective graph.
import seaborn as sns
import cufflinks as cf
#init_notebook_mode(connected=True)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()
print(__version__) # requires version >= 1.9.0
import matplotlib.pyplot as plt
#import pandas_profiling
%matplotlib inline
df=pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv",index_col=False)
column_contain=['Clothing ID','Age','Title','Review Text','Rating','Recommended IND','Positive Feedback Count','Division Name','Department Name','Class Name']
#We will have a look of our data.
df.head()
#Here we will check count,mean,std,min,max for our datafram
df.describe()
# Have a look of your column names
df.columns
#Checking correlation in both column
df['Review Text']=df['Review Text'].astype(str)
df['Review Length']=df['Review Text'].apply(len)
#Shoing FactGrid for different ratings
g=sns.FacetGrid(data=df,col='Rating',palette='plasma')
g.map(plt.hist,'Review Length',bins=50)
#Displaying box plot for more clarity in Rating and Review Length
plt.figure(figsize=(10,10))
sns.boxplot(x='Rating',y='Review Length',data=df)
#Define Corr of Rating 
rating=df.groupby('Rating').mean()
rating.corr()
sns.heatmap(data=rating.corr(),annot=True)
df.groupby(['Rating', pd.cut(df['Age'], np.arange(0,100,10))])\
       .size()\
       .unstack(0)\
       .plot.bar(stacked=True)
plt.figure(figsize=(15,15))
df.groupby(['Department Name', pd.cut(df['Age'], np.arange(0,100,10))])\
       .size()\
       .unstack(0)\
       .plot.bar(stacked=True)
z=df.groupby(by=['Department Name'],as_index=False).count().sort_values(by='Class Name',ascending=False)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x=z['Department Name'],y=z['Class Name'], data=z,palette='plasma')
plt.xlabel("Department Name")
plt.ylabel("Count")
plt.title("Counts Vs Department Name")
w=df.groupby(by=['Division Name'],as_index=False).count().sort_values(by='Class Name',ascending=False)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x=w['Division Name'],y=w['Class Name'], data=w,palette='Greens')
plt.xlabel("Division Name")
plt.ylabel("Count")
plt.title("Counts Vs Division Name")
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
import re

top_N = 100
#convert list of list into text
#a=''.join(str(r) for v in df_usa['title'] for r in v)

a = df['Review Text'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)

#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)

word_tokens = word_tokenize(b)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]        

# Calculate frequency distribution
word_dist = nltk.FreqDist(cleaned_data_title)
rslt = pd.DataFrame(word_dist.most_common(50),
                    columns=['Word', 'Frequency'])

plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(20))

def wc(data,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 500,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
    
wc(cleaned_data_title,'#ff4d4d','Most Used Words')
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
import re
#convert list of list into text
#a=''.join(str(r) for v in df_usa['title'] for r in v)
top_N=100
a = df['Review Text'].str.lower().str.cat(sep=' , ')
#a=df.iloc[0:10, 4:5].str.lower().str.cat(sep=' ')
#a
# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)
#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)
#nltk_words
#stop_words
word_tokens = word_tokenize(b)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
#cntp=len(word_tokens)
type(filtered_sentence)
#filtered_sentence
# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 3]
#without_single_chr
# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
from collections import Counter
counter=Counter(filtered_sentence)
#print(counter)
word_dist = nltk.FreqDist(cleaned_data_title)
#word_dist
rslt = pd.DataFrame(word_dist.most_common(50),
                    columns=['Word', 'Frequency'])
#rslt
from textblob import TextBlob

bloblist_desc = list()

df_review_str=df['Review Text'].astype(str)
for row in df_review_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])
 
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

wc(positive_reviews['Review'],'green','Most Used Words')
wc(negative_reviews['Review'],'darkgray','Most Used Words')
import string
def text_process(review):
    nopunc=[word for word in review if word not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

dbf=df['Review Text'].head(5).apply(text_process)

#wc(df['Review Text'].head(5).apply(text_process),'#8533ff','Most Used Words')
df['Review Text'].head(5).apply(text_process)
df=df.dropna(axis=0,how='any')
rating_class = df[(df['Rating'] == 1) | (df['Rating'] == 5)]
X_review=rating_class['Review Text']
y=rating_class['Rating']
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_review)
print(len(bow_transformer.vocabulary_))
X_review = bow_transformer.transform(X_review)

print(X_review)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train,y_train)
predict=nb.predict(X_test)
predict
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))
rating_positive=df['Review Text'][3]
rating_positive
rating_positive_transformed = bow_transformer.transform([rating_positive])
nb.predict(rating_positive_transformed)[0]
rating_negative=df['Review Text'][61]
rating_negative
rating_negative_transformed = bow_transformer.transform([rating_negative])
nb.predict(rating_negative_transformed)[0]
X_predict_recommend=df['Review Text']
#X_predict_recommend
y_recommend=df['Recommended IND']
#y_recommend

bow_transformer=CountVectorizer(analyzer=text_process).fit(X_predict_recommend)
X_predict_recommend = bow_transformer.transform(X_predict_recommend)
X_train, X_test, y_train, y_test = train_test_split(X_predict_recommend, y_recommend, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)
predict_recommendation=nb.predict(X_test)
print(confusion_matrix(y_test, predict_recommendation))
print('\n')
print(classification_report(y_test, predict_recommendation))
rating_positive
rating_positive_transformed = bow_transformer.transform([rating_positive])
nb.predict(rating_positive_transformed)[0]
rating_negative
rating_negative_transformed = bow_transformer.transform([rating_negative])
nb.predict(rating_negative_transformed)[0]
