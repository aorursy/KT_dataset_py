import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.probability import FreqDist

from nltk import ngrams

import string,re



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



import warnings,os
plt.figure(figsize=(16,7))

plt.style.use('ggplot')

warnings.filterwarnings('ignore')
# Locate the data directories

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip',sep='\t')

test=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep='\t')
train.shape, test.shape
train.head()
test.head()
train.info()
train.isnull().sum()
train.head()
train['sentiment_class'] = train['Sentiment'].map({0:'negative',1:'somewhat negative',2:'neutral',3:'somewhat positive',4:'positive'})

train.head()
def remove_punctuation(text):

    return "".join([t for t in text if t not in string.punctuation])
train['Phrase']=train['Phrase'].apply(lambda x:remove_punctuation(x))

train.head()
def words_with_more_than_three_chars(text):

    return " ".join([t for t in text.split() if len(t)>3])
train['Phrase']=train['Phrase'].apply(lambda x:words_with_more_than_three_chars(x))

train.head()
stop_words=stopwords.words('english')

train['Phrase']=train['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

train.head()
train.groupby('Sentiment')['Sentiment'].count()
train.groupby('sentiment_class')['sentiment_class'].count().plot(kind='bar',title='Target class',figsize=(16,7),grid=True)
((train.groupby('sentiment_class')['sentiment_class'].count()/train.shape[0])*100).plot(kind='pie',figsize=(7,7),title='% Target class', autopct='%1.0f%%')
train['PhraseLength']=train['Phrase'].apply(lambda x: len(x))
train.sort_values(by='PhraseLength', ascending=False).head()
plt.figure(figsize=(16,7))

bins=np.linspace(0,200,50)

plt.hist(train[train['sentiment_class']=='negative']['PhraseLength'],bins=bins,density=True,label='negative')

plt.hist(train[train['sentiment_class']=='somewhat negative']['PhraseLength'],bins=bins,density=True,label='somewhat negative')

plt.hist(train[train['sentiment_class']=='neutral']['PhraseLength'],bins=bins,density=True,label='neutral')

plt.hist(train[train['sentiment_class']=='somewhat positive']['PhraseLength'],bins=bins,density=True,label='somewhat positive')

plt.hist(train[train['sentiment_class']=='positive']['PhraseLength'],bins=bins,density=True,label='positive')

plt.xlabel('Phrase length')

plt.legend()

plt.show()
# Install wordcoud library

# !pip install wordcloud
from wordcloud import WordCloud, STOPWORDS 

stopwords = set(STOPWORDS) 
word_cloud_common_words=[]  

for index, row in train.iterrows(): 

    word_cloud_common_words.append((row['Phrase'])) 

word_cloud_common_words



wordcloud = WordCloud(width = 1600, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 5).generate(''.join(word_cloud_common_words)) 

  

# plot the WordCloud image                        

plt.figure(figsize = (16, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
text_list=[]  

for index, row in train.iterrows(): 

    text_list.append((row['Phrase'])) 

text_list



total_words=''.join(text_list)

total_words=word_tokenize(total_words)
freq_words=FreqDist(total_words)

word_frequency=FreqDist(freq_words)
# 10 common words

print(word_frequency.most_common(10))
# visualize 

pd.DataFrame(word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)
neg_text_list=[]  

for index, row in train[train['Sentiment']==0].iterrows(): 

    neg_text_list.append((row['Phrase'])) 

neg_text_list



neg_total_words=' '.join(neg_text_list)

neg_total_words=word_tokenize(neg_total_words)



neg_freq_words=FreqDist(neg_total_words)

neg_word_frequency=FreqDist(neg_freq_words)
# visualize 

pd.DataFrame(neg_word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)
pos_text_list=[]  

for index, row in train[train['Sentiment']==4].iterrows(): 

    pos_text_list.append((row['Phrase'])) 

pos_text_list



pos_total_words=' '.join(pos_text_list)

pos_total_words=word_tokenize(pos_total_words)



pos_freq_words=FreqDist(pos_total_words)

pos_word_frequency=FreqDist(pos_freq_words)
# visualize 

pd.DataFrame(pos_word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)
text="Tom and Jerry love mickey. But mickey dont love Tom and Jerry. What a love mickey is getting from these two friends"

bigram_frequency = FreqDist(ngrams(word_tokenize(text),3))

bigram_frequency.most_common()[0:5]
text_list=[]  

for index, row in train.iterrows(): 

    text_list.append((row['Phrase'])) 

text_list



total_words=' '.join(text_list)

total_words=word_tokenize(total_words)



freq_words=FreqDist(total_words)

word_frequency=FreqDist(ngrams(freq_words,2))

word_frequency.most_common()[0:5]
# visualize 

pd.DataFrame(word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)
train['tokenized_words']=train['Phrase'].apply(lambda x:word_tokenize(x))

train.head()
count_vectorizer=CountVectorizer()

phrase_dtm=count_vectorizer.fit_transform(train['Phrase'])
phrase_dtm.shape
X_train,X_val,y_train,y_val=train_test_split(phrase_dtm,train['Sentiment'],test_size=0.3, random_state=38)

X_train.shape,y_train.shape,X_val.shape,y_val.shape
model=LogisticRegression()
model.fit(X_train,y_train)
accuracy_score(model.predict(X_val),y_val)*100
del X_train

del X_val

del y_train

del y_val
tfidf=TfidfVectorizer()

tfidf_dtm=tfidf.fit_transform(train['Phrase'])
X_train,X_val,y_train,y_val=train_test_split(tfidf_dtm,train['Sentiment'],test_size=0.3, random_state=38)

X_train.shape,y_train.shape,X_val.shape,y_val.shape
tfidf_model=LogisticRegression()
tfidf_model.fit(X_train,y_train)
accuracy_score(tfidf_model.predict(X_val),y_val)*100
print(tfidf_model.predict(X_val)[0:10])
def predict_new_text(text):

    tfidf_text=tfidf.transform([text])

    return tfidf_model.predict(tfidf_text)
predict_new_text("The movie is bad and sucks!")
test['Phrase']=test['Phrase'].apply(lambda x:remove_punctuation(x))

test['Phrase']=test['Phrase'].apply(lambda x:words_with_more_than_three_chars(x))

test['Phrase']=test['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

test_dtm=tfidf.transform(test['Phrase'])
# Predict with test data

test['Sentiment']=tfidf_model.predict(test_dtm)

test.set_index=test['PhraseId']

test.head()
# save results to csv file

# test.to_csv('Submission.csv',columns=['PhraseId','Sentiment'],index=False)