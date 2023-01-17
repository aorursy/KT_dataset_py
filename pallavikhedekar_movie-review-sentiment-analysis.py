# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
from wordcloud import WordCloud # Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head()
test.head()

train['sentiment_label'] = ''
train.loc[train.Sentiment == 0, 'sentiment_label'] = 'Negative'
train.loc[train.Sentiment == 1, 'sentiment_label'] = 'Somewhat Negative'
train.loc[train.Sentiment == 2, 'sentiment_label'] = 'Neutral'
train.loc[train.Sentiment == 3, 'sentiment_label'] = 'Somewhat Positive'
train.loc[train.Sentiment == 4, 'sentiment_label'] = 'Positive'
train.head()
train.sentiment_label.value_counts()
train.shape
from nltk.stem.wordnet import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[%s]' % re.escape(string.digits), '', text)
    text = re.sub('[%s]' % re.escape(' +'), ' ', text)
    text=re.sub('[^a-zA-Z]',' ',text)
    text=[wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(str(text).lower())]
    text=' '.join(text)
    text = text.lower()
    text = text.strip()
    return text

    
train['cleaned_phrase'] = ''
train['cleaned_phrase'] = [clean_text(phrase) for phrase in train.Phrase]
test['cleaned_phrase'] = ''
test['cleaned_phrase'] = [clean_text(phrase) for phrase in test.Phrase]


train['phrase_length'] = [len(sent.split(' ')) for sent in train.cleaned_phrase]
test['phrase_length'] = [len(sent.split(' ')) for sent in test.cleaned_phrase]

train.head()
test.head()
import matplotlib.pyplot as plt
import seaborn as sns
classwise_count = train['sentiment_label'].value_counts()
classwise_count
fig, ax = plt.subplots(1, 1,dpi=80, figsize=(10,5))
sns.barplot(x=classwise_count.index,y=classwise_count)
ax.set_ylabel('Number of reviews')    
ax.set_xlabel('Sentiment Label')
ax.set_xticklabels(classwise_count.index , rotation=30)
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
Stopwords = list(ENGLISH_STOP_WORDS) + stopwords.words()
def wordcloud(sentiment):
    stopwordslist = Stopwords
    ## extend list of stopwords with the common words between the 3 classes which is not helpful to represent them
    stopwordslist.extend(['movie','movies','film','nt','rrb','lrb','make','work','like','story','time','little'])
    reviews = train.loc[train.Sentiment.isin(sentiment)]
    print("Word Cloud for Sentiment Labels: ", reviews.sentiment_label.unique())
    phrases = ' '.join(reviews.cleaned_phrase)
    words = " ".join([word for word in phrases.split()])
    wordcloud = WordCloud(stopwords=stopwordslist,width=3000,height=2500,background_color='white',).generate(words)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
    plt.axis("off")
    plt.show()
wordcloud([3,4])

wordcloud([0,1])
wordcloud([2])
print('Number of sentences in training set:',len(train['SentenceId'].unique()))
print('Number of sentences in test set:',len(test['SentenceId'].unique()))
print('Average words per sentence in train:',train.groupby('SentenceId')['Phrase'].count().mean())
print('Average words per sentence in test:',test.groupby('SentenceId')['Phrase'].count().mean())
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train['Phrase'])
test_vectorized = vectorizer.transform(test['Phrase'])
y = train['Sentiment']
from sklearn.model_selection import train_test_split
x_train , x_val, y_train , y_val = train_test_split(train_vectorized,y,test_size = 0.2)
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

svm = LinearSVC()
svm.fit(x_train,y_train)
print(classification_report( svm.predict(x_val) , y_val))
print(accuracy_score( svm.predict(x_val) , y_val ))
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
classifier = linear_model.LogisticRegression(C=2.6, solver='sag')
classifier.fit(x_train, y_train)
print(classification_report( classifier.predict(x_val) , y_val))
print(accuracy_score( classifier.predict(x_val) , y_val ))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
print(classification_report( classifier.predict(x_val) , y_val))
print(accuracy_score( classifier.predict(x_val) , y_val ))
train_text=train.cleaned_phrase.values
test_text=test.cleaned_phrase.values
target=train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)


x_train,x_val,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(x_train.shape,y_train.shape)
print(x_val.shape,y_val.shape)



from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from nltk import FreqDist
all_words=' '.join(x_train)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
print(num_unique_word)
max_features = num_unique_word
max_words = max(train.phrase_length)
batch_size = 128
epochs = 3
num_classes=5
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(test_text)


x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_val = sequence.pad_sequences(x_val, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
print(x_train.shape,x_val.shape,x_test.shape)


model1=Sequential()
model1.add(Embedding(max_features,100,mask_zero=True))
model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model1.add(Dense(num_classes,activation='softmax'))
model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model1.summary()
%%time
history1=model1.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=10, batch_size=batch_size, verbose=1)
y_pred1=model1.predict_classes(x_test,verbose=1)
sub.Sentiment=y_pred1
sub.to_csv('sub.csv',index=False)
sub.head()