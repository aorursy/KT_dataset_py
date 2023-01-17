import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import os

from xgboost import XGBClassifier

from wordcloud import WordCloud

from nltk import pos_tag

import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords, wordnet

from nltk.tokenize import sent_tokenize, word_tokenize

import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,SGDClassifier, LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Embedding, LSTM

from keras.models import Sequential

from keras.utils import to_categorical
# načítanie dát

train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train_df.head()
test_df.head()
# filling nan values in the columns. 

train_df.keyword.fillna('', inplace=True)

train_df.location.fillna('', inplace=True)



test_df.keyword.fillna('', inplace=True)

test_df.location.fillna('', inplace=True)
train_df['text'] = train_df['text'] + ' ' + train_df['keyword'] + ' ' + train_df['location']

test_df['text'] = test_df['text'] + ' ' + test_df['keyword'] + ' ' + test_df['location']



del train_df['keyword']

del train_df['location']

del train_df['id']

del test_df['keyword']

del test_df['location']

del test_df['id']
train_df.head()
test_df.head()
sns.countplot(train_df.target)
# treba odstrániť slová ako "a", "that" alebo "there", tzv. stopwords, ktoré nám nenapomáhajú pri rozlišovaní

stop = set(stopwords.words('english'))

punctuations = list(string.punctuation)

stop.update(punctuations)

print(stop)
# Funkcie na očistenie textu od čísel, linkov

def remove_numbers(text):

    text = ''.join([i for i in text if not i.isdigit()])         

    return text



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F" 

                           u"\U0001F300-\U0001F5FF"

                           u"\U0001F680-\U0001F6FF" 

                           u"\U0001F1E0-\U0001F1FF"  

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
train_df.text = train_df.text.apply(remove_numbers)

train_df.text = train_df.text.apply(remove_URL)

train_df.text = train_df.text.apply(remove_html)

train_df.text = train_df.text.apply(remove_emoji)

train_df.head()
test_df.text = test_df.text.apply(remove_numbers)

test_df.text = test_df.text.apply(remove_URL)

test_df.text = test_df.text.apply(remove_html)

test_df.text = test_df.text.apply(remove_emoji)

test_df.head()
# určenie slovného druhu

def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    clean_text = []

    for w in word_tokenize(text):

        if w.lower() not in stop:

            pos = pos_tag([w])

            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))

            clean_text.append(new_w)

    return " ".join(clean_text)
train_df.text = train_df.text.apply(clean_text)

test_df.text = test_df.text.apply(clean_text)
real = train_df.text[train_df.target[train_df.target==1].index]

fake = train_df.text[train_df.target[train_df.target==0].index]
# rozdelíme si dáta na treningovú a validačnú časť

x_train_text, x_val_text, y_train, y_val = train_test_split(train_df.text, train_df.target, test_size=0.2, random_state=0)
#niektoré slová, ktoré sa vyskytujú veľmi často môžeme ignorovať, keďže slová ktoré sa príliš často vyskytujú

#nám nepomôžu pri predpovedaní - to isté platí pri slovách ktoré sú veľmi zriedkavé

#na ignorovanie takýchto slov používame min_df a max_df

tv=TfidfVectorizer(min_df=0,max_df=0.8,use_idf=True,ngram_range=(1,3))





tv_train_reviews=tv.fit_transform(x_train_text)

tv_val_reviews=tv.transform(x_val_text)

tv_test_reviews=tv.transform(test_df.text)



print('tfidf_train:',tv_train_reviews.shape)

print('tfidf_validation:',tv_val_reviews.shape)

print('tfidf_test:',tv_test_reviews.shape)
model = Sequential()



model.add(Dense(units = 512 , activation = 'relu' , input_dim = tv_train_reviews.shape[1]))

model.add(Dropout(0.2))

model.add(Dense(units = 256 , activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(units = 100 , activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(units = 10 , activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(units = 1 , activation = 'sigmoid'))



model.compile(optimizer = 'nadam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])



model.summary()
history = model.fit(tv_train_reviews, y_train, validation_data=(tv_val_reviews, y_val), batch_size=128, epochs=5)
# Porovnanie treningových a validačných dát

plt.figure(figsize=(10,12))

plt.subplot(221)

plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')



plt.subplot(222)

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()
model_val_predict = model.predict_classes(tv_val_reviews)

cm = confusion_matrix(y_val, model_val_predict)

cm = pd.DataFrame(cm , index = [i for i in range(2)] , columns = [i for i in range(2)])

plt.figure(figsize = (8,6))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
y_pred = model.predict_classes(tv_test_reviews)

submission.target = y_pred

submission.to_csv("submission.csv" , index = False)