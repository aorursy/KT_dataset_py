import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import string
import nltk

import warnings
warnings.filterwarnings(action='ignore')

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import re
from wordcloud import WordCloud, STOPWORDS 

import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report

import keras
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/news-aggregator-dataset/uci-news-aggregator.csv')
df.head()
print('Feature ',end=' ')
if(any(df.isnull().any())):
    print('Missing Data\n')
    print(df.isnull().sum())
else:
    print('NO missing data')
df['PUBLISHER'] = df['PUBLISHER'].fillna(df['PUBLISHER'].mode()[0]) # Mode- 'Reuters'
df.info()
print('Data Size {}'.format(df.shape))
if(any(df.duplicated())==True):
    print('Duplicate rows found')
    print('Number of duplicate rows= ',df[df.duplicated()].shape[0])
    df.drop_duplicates(inplace=True,keep='first')
    df.reset_index(inplace=True,drop=True)
    print('Dropping duplicates\n')
    print(df.shape)
else:
    print('NO duplicate data')
# (b = business, t = science and technology, e = entertainment, m = health)

def label_to_name(label):
    if(label=='e'):
        return 'entertainment'
    elif(label=='b'):
        return 'business'
    elif(label=='t'):
        return 'science and technology'
    else:
        return 'health'
    
df['CATEGORY'] = df['CATEGORY'].apply(label_to_name)
print('Distribution of labels in %\n')
print(df['CATEGORY'].value_counts()/df.shape[0]*100)

sns.set(font_scale=1.2)
plt.figure(figsize=(12,6))
sns.countplot(df['CATEGORY']);
df.drop(columns=['ID','URL','PUBLISHER','STORY','HOSTNAME','TIMESTAMP'],inplace=True)


# lowercasing
df['lower'] = df['TITLE'].str.lower()



PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
df["punc_removed"] = df["lower"].apply(lambda text: remove_punctuation(text))



STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df["stopwords_removed"] = df["punc_removed"].apply(lambda text: remove_stopwords(text))


df.head()


# lemmatizer = WordNetLemmatizer()
# def lemmatize_words(text):
#     return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
# df["lemmatized_without_stopwords"] = df["punc_removed"].apply(lambda text: lemmatize_words(text))

# df["lemmatized_stopwords"] = df["stopwords_removed"].apply(lambda text: lemmatize_words(text))




# NO EMOJI IN DATA
# print(all(df['lemmatized'] == df['removed_emoji'])) # TRUE
# def remove_emoji(string):
#     emoji_pattern = re.compile("["
#                            u"\U0001F600-\U0001F64F"  # emoticons
#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
#                            "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', string)
# df["removed_emoji"] = df["lemmatized"].apply(lambda text: remove_emoji(text))




# NO EMOTICONS IN DATA
# print(all(df['lemmatized'] == df['removed_emoticons'])) # TRUE
# def remove_emoticons(text):
#     emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
#     return emoticon_pattern.sub(r'', text)

# df["removed_emoticons"] = df["lemmatized"].apply(lambda text: remove_emoticons(text))
comment_words = ' '
stopwords = set(STOPWORDS) 
  

for val in df.stopwords_removed[0:10000]:  
    tokens = val.split()     
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
le = LabelEncoder()
df['CATEGORY']=le.fit_transform(df['CATEGORY'])

df.head(3)
# convert data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['stopwords_removed'])
y = df['CATEGORY']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=df.CATEGORY)
print('Training Data ',x_train.shape,y_train.shape)
print('Test Data     ',x_test.shape,y_test.shape)


results = pd.DataFrame(columns=['Model','Accuracy','F1-score'])

models_name = ['Logistic Regression','Decision Tree','Multinomial NaiveBayes']

model_list = [LogisticRegression(), DecisionTreeClassifier(),MultinomialNB()]

for idx,model in enumerate(model_list):
    clf = model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results.loc[idx] = [models_name[idx],accuracy_score(y_test, predictions),f1_score(y_test, predictions, average = 'weighted')]

results.sort_values(by='Accuracy',inplace=True,ascending=False)
results
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['stopwords_removed'].values)
y = df['CATEGORY']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=df.CATEGORY)
print('Training Data ',x_train.shape,y_train.shape)
print('Test Data     ',x_test.shape,y_test.shape)


results = pd.DataFrame(columns=['Model','Accuracy','F1-score'])

models_name = ['Logistic Regression','Decision Tree','Multinomial NaiveBayes']

model_list = [LogisticRegression(), DecisionTreeClassifier(),MultinomialNB()]

for idx,model in enumerate(model_list):
    clf = model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results.loc[idx] = [models_name[idx],accuracy_score(y_test, predictions),f1_score(y_test, predictions, average = 'weighted')]

results.sort_values(by='Accuracy',inplace=True,ascending=False)
results
labels = to_categorical(df['CATEGORY'], num_classes=4)

n_most_common_words = 10000
max_len = 130
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df["lower"].values)
sequences = tokenizer.texts_to_sequences(df["lower"].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.2, random_state=42,stratify=df.CATEGORY)

epochs = 10
emb_dim = 150
batch_size = 256
print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.15, recurrent_dropout=0.15))

model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test,y_test),callbacks=callbacks_list)

fig1 = plt.figure(figsize=(12,5))
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
#fig1.savefig('loss.png')
plt.show()
fig2=plt.figure(figsize=(12,5))
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
#fig2.savefig('accuracy.png')
plt.show()
print('** Results for LSTM Model **\n')
predictions = model.predict_classes(X_test)
print("Accuracy score: ", accuracy_score(y_test.argmax(1), predictions)) # to convert OHE vector back to label
print("F1 score: ", f1_score(y_test.argmax(1), predictions, average = 'weighted'))