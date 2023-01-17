import pandas as pd 

import numpy as np 



#Plots

import matplotlib.pyplot as plt

import seaborn as sns



#Map

from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter

import folium 

from folium import plugins 



#Worldcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



#Regex

import re



#String

import string



#Sklearn

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix



#Spacy

import spacy

from spacy import displacy

nlp=spacy.load('en_core_web_sm')



sns.set_style('whitegrid')

%matplotlib inline
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.info()
print("Train dataset: {} rows arranged in {} columns.".format(train.shape[0],train.shape[1]))

print("Test dataset: {} rows arranged in {} columns.".format(test.shape[0],test.shape[1]))
train.head()
test.head()
missing_columns = ['keyword', 'location']





fig = plt.figure(figsize=(14,6))



ax1=fig.add_subplot(121)

sns.barplot(x=train[missing_columns].isnull().sum().index, y=train[missing_columns].isnull().sum().values,palette='mako',ax=ax1)

ax1.set_title('Missing values in train set')



ax2=fig.add_subplot(122)

sns.barplot(x=test[missing_columns].isnull().sum().index, y=test[missing_columns].isnull().sum().values,palette='mako',ax=ax2)

ax2.set_title('Missing values in test set')



fig.suptitle('Missing values in dataset')

plt.show()
#Extract number of target values.

values=train.target.value_counts()

plt.figure(figsize=(7,6))

sns.barplot(x=values.index,y=values,palette=['blue','red'])

plt.ylabel('Samples')

plt.xlabel('0:Not disaster | 1:Disaster')

plt.title('Distribution of target values',fontsize=16)

plt.show()
disaster = train.target.value_counts()[1]/len(train.target)

not_disaster = train.target.value_counts()[0]/len(train.target)

percentage = {'Disaster tweets %':[disaster], 'Non disaster tweets %':[not_disaster]}

percentage_data = pd.DataFrame(percentage)

percentage_data.head()
data = train.groupby('target').size()



data.plot(kind='pie', subplots=True, figsize=(10, 8), autopct = "%.2f%%", colors=['blue','red'])

plt.title("Pie chart of different types of disasters",fontsize=16)

plt.ylabel("")

plt.legend()

plt.show()
train.length = train.text.apply(len)
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

sns.boxplot(x=train.target[train.target==0],y=train.length, ax=ax1,color='blue')

describe = train.length[train.target==0].describe().to_frame().round(2)



ax2 = fig.add_subplot(122)

ax2.axis('off')

font_size = 16

bbox = [0, 0, 1, 1]

table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)

table.set_fontsize(font_size)

fig.suptitle('Distribution of text length for non disaster tweets.', fontsize=16)



plt.show()
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

sns.boxplot(x=train.target[train.target==1],y=train.length, ax=ax1,color='red')

describe = train.length[train.target==1].describe().to_frame().round(2)



ax2 = fig.add_subplot(122)

ax2.axis('off')

font_size = 16

bbox = [0, 0, 1, 1]

table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)

table.set_fontsize(font_size)

fig.suptitle('Distribution of text length for disaster tweets.', fontsize=16)



plt.show()
print('Unique keywords in train data: {}'.format(len(train.keyword.unique())))

print('Unique keywords in test data: {}'.format(len(test.keyword.unique())))
fig = plt.figure(figsize=(14,6))



ax1 = fig.add_subplot(111)

train_keywords=sns.barplot(x=train.keyword.value_counts()[:25].index,y=train.keyword.value_counts()[:25][:],palette='icefire',ax=ax1)

train_keywords.set_xticklabels(train_keywords.get_xticklabels(),rotation=90)

train_keywords.set_ylabel('Keyword frequency')

plt.title('Top 25 common keywords in train data',fontsize=16)

plt.show()
plt.figure(figsize=(14,6))

common_keyword=sns.barplot(x=train.location.value_counts()[:25].index,y=train.location.value_counts()[:25][:],palette='icefire')

common_keyword.set_xticklabels(common_keyword.get_xticklabels(),rotation=90)

common_keyword.set_ylabel('Location frequency',fontsize=12)

plt.title('Top 25 common location of tweets for train data',fontsize=16)

plt.show()
fig = plt.figure(figsize=(14,6))



ax1 = fig.add_subplot(111)

test_keywords=sns.barplot(x=test.keyword.value_counts()[:25].index,y=test.keyword.value_counts()[:25][:],palette='icefire',ax=ax1)

test_keywords.set_xticklabels(test_keywords.get_xticklabels(),rotation=90)

test_keywords.set_ylabel('Keyword frequency')

plt.title('Top 25 common keywords in test data',fontsize=16)

plt.show()
train['number_of_words'] = train.text.apply(lambda x: len((str(x).split())))

train['number_of_unique_words'] = train.text.apply(lambda x: len(set(str(x).split())))
fig,ax = plt.subplots(ncols=2,figsize=(14,7))



word_count_1 = sns.distplot(train.number_of_words[train.target==1],color='red',ax=ax[0])

word_count_0 = sns.distplot(train.number_of_words[train.target==0],color='blue',ax=ax[0])

unique_count_1 = sns.distplot(train.number_of_unique_words[train.target==1],color='red',ax=ax[1])

unique_count_0 = sns.distplot(train.number_of_unique_words[train.target==0],color='blue',ax=ax[1])

word_count_1.set_title('Number of words used to disaster vs. non disaster tweets')

unique_count_1.set_title('Number of unique words used to disaster vs. non disaster tweets')

plt.suptitle('Analysis of number of words used in tweets',fontsize=16)





plt.show()
print('Unique keywords in train data: {}'.format(len(train.location.unique())))

print('Unique keywords in test data: {}'.format(len(test.location.unique())))
plt.figure(figsize=(14,6))

common_keyword=sns.barplot(x=test.location.value_counts()[:25].index,y=test.location.value_counts()[:25][:],palette='icefire')

common_keyword.set_xticklabels(common_keyword.get_xticklabels(),rotation=90)

common_keyword.set_ylabel('Location frequency',fontsize=12)

plt.title('Top 25 common location of tweets for test data',fontsize=16)

plt.show()
data = train.location.value_counts()[:25,]

data = pd.DataFrame(data)

data = data.reset_index()

data.columns = ['location', 'counts'] 

geolocator = Nominatim(user_agent="Location Map")

geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)



dict_latitude = {}

dict_longitude = {}

for i in data.location.values:

    print(i)

    location = geocode(i)

    dict_latitude[i] = location.latitude

    dict_longitude[i] = location.longitude

data['latitude'] = data.location.map(dict_latitude)

data['longitude'] = data.location.map(dict_longitude)
location_map = folium.Map(location=[7.0, 7.0], zoom_start=2)

markers = []

for i, row in data.iterrows():

    loss = row['counts']

    if row['counts'] > 0:

        count = row['counts']*0.4

    folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=float(count), color='red', fill=True).add_to(location_map)

location_map
example="AMAZING NOTEBOOK(shameless self promotion :D): https://www.kaggle.com/michawilkosz/simple-way-to-top-26-blended-regression-model"
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
print(remove_URL(example))
train['text'] = train['text'].apply(lambda x : remove_URL(x))

test['text'] = test['text'].apply(lambda x : remove_URL(x))
example = """<div>

<h1>House Prices Notebook</h1>

<p>Simple way to top 26 blended regression model</p>

<a href="https://www.kaggle.com/michawilkosz/simple-way-to-top-26-blended-regression-model</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)
print(remove_html(example))
train['text']=train['text'].apply(lambda x : remove_html(x))

test['text']=test['text'].apply(lambda x : remove_html(x))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
print(remove_emoji("Oh no! Another pandemic 😔😔"))
train['text']=train['text'].apply(lambda x : remove_emoji(x))

test['text']=test['text'].apply(lambda x : remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"
print(remove_punct(example))
test.dropna(how='any',inplace=True,axis=1)

train.dropna(how='any',inplace=True,axis=1)
def exctract_text(data, target):

    extracted=[]

    

    for x in data[data['target']==target]['text'].str.split():

        for i in x:

            extracted.append(i)

    return extracted
extracted_text_1 = exctract_text(train,1)

extracted_text_0 = exctract_text(train,0)
plt.figure(figsize=(14,6))

word_cloud = WordCloud(background_color="white",max_font_size=60).generate(" ".join(extracted_text_1[:50]))

plt.imshow(word_cloud,interpolation='bilinear')

plt.axis('off')

plt.title('Most common words in disaster tweets.',fontsize=20)

plt.show()
plt.figure(figsize=(14,6))

word_cloud = WordCloud(background_color="white",max_font_size=60).generate(" ".join(extracted_text_0[:50]))

plt.imshow(word_cloud,interpolation='bilinear')

plt.axis('off')

plt.title('Most common words in non disaster tweets.',fontsize=20)

plt.show()
import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



marks = string.punctuation

marks = list(marks)

marks.append("...")

marks.append("....")







nlp = spacy.load('en')

stop_words = list(STOP_WORDS)

def tokenizer(sentence):

    doc = nlp(sentence)

    clean_tokens = []

    for token in doc:

        if token.lemma_ != '-PRON-':

            token = token.lemma_.lower().strip()

        else:

            token = token.lower_

        if token not in stop_words and token not in marks:

            clean_tokens.append(token)

    return clean_tokens
bow_vector = CountVectorizer(tokenizer = tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = tokenizer)
X = train['text']

y = train['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



#Pipeline

lr_pipe = Pipeline([('vectorizer', bow_vector),

                 ('classifier', classifier)])



# model generation

lr_pipe.fit(X_train,y_train)
# Predicting with a test dataset

lr_pred = lr_pipe.predict(X_test)



# Model Accuracy

print("Logistic Regression Accuracy:",accuracy_score(y_test, lr_pred))

print("Logistic Regression Precision:",precision_score(y_test, lr_pred))

print("Logistic Regression Recall:",recall_score(y_test, lr_pred))
svc = LinearSVC()

svc_pipe = Pipeline([('tfidf', tfidf_vector), ('clf', svc)])

svc_pipe.fit(X_train,y_train)
svc_pred = svc_pipe.predict(X_test)



# Model Accuracy

print("SVC Accuracy:",accuracy_score(y_test, svc_pred))

print("SVC Precision:",precision_score(y_test, svc_pred))

print("SVC Regression Recall:",recall_score(y_test, svc_pred))
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization



max_len=512
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    callbacks=[checkpoint],

    batch_size=16

)
model.load_weights('model.h5')

test_pred = model.predict(test_input)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)