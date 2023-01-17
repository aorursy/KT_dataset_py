!nvidia-smi
!pip install --upgrade pip
!pip install tensorflow-gpu
!pip install --upgrade grpcio
!pip install tqdm
!pip install bert-for-tf2
!pip install sentencepiece
import os

import math

import datetime

from tqdm import tqdm

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

import bert

from bert import BertModelLayer

from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline

%config InlineBackend.figure_format='retina'

sns.set(font_scale=1.2)

plt.style.use('ggplot')

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)
import re

import matplotlib.image as image

import matplotlib.colors

from collections import defaultdict

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import squarify as sq

from colorama import Fore, Back, Style
df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')

df.head()
df.shape
# Checking the data for duplicates

df[df.duplicated('description',keep=False)].sort_values('description').head(5)
# Dropping all duplicates

df.drop_duplicates(('description', 'title'), inplace=True)

df[pd.notnull(df.price)]

df.shape
# Missing values

total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.style.background_gradient(cmap='seismic')
# Imputing missing values

for col in ('region_2', 'designation', 'taster_twitter_handle', 'taster_name', 'region_1'):

    df[col]=df[col].fillna('Unknown')

df['province'] = df['province'].fillna(df['province'].mode())

df['price'] = df['price'].fillna(df['price'].mean())
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
data = df['country'].replace("US", "United States").value_counts()

iplot([go.Choropleth(

    locationmode='country names',

    locations=data.index.values,

    text=data.index,

    z=data.values,

    colorscale='portland'

)])
# Countries with the most wine reviews

countries = df.country.value_counts()

# Limit top countries to those with more than 500 reviews

temp_dict = countries[countries>500].to_dict()

temp_dict['Other'] = countries[countries<501].sum()

less_countries = pd.Series(temp_dict)

less_countries.sort_values(ascending=False, inplace=True)

# Turn Series into DataFrame for display purposes

df1 = less_countries.to_frame()

df1.columns=['Number of Reviews']

df1.index.name = 'Country'

df1.style.background_gradient(cmap='coolwarm')
# Tree map 

cmap = plt.cm.gist_rainbow_r

norm = matplotlib.colors.Normalize(vmin=0, vmax=15)

colors = [cmap(norm(value)) for value in range(15)]

np.random.shuffle(colors)

fig,ax = plt.subplots(1,1,figsize=(16, 10))

sq.plot(sizes=less_countries.values, label=less_countries.index.values, alpha=0.5, ax=ax, color=colors)

plt.axis('off')

plt.title('Countries by Number of Wine Reviews')
fig = iplot([go.Scatter(x=df.head(1000)['points'],

                  y=df.head(1000)['price'],

                  mode='markers', marker_color='darkred')])
data = df.assign(n=0).groupby(['points', 'price'])['n'].count().reset_index()

data = data[data["price"] < 100]

v = data.pivot(index='price', columns='points', values='n').fillna(0).values.tolist()

iplot([go.Surface(z=v)])
w = df.groupby(['country','points'])['price'].agg(['count','min','max','mean']).sort_values(by='mean',ascending=False)[:10]

w.reset_index(inplace=True)

w.style.background_gradient(cmap='Wistia', high=0.5)
print(Fore.YELLOW + 'Number of variety of wines', df['variety'].nunique())

fig,ax = plt.subplots(1,2,figsize=(16,8))

ax1,ax2 = ax.flatten()

w = df.groupby(['variety'])['price'].max().sort_values(ascending=False).to_frame()[:15]

sns.barplot(x = w['price'], y = w.index, color='brown',ax=ax1)

ax1.set_title('The grapes used for most expensive wine')

ax1.set_ylabel('Variety')

ax1.set_xlabel('')

w = df.groupby(['variety'])['points'].max().sort_values(ascending=False).to_frame()[:15]

sns.barplot(x = w['points'], y = w.index, color='brown',ax=ax2)

ax2.set_title('The grapes used for most rated wine')

ax2.set_ylabel('')

ax2.set_xlabel('')

plt.subplots_adjust(wspace=0.3);
fig,ax = plt.subplots(1,2,figsize=(16,8))

ax1,ax2 = ax.flatten()

w = df.groupby(['variety'])['price'].min().sort_values(ascending=True).to_frame()[:15]

sns.barplot(x = w['price'], y = w.index, color='y',ax=ax1)

ax1.set_title('The grapes used for least priced wine')

ax1.set_xlabel('')

ax1.set_ylabel('Variety')

w = df.groupby(['variety'])['points'].min().sort_values(ascending=True).to_frame()[:15]

sns.barplot(x = w['points'], y = w.index, color='y', ax=ax2)

ax2.set_title('The grapes used for least rated wine')

ax2.set_xlabel('')

ax2.set_ylabel('')

plt.subplots_adjust(wspace=0.4);
fig,ax = plt.subplots(1,2,figsize=(16,8))

ax1,ax2 = ax.flatten()

w = df.groupby(['country'])['price'].max().sort_values(ascending=False).to_frame()[:15]

sns.barplot(x = w['price'], y = w.index, color='purple',ax=ax1)

ax1.set_title('Most expensive wine by country')

ax1.set_ylabel('Variety')

ax1.set_xlabel('')

w = df.groupby(['country'])['price'].min().sort_values(ascending=True).to_frame()[:15]

sns.barplot(x = w['price'], y = w.index, color='purple',ax=ax2)

ax2.set_title('Least priced wine by country')

ax2.set_ylabel('')

ax2.set_xlabel('')

plt.subplots_adjust(wspace=0.3);
fig,ax = plt.subplots(1,2,figsize=(16,8))

ax1,ax2 = ax.flatten()

w = df.groupby(['country'])['points'].max().sort_values(ascending=False).to_frame()[:15]

sns.barplot(x = w['points'], y = w.index, color='yellow',ax=ax1)

ax1.set_title('Most rated wine by country')

ax1.set_ylabel('Variety')

ax1.set_xlabel('')

w = df.groupby(['country'])['points'].min().sort_values(ascending=True).to_frame()[:15]

sns.barplot(x = w['points'], y = w.index, color='yellow',ax=ax2)

ax2.set_title('Least rated wine by country')

ax2.set_ylabel('')

ax2.set_xlabel('')

plt.subplots_adjust(wspace=0.3);
print(Fore.BLUE + Style.BRIGHT + 'Number of province list in data:', df['province'].nunique())

plt.figure(figsize=(14,10))

w = df['province'].value_counts().to_frame()[0:20]

#plt.xscale('log')

sns.barplot(x= w['province'], y =w.index, data=w, color='crimson', orient='h')

plt.title('Distribution of Wine Reviews by Top 20 Provinces');
print(Fore.RED + Style.BRIGHT + 'Number of vineyard designation', df['designation'].nunique())

w = df.groupby(['designation'])['price'].mean().to_frame().sort_values(by='price',ascending=False)[:15]

f,ax = plt.subplots(1,2,figsize= (14,6))

ax1,ax2 = ax.flatten()

sns.barplot(w['price'], y = w.index, color='cyan', ax = ax1)

ax1.set_xlabel('')

ax1.set_ylabel('Designation(Vineyard)')

ax1.set_title('Most expensive wine prepared in the vineyard')

w = df.groupby(['designation'])['points'].mean().to_frame().sort_values(by = 'points', ascending = False)[:15]

sns.barplot(w['points'], y = w.index, color='cyan', ax = ax2)

ax2.set_xlabel('')

ax2.set_ylabel('')

ax2.set_title('Most rated wine prepared in the vineyard')

plt.subplots_adjust(wspace=0.3)
print(Fore.RED + Style.BRIGHT + 'Number of wineries:', df['winery'].nunique())

f,ax = plt.subplots(1,2,figsize=(16,6))

ax1,ax2 = ax.flatten()

w = df.groupby(['winery'])['price'].max().to_frame().sort_values(by='price',ascending=False)[:15]

sns.barplot(w['price'],y = w.index, color='black',ax = ax1)

ax1.set_title('Wineries with the most expensive wines')

w = df.groupby(['winery'])['points'].max().to_frame().sort_values(by = 'points', ascending = False)[:15]

sns.barplot(w['points'], y = w.index, color='black')

plt.title('Wineries with the most rated wines');
stopwords = set(STOPWORDS)

newStopWords = ['fruit', "Drink", "black", 'wine', 'drink']

stopwords.update(newStopWords)

wordcloud = WordCloud(

    stopwords=stopwords,

    colormap='Set1',

    max_words=300,

    max_font_size=200, 

    width=1000, height=800,

    random_state=42,

).generate(" ".join(df['description'].astype(str)))

print(wordcloud)

fig = plt.figure(figsize = (12,14))

plt.imshow(wordcloud)

plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)

plt.axis('off')
wordcloud = WordCloud(

    background_color='white',

    stopwords=stopwords,

    colormap='autumn_r',

    max_words=300,

    max_font_size=200, 

    width=1000, height=800,

    random_state=42,

).generate(" ".join(df['variety'].astype(str)))

print(wordcloud)

fig = plt.figure(figsize = (12,14))

plt.imshow(wordcloud)

plt.title("WORD CLOUD - VARIETY",fontsize=25)

plt.axis('off')
!python -m spacy download en_core_web_lg

import spacy

nlp = spacy.load('en_core_web_lg')

def normalize_text(text):

    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)

    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)

    tm3 = re.sub('<[^>]+>©', '', tm1, flags=re.DOTALL)

    return tm3.replace("\n", "")
# Removing code syntax from text 

df['description_Cleaned_1'] = df['description'].apply(normalize_text)
print(Fore.MAGENTA + 'Before normalizing text-----\n')

print(df['description'])

print(Fore.YELLOW + Style.DIM + '\nAfter normalizing text-----\n')

print(df['description_Cleaned_1'])
from spacy import displacy

about_interest_text = ('I like different types of wine')

about_interest_doc = nlp(about_interest_text)

displacy.render(about_interest_doc, style='dep')
# Stop words

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

len(spacy_stopwords)

for stop_word in list(spacy_stopwords)[:10]:

    print(Fore.CYAN + stop_word)
doc = nlp(df["description"][3])
review = str(" ".join([i.lemma_ for i in doc]))
doc = nlp(review)

spacy.displacy.render(doc, style='ent', jupyter=True)
# Part of Speech Tagging

import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

punctuations = string.punctuation

stopwords = STOP_WORDS
# POS tagging

for i in nlp(review):

    print(i, Fore.GREEN + "=>",i.pos_)
# Parser for reviews

parser = English()

def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens
tqdm.pandas()

df["processed_description"] = df["description"].progress_apply(spacy_tokenizer)
# Topic Modeling

# Creating a vectorizer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

data_vectorized = vectorizer.fit_transform(df["processed_description"])
NUM_TOPICS = 10
# Latent Dirichlet Allocation Model

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)

data_lda = lda.fit_transform(data_vectorized)
# Non-Negative Matrix Factorization Model

nmf = NMF(n_components=NUM_TOPICS)

data_nmf = nmf.fit_transform(data_vectorized)
# Latent Semantic Indexing Model using Truncated SVD

lsi = TruncatedSVD(n_components=NUM_TOPICS)

data_lsi = lsi.fit_transform(data_vectorized)
# Functions for printing keywords for each topic

def selected_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]])
# Keywords for topics clustered by Latent Dirichlet Allocation

print(Back.RED + "LDA Model:")

selected_topics(lda, vectorizer)
# Keywords for topics clustered by Latent Semantic Indexing

print(Back.BLUE + "NMF Model:")

selected_topics(nmf, vectorizer)
# Keywords for topics clustered by Non-Negative Matrix Factorization

print(Back.MAGENTA + "LSI Model:")

selected_topics(lsi, vectorizer)
# Transforming an individual sentence

text = spacy_tokenizer("Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.")

x = lda.transform(vectorizer.transform([text]))[0]

print(x)
# Description and variety of grapes

df = df[['description', 'variety']]

df.head()
# Getting top 8 most described variety

temp_df = df.variety.value_counts()

temp_df.head(8)
# For this project we are taking top 8 variety only

mask = df['variety'].isin(['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend',

                           'Bordeaux-style Red Blend', 'Riesling', 'Sauvignon Blanc',

                           'Syrah'])

df = df[mask]

df.head()
chart = sns.countplot(df.variety, color='darkred')

plt.title("Number of descriptions per Variety")

chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.description, df.variety, test_size = 0.2, random_state = 42)
train = { 'text': X_train, 'intent': y_train }

train_df = pd.DataFrame(train)

test = { 'text': X_test, 'intent': y_test }

test_df = pd.DataFrame(test)
train_df.head()
# Making the training dataset uniform - taking the wine with least number of count (i.e. Syrah)

syrah_df = train_df[train_df['intent']=='Syrah']

# Selecting other varities of wines

riesling_df = train_df[train_df['intent']=='Riesling']

pinot_noir_df = train_df[train_df['intent']=='Pinot Noir']

chardonnay_df = train_df[train_df['intent']=='Chardonnay']

cabernet_sauvignon_df = train_df[train_df['intent']=='Cabernet Sauvignon']

red_blend_df = train_df[train_df['intent']=='Red Blend']

bordeaux_style_red_blend_df = train_df[train_df['intent']=='Bordeaux-style Red Blend']

sauvignon_blanc_df = train_df[train_df['intent']=='Sauvignon Blanc']
# Setting their count equal to that of Syrah

pinot_noir_df = pinot_noir_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)

chardonnay_df = chardonnay_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)

cabernet_sauvignon_df = cabernet_sauvignon_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)

red_blend_df = red_blend_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)

bordeaux_style_red_blend_df = bordeaux_style_red_blend_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)

riesling_df = riesling_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)

sauvignon_blanc_df = sauvignon_blanc_df.sample(n=len(syrah_df), random_state=RANDOM_SEED)
# Adding all the data together

syrah_df = syrah_df.append(pinot_noir_df).reset_index(drop=True)

syrah_df = syrah_df.append(chardonnay_df).reset_index(drop=True)

syrah_df = syrah_df.append(cabernet_sauvignon_df).reset_index(drop=True)

syrah_df = syrah_df.append(red_blend_df).reset_index(drop=True)

syrah_df = syrah_df.append(bordeaux_style_red_blend_df).reset_index(drop=True)

syrah_df = syrah_df.append(riesling_df).reset_index(drop=True)

syrah_df = syrah_df.append(sauvignon_blanc_df).reset_index(drop=True)

train_df = syrah_df

train_df.shape
chart = sns.countplot(train_df.intent, color='darkred')

plt.title("Number of descriptions per Variety (Resampled)")

chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');
# Shuffling the data

train_df = train_df.sample(frac=1).reset_index(drop=True)
!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
os.makedirs("model", exist_ok=True)
!mv uncased_L-12_H-768_A-12/ model
bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = os.path.join("model/", bert_model_name)

bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")

bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
class IntentDetectionData:

    DATA_COLUMN = "text"

    LABEL_COLUMN = "intent"

    def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):

        self.tokenizer = tokenizer

        self.max_seq_len = 0

        self.classes = classes

        train, test = map(lambda df: df.reindex(df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index), [train, test])

        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)

        self.max_seq_len = min(self.max_seq_len, max_seq_len)

        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

    def _prepare(self, df):

        x, y = [], []

        for _, row in tqdm(df.iterrows()):

            text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]

            tokens = self.tokenizer.tokenize(text)

            tokens = ["[CLS]"] + tokens + ["[SEP]"]

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            self.max_seq_len = max(self.max_seq_len, len(token_ids))

            x.append(token_ids)

            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):

        x = []

        for input_ids in ids:

            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]

            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))

            x.append(np.array(input_ids))

        return np.array(x)
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
tokenizer.tokenize("I like red wine more than white wine")
tokens = tokenizer.tokenize("Wines from some countries are very underrated!")

tokenizer.convert_tokens_to_ids(tokens)
def create_model(max_seq_len, bert_ckpt_file):

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:

        bc = StockBertConfig.from_json_string(reader.read())

        bert_params = map_stock_config_to_params(bc)

        bert_params.adapter_size = None

        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")

    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)

    cls_out = keras.layers.Dropout(0.5)(cls_out)

    logits = keras.layers.Dense(units=768, activation="swish")(cls_out)

    logits = keras.layers.Dropout(0.5)(logits)

    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)

    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    return model
test_df.head()
classes = train_df.intent.unique().tolist()

data = IntentDetectionData(train_df, test_df, tokenizer, classes, max_seq_len=128)
data.train_x.shape
data.train_x[0]
data.train_y[0]
data.max_seq_len
model = create_model(data.max_seq_len, bert_ckpt_file)
model.summary()
model.compile(

  optimizer=keras.optimizers.Adam(1e-5),

  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]

)
log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(

  x=data.train_x, 

  y=data.train_y,

  validation_split=0.1,

  batch_size=16,

  shuffle=True,

  epochs=10,

  callbacks=[tensorboard_callback]

)
ax = plt.figure().gca()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])

ax.plot(history.history['val_loss'])

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train', 'test'])

plt.title('Loss over training epochs')
ax = plt.figure().gca()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])

ax.plot(history.history['val_acc'])

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train', 'test'])

plt.title('Accuracy over training epochs')
_, train_acc = model.evaluate(data.train_x, data.train_y)

_, test_acc = model.evaluate(data.test_x, data.test_y)

print(Fore.RED + "train acc", train_acc)

print(Fore.BLUE + "test acc", test_acc)
y_pred = model.predict(data.test_x).argmax(axis=-1)
print(classification_report(data.test_y, y_pred, target_names=classes))
cm = confusion_matrix(data.test_y, y_pred)

df_cm = pd.DataFrame(cm, index=classes, columns=classes)
hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='hot')

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label');
sentences = [

  "Strong wine made of red grapes",

  "Grapy plummy and juicy taste"

]

pred_tokens = map(tokenizer.tokenize, sentences)

pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)

pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)

pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):

    print("text:", text, "\nintent:", classes[label])

    print()