import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pickle

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [18, 8]
df = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

print("DataFrame shape:", df.shape)
df.head()
print("Clothing ID nunique: {}".format(df['Clothing ID'].nunique()))
sns.distplot(df['Age']).set_title(
                            'Age Distribution', fontsize=20, weight='bold');
division_count = df['Division Name'].value_counts()

pie1 = go.Pie(labels=division_count.index,
              values=division_count.values,
              hole=0.5)

layout1 = go.Layout(title='Division Name', font=dict(size=18), legend=dict(orientation='h'))

fig1 = go.Figure(data=[pie1], layout=layout1)
py.iplot(fig1)
departement_count = df['Department Name'].value_counts()

sns.barplot(x=departement_count.values,
            y=departement_count.index,
            palette='magma').set_title('Departemen Name', fontsize=20);
sns.heatmap(pd.crosstab(df['Class Name'], df['Division Name']),
            annot=True, linewidths=.5, fmt='g', cmap='Reds',
            cbar=False);
splot1 = sns.countplot(df['Recommended IND'])

for p in splot1.patches:
    splot1.annotate(format(p.get_height() / df.shape[0] * 100, '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    rotation=0, ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

plt.xlabel(None)
plt.title('Recommend or not? (in percentage)')
plt.grid(False)
plt.show()
rating_count = df.Rating.value_counts()

sns.barplot(y=rating_count.values,
            x=rating_count.index, palette='Set1').set_title('Rating', fontsize=20);
rating_4_and_5 = round(df[(df['Rating'] == 5) | (df['Rating'] == 4)].shape[0] / df.shape[0] * 100, 2)
rating_1_and_2 = round(df[(df['Rating'] == 1) | (df['Rating'] == 2)].shape[0] / df.shape[0] * 100, 2)

print("There are {}% users who give 4 or 5 stars rating".format(rating_4_and_5))
print('')
print("There are {}% users who give 1 or 2 stars rating".format(rating_1_and_2))
df[(df['Rating'] == 5) | (df['Rating'] == 4)]['Recommended IND'].value_counts()
df[(df['Rating'] == 1) | (df['Rating'] == 2)]['Recommended IND'].value_counts()
departement_recommended = df.groupby('Department Name')['Recommended IND'].value_counts(normalize=True).rename('Percentage').mul(100).round(2).reset_index()

sns.barplot(x='Department Name', y='Percentage', hue='Recommended IND', data=departement_recommended);
departement_ratings = df.groupby('Department Name')['Rating'].value_counts(normalize=True).rename('Rating Percentage').mul(100).round(2).reset_index()

sns.barplot(x='Department Name', y='Rating Percentage', hue='Rating', data=departement_ratings, palette='Set1');
pd.set_option('max_colwidth', 300)
df.dropna(subset=['Review Text'], inplace=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

df['Polarity Score'] = df['Review Text'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])  # compound (aggregated score)
def sentiment_analyst(polarity_score):
    if polarity_score >= 0.05:
        return 'Positive'
    elif polarity_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Polarity Score'].apply(sentiment_analyst)
sentiment_recommended = df.groupby('Sentiment')['Recommended IND'].value_counts(normalize=True).rename('Recommend Percentage').mul(100).round(2).reset_index()

sns.barplot(x='Sentiment', y='Recommend Percentage', hue='Recommended IND', data=sentiment_recommended);
df[(df['Sentiment'] == 'Negative') & (df['Recommended IND'] == 1)]['Rating'].value_counts()
def clean_text(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'\n', ' ', tweet)
    tweet = re.sub(r'bc', 'because', tweet)
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"can't", "cannot", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"aren't", "are not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"haven't", "have not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"you're", "you are", tweet)
    tweet = re.sub(r"i'm", "i am", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"here's", "here is", tweet)
    tweet = re.sub(r"you've", "you have", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"y'all", "you all", tweet)
    tweet = re.sub(r"would've", "would have", tweet)
    tweet = re.sub(r"it'll", "it will", tweet)
    tweet = re.sub(r"we'll", "we will", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"didn't", "did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"they've", "they have", tweet)
    tweet = re.sub(r"i'd", "i would", tweet)
    tweet = re.sub(r"should've", "should have", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "i will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"let's", "let us", tweet)
    
    tokenizer = tweet.split()
    words = [word for word in tokenizer if word.isalpha()]
    return ' '.join(words)

df['Review Text'] = df['Review Text'].apply(clean_text)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print('Using Tensorflow version:', tf.__version__)
sns.distplot(df['Review Text'].apply(len));
print("Max Len:", max(map(len, df['Review Text'])))
print("I'll use 200 for max padding length")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Review Text'])

vocab_size = len(tokenizer.word_index)
print('Vocabulary size:', vocab_size)
sequences = tokenizer.texts_to_sequences(df['Review Text'])
padded_seq = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
GLOVE_EMBEDDING_PATH = '/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
print('number unknown words (glove): ', len(unknown_words_glove))
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, GlobalMaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
# first model
model1 = Sequential()

embeddings = Embedding(vocab_size+1, 300, weights=[glove_matrix], input_length=200, trainable=False)

model1.add(embeddings)
model1.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
model1.add(GlobalMaxPooling1D())
model1.add(Dense(1, activation='sigmoid')) # remember to use sigmoid for 1/0, not SOFTMAX!!

model1.summary()
optimizer = Adam(0.001)

model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
checkpoint1 = ModelCheckpoint('weights1.h5', save_best_only=True, monitor='val_accuracy', mode='max')
from sklearn.model_selection import train_test_split

X = padded_seq
y = df['Recommended IND'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2020)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
history1 = model1.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val), callbacks=[checkpoint1], verbose=1)
# Get training and test loss histories
training_loss = history1.history['loss']
test_loss = history1.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization
import tensorflow_hub as hub
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
    out = Dense(1, activation='sigmoid')(clf_output)  # remember to use sigmoid for 1/0, not SOFTMAX..
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])  # 0.00002 - 0.00005
    
    return model
%%time
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
X_train_bert = bert_encode(df['Review Text'], tokenizer, max_len=200)

y_train = df['Recommended IND'].values
model2 = build_model(bert_layer, max_len=200)

model2.summary()
checkpoint2 = ModelCheckpoint('weights2.h5', monitor='val_loss', save_best_only=True)

history2 = model2.fit(
            X_train_bert, y_train,
            validation_split=0.2,
            epochs=3,
            batch_size=32, callbacks=[checkpoint2])
# Get training and test loss histories
training_loss2 = history2.history['loss']
test_loss2 = history2.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss2) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss2, 'r--')
plt.plot(epoch_count, test_loss2, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()