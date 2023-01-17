import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install tensorflow==1.14.0
!pip install pyspellchecker
!pip install pandas-profiling --ignore-installed
# to hide warnings
import warnings
warnings.filterwarnings('ignore')

# basic data processing
import os
import datetime
import pandas as pd
import numpy as np

# for EDA
from pandas_profiling import ProfileReport

# for text preprocessing
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from spellchecker import SpellChecker

# progress bar
from tqdm.auto import tqdm
from tqdm import tqdm_notebook

# instantiate
tqdm.pandas(tqdm_notebook)

# for wordcloud
from PIL import Image
from wordcloud import WordCloud

# for aesthetics and plots
from IPython.display import display, Markdown, clear_output
from termcolor import colored

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.offline import plot, iplot
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook"

# for model
import tensorflow as tf
import tensorflow_hub as hub
import keras.layers as layers
from keras.models import Model
from keras import backend as K
import keras
from keras.models import load_model

display(Markdown('_All libraries are imported successfully!_'))
col_names =  ['target', 'id', 'date', 'flag','user','text']

df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", names=col_names)

print(colored('DATA','blue',attrs=['bold']))
display(df.head())
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title='Pandas Profiling Report')
profile
# dropping irrelevant columns
df.drop(['id', 'date', 'flag', 'user'], axis=1, inplace=True)

# replacing positive sentiment 4 with 1
df.target = df.target.replace(4,1)

target_count = df.target.value_counts()

category_counts = len(target_count)
display(Markdown('__Number of categories__: {}'.format(category_counts)))
# set of stop words declared
stop_words = stopwords.words('english')

display(Markdown('__List of stop words__:'))
display(Markdown(str(stop_words)))
updated_stop_words = stop_words.copy()
for word in stop_words:
    if "n't" in word or "no" in word or word.endswith('dn') or word.endswith('sn') or word.endswith('tn'):
        updated_stop_words.remove(word)

# custom select words you don't want to eliminate
words_to_remove = ['for','by','with','against','shan','don','aren','haven','weren','until','ain','but','off','out']
for word in words_to_remove:
    updated_stop_words.remove(word)

display(Markdown('__Updated list of stop words__:'))
display(Markdown(str(updated_stop_words)))
# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Defining regex patterns.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# creating instance of spellchecker
spell = SpellChecker()

# creating instance of lemmatizer
lemm = WordNetLemmatizer()


def preprocess(tweet):
    # lowercase the tweets
    tweet = tweet.lower().strip()
    
    # REMOVE all URls
    tweet = re.sub(urlPattern,'',tweet)
    
    # Replace all emojis.
    for emoji in emojis.keys():
        tweet = tweet.replace(emoji, "emoji" + emojis[emoji])        
    
    # Remove @USERNAME
    tweet = re.sub(userPattern,'', tweet)        
    
    # Replace all non alphabets.
    tweet = re.sub(alphaPattern, " ", tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    splitted_tweet = tweet.split()
    # spell checks
#     misspelled = spell.unknown(splitted_tweet)
#     if misspelled == set():
#         pass
#     else:
#         for i,word in enumerate(misspelled):
#             splitted_tweet[i] = spell.correction(word)

    tweetwords = ''
    for word in splitted_tweet:
        # Checking if the word is a stopword.
        if word not in updated_stop_words:
            if len(word)>1:
                # Lemmatizing the word.
                lem_word = lemm.lemmatize(word)
                tweetwords += (lem_word+' ')
    
    return tweetwords
df['text'] = df['text'].progress_apply(lambda x: preprocess(x))
print(colored('DATA','blue',attrs=['bold']))
display(df.head())
def plot_wordcloud(text, mask, title = None):
    wordcloud = WordCloud(background_color='black', max_words = 200,
                          max_font_size = 200, random_state = 42, mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=(25,25))
    
    plt.imshow(wordcloud)
    plt.title(title, fontdict={'size': 40, 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()
pos_text = " ".join(df[df['target'] == 1]['text'])
pos_mask = np.array(Image.open('/kaggle/input/sentiment140-vis-img/source/upvote.png'))

plot_wordcloud(pos_text, pos_mask, title = 'Most common 200 words in positive tweets')
neg_text = " ".join(df[df['target'] == 0].text)
neg_mask = np.array(Image.open('/kaggle/input/sentiment140-vis-img/source/downvote.png'))

plot_wordcloud(neg_text, neg_mask, title = 'Most common 200 words in negative tweets')
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=369):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

train_df, val_df, test_df = train_validate_test_split(df)

print('Train: {}, Validation: {}, Test: {}'.format(train_df.shape, val_df.shape, test_df.shape))

print(colored('\nTRAIN DATA','magenta',attrs=['bold']))
display(train_df.head())

train_text = train_df['text'].tolist()
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = np.asarray(pd.get_dummies(train_df['target']), dtype = np.int8)

val_text = val_df['text'].tolist()
val_text = np.array(val_text, dtype=object)[:, np.newaxis]
val_label = np.asarray(pd.get_dummies(val_df['target']), dtype = np.int8)

test_text = test_df['text'].tolist()
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = np.asarray(pd.get_dummies(test_df['target']), dtype = np.int8)
# we can change this model. check the url 'https://tfhub.dev/google/' for more
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value
print("Embedding size: ",embed_size)
# Compute a representation for each message, showing various lengths supported.
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = ("Universal Sentence Encoder embeddings also support short paragraphs. "
             "There is no hard limit on how long the paragraph is. Roughly, the longer the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embed(messages))

    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        print("Message: {}".format(messages[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
        print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
                 signature="default", as_dict=True)["default"]

input_text = layers.Input(shape=(1,), dtype="string")
embedding = layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)

# experiment on the custom FC layer here
#------------------------------------------------------#
x = layers.Dense(256, activation='relu')(embedding)
x = layers.Dropout(0.25)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.125)(x)
x = layers.Dense(category_counts, activation='sigmoid')(x)
#------------------------------------------------------#

model_sa = Model(inputs=[input_text], outputs=x)

# we are selecting Adam optimizer - one of the best optimizer in this field
opt = keras.optimizers.Adam(learning_rate=0.001)

# setting `binary_crossentropy` as loss function for the classifier
model_sa.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model_sa.summary()
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model_sa.fit(train_text, train_label,
                            validation_data=(val_text, val_label),
                            epochs=10,
                            batch_size=64,
                            shuffle=True)
    model_sa.save_weights('best_model.h5')
# load the saved model
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model_sa.load_weights('best_model.h5')
    _, train_acc = model_sa.evaluate(train_text, train_label)
    _, test_acc = model_sa.evaluate(test_text, test_label)

clear_output()
display(Markdown('__Train Accuracy__: {}, __Test Accuracy__: {}'.format(round(train_acc,4), round(test_acc,4))))
fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=list(range(50)), y=history.history['accuracy'], name='train'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=list(range(50)), y=history.history['val_accuracy'], name='validation'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=list(range(50)), y=history.history['loss'], name='train'),
              row=1, col=2)
fig.add_trace(go.Scatter(x=list(range(50)), y=history.history['val_loss'], name='validation'),
              row=1, col=2)

fig.update_layout(height=600, width=900, showlegend=False,hovermode="x",
                  title_text="Train and Validation Accuracy and Loss")
fig.show()
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model_sa.load_weights('best_model.h5')
    predicts = model_sa.predict(test_text, batch_size=32)

categories = train_df['target'].unique().tolist()

predict_logits = predicts.argmax(axis=1)
test_df['predicted'] = [categories[i] for i in predict_logits]

def highlight_rows(x):
    if x['target'] != x['predicted']:
        return ['background-color: #d65f5f']*3
    else:
        return ['background-color: lightgreen']*3

clear_output()
display(test_df.head(20).style.apply(highlight_rows, axis=1))