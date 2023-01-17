# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

#import cufflinks and offline mode

import cufflinks as cf

cf.go_offline()



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



# Venn diagram

from matplotlib_venn import venn2

import re

import nltk

from nltk.corpus import stopwords

import string

import gc



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

train = pd.read_csv("../input/google-quest-challenge/train.csv")
print(train.shape, test.shape)
train.head(4)
test.head()
sample_submission.head()
sample_submission.shape
sample_submission.columns[1:]
targets = list(sample_submission.columns[1:])

fig, axes = plt.subplots(6, 5, figsize=(18, 15))

axes = axes.ravel()

bins = np.linspace(0, 1, 20)



for i, col in enumerate(targets):

    ax = axes[i]

    sns.distplot(train[col], label=col, kde=False, bins=bins, ax=ax)

    # ax.set_title(col)

    ax.set_xlim([0, 1])

    ax.set_ylim([0, 6079])

plt.tight_layout()

plt.show()

plt.close()
#https://www.kaggle.com/urvishp80/quest-encoding-ensemble: a thanks 

#

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"couldnt" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"doesnt" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"havent" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"shouldnt" : "should not",

"that's" : "that is",

"thats" : "that is",

"there's" : "there is",

"theres" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"theyre":  "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}





def clean_text(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = text.lower().split()

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops]    

    text = " ".join(text)

    return(text)



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)



def clean_data(df, columns: list):

    for col in columns:

        df[col] = df[col].apply(lambda x: clean_text(x.lower()))

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))



    return df
columns = ['question_title','question_body','answer', 'category']

train = clean_data(train, columns)

test = clean_data(test, columns)

print("Data cleaning Done........")
colText = ['question_title', 'question_body', 'answer', 'category']
train['text'] = train['question_body'] + " "+ "answer:" +" " + train['answer']
test['text'] = test['question_body'] + " "+ "answer:" " "+ + test['answer']
training_sentences = train['text'].values

testing_sentences = test['text'].values
training_sentences[4]
# len max of text in the training_sentences

max(train['text'].apply(lambda x: len(x.split())))
vocab_size = 7000

embedding_dim = 16

max_length = 4144

trunc_type='post'

oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
training_labels = train[sample_submission.columns[1:]].values

training_labels
training_labels.shape
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100, activation='relu'),

    tf.keras.layers.Dense(100, activation='relu'),

    tf.keras.layers.Dense(30, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

model.fit(padded, training_labels, validation_split=0.33, epochs=num_epochs, batch_size=50)
predictions = model.predict(testing_padded)
predictions.shape
output = pd.DataFrame(predictions, columns = sample_submission.columns[1:])
output.index = test['qa_id']
output.head()
output.reset_index(inplace = True)

output.head()
output.to_csv('submission.csv', index=False)