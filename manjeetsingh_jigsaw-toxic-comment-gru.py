# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU,SimpleRNN

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
test
train["toxic"].value_counts()
import re

import string

def clean_text_round1(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('[‘’“”…]', '', text)

    text = text.lower()

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    return text



round1 = lambda x: clean_text_round1(x)



train['comment_text'] = pd.DataFrame(train['comment_text'].apply(round1))

validation['comment_text'] = pd.DataFrame(validation['comment_text'].apply(round1))

test['content'] = pd.DataFrame(test['content'].apply(round1))





train.head()
Negative_sentiments = " ".join([text for text in train['comment_text'][train['toxic'] == 1]])



from wordcloud import WordCloud

from wordcloud import STOPWORDS



stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'lightgreen', stopwords = stopwords, width = 1200, height = 800).generate(Negative_sentiments)



plt.rcParams['figure.figsize'] = (15, 15)

plt.title('Most Common Negative Toxic Words', fontsize = 30)

print(wordcloud)

plt.axis('off')

plt.imshow(wordcloud)

plt.show()
Positive_sentiments = " ".join([text for text in train['comment_text'][train['toxic'] == 0]])



from wordcloud import WordCloud

from wordcloud import STOPWORDS



stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'pink', stopwords = stopwords, width = 1200, height = 800).generate(Positive_sentiments)



plt.rcParams['figure.figsize'] = (15, 15)

plt.title('Most Common Positive Words', fontsize = 30)

print(wordcloud)

plt.axis('off')

plt.imshow(wordcloud)

plt.show()
Negative_sentiments = " ".join([text for text in train['comment_text'][train['severe_toxic'] == 1]])



from wordcloud import WordCloud

from wordcloud import STOPWORDS



stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'cyan', stopwords = stopwords, width = 1200, height = 800).generate(Negative_sentiments)



plt.rcParams['figure.figsize'] = (15, 15)

plt.title('Most Common Negative severe toxic words', fontsize = 30)

print(wordcloud)

plt.axis('off')

plt.imshow(wordcloud)

plt.show()
Positive_sentiments = " ".join([text for text in train['comment_text'][train['severe_toxic'] == 0]])



from wordcloud import WordCloud

from wordcloud import STOPWORDS



stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow', stopwords = stopwords, width = 1200, height = 800).generate(Positive_sentiments)



plt.rcParams['figure.figsize'] = (15, 15)

plt.title('Most Common Positive Words', fontsize = 30)

print(wordcloud)

plt.axis('off')

plt.imshow(wordcloud)

plt.show()
def new_len(x):

    if type(x) is str:

        return len(x.split())

    else:

        return 0



train["comment_words"] = train["comment_text"].apply(new_len)

nums = train.query("comment_words != 0 and comment_words < 200").sample(frac=0.1)["comment_words"]

fig = ff.create_distplot(hist_data=[nums],

                         group_labels=["All comments"],

                         colors=["black"])



fig.update_layout(title_text="Comment words", xaxis_title="Comment words", template="simple_white", showlegend=False)

fig.show()
train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)

train = train.loc[:12000,:]

train.shape
train['comment_text'].apply(lambda x:len(str(x).split())).max()
def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 

                                                  stratify=train.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
# using keras tokenizer here

token = text.Tokenizer(num_words=None)

max_len = 1500



token.fit_on_texts(list(xtrain) + list(xvalid))

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



#zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)



word_index = token.word_index
# create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
%%time

with strategy.scope():

    # GRU with glove embeddings and two dense layers

     model = Sequential()

     model.add(Embedding(len(word_index) + 1,

                     300,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

     model.add(SpatialDropout1D(0.3))

     model.add(GRU(300))

     model.add(Dense(1, activation='sigmoid'))



     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   

    

model.summary()
model.fit(xtrain_pad, ytrain, batch_size=64*strategy.num_replicas_in_sync)
scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
scores_model.append({'Model': 'Bi-directional LSTM','AUC_Score': roc_auc(scores,yvalid)})