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
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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
train1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
train1.head(20)
train1.drop(['id','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
train1.head(20)
train1['toxic'].value_counts()
train1.shape
train1 = train1.loc[:100000, :]
train1.shape
train1['toxic'].value_counts()
train1['comment_text'][0]
train1['comment_text'][1]
sns.countplot(train1['toxic'])
text = train1['comment_text'].loc[train1['toxic'] == 0]
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

plt.figure(figsize=(10,10))
plt.title('Common words in non toxic sentences')
plt.imshow(WordCloud(stopwords=stopwords).generate(str(text)))
text = train1['comment_text'].loc[train1['toxic'] == 1]
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

plt.figure(figsize=(10,10))
plt.title('Common words in toxic sentences')
plt.imshow(WordCloud(stopwords=stopwords).generate(str(text)))
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stopword = stopwords.words('english')
train1['comment_text_vector'] = train1['comment_text']
train1['comment_text_vector'] = train1['comment_text_vector'].apply(lambda x: re.sub('[^a-zA-Z]',' ',x))
train1['comment_text_vector'] = train1['comment_text_vector'].apply(lambda x: x.lower())
train1['comment_text_vector'] = train1['comment_text_vector'].apply(lambda x: x.split())
train1['comment_text_vector'] = train1['comment_text_vector'].apply(lambda x: [item for item in x if item not in stopword])

train1['comment_text_vector'][0]
lemmatizer = nltk.stem.WordNetLemmatizer()
train1['comment_text_vector'] = train1['comment_text_vector'].apply(lambda x: [lemmatizer.lemmatize(item, 'v') for item in x ])
train1['comment_text_vector'][0]
from tensorflow.keras.preprocessing.text import one_hot
train1['comment_text_vector'] = train1['comment_text_vector'].apply(lambda x: [one_hot(item, 10000) for item in x] )
train1['comment_text_vector'][0]
from nltk import flatten
X = train1['comment_text_vector'].apply(lambda x: flatten(x) )
X
from tensorflow.keras.preprocessing.sequence import pad_sequences

sent_length = 1650
X = pad_sequences(X, padding='pre', maxlen=sent_length)


X
from sklearn.model_selection import train_test_split
Y = np.asarray(train1['toxic'])
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=1)
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential

features = 40
vocab_size = 10000
with strategy.scope():
    model = Sequential()
    model.add(Embedding(vocab_size, features, input_length=1650 ))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    print(model.summary())
          
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=100*strategy.num_replicas_in_sync )
from sklearn import metrics
def roc_auc(predictions,target):
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
score = model.predict(X_test)

print("Auc:  %.2f%%" % (roc_auc(score[:30001],y_test)))