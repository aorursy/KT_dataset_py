!pip install wordcloud
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

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

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D, Input

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import seaborn as sns

from pandas_profiling import ProfileReport



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip') 

test_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

labels_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')



print('The Training data has {} rows and {} columns '.format(train_df.shape[0], train_df.shape[1]))
train_df.head()
test_df.head()
labels_df.head()
ProfileReport(train_df)
labels = train_df['toxic'].value_counts().index

values = train_df['toxic'].value_counts().values

color = ['green', 'lightblue']



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values, marker=dict(colors=color))])

fig.show()
train_df.info()
plt.figure(figsize=(10,5))

colormap = plt.cm.plasma

sns.heatmap(train_df.corr(), annot=True, cmap=colormap)
train_df['comment_text'][0]
train_df['comment_text'][2]
sns.countplot(train_df['toxic'])
sns.countplot(y=train_df['obscene'])
texts = train_df['comment_text'][0]

wordcloud = WordCloud().generate(texts)



# Display the generated image:

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
texts = train_df['comment_text'][2]

wordcloud = WordCloud().generate(texts)



# Display the generated image:

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
def roc_auc(predictions, target):

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc
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
train_df['char_length'] = train_df['comment_text'].apply(lambda x : len(str(x)))
test_df['char_length'] = test_df['comment_text'].apply(lambda x : len(str(x)))
import re



def clean_text(texts):

    texts = texts.lower()

    texts = re.sub(r"what's", "what is ", texts)

    texts = re.sub(r"\'s", " ", texts)

    texts = re.sub(r"\'ve", " have ", texts)

    texts = re.sub(r"can't", "cannot ", texts)

    texts = re.sub(r"n't", " not ", texts)

    texts = re.sub(r"i'm", "i am ", texts)

    texts = re.sub(r"\'re", " are ", texts)

    texts = re.sub(r"\'d", " would ", texts)

    texts = re.sub(r"\'ll", " will ", texts)

    texts = re.sub(r"\'scuse", " excuse ", texts)

    texts = re.sub('\W', ' ', texts)

    texts = re.sub('\s+', ' ', texts)

    texts = texts.strip(' ')

    return texts
# clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]

train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))

# clean the comment_text in test_df [Thanks, Pulkit Jha.]

test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))
train_df = train_df.drop('char_length',axis=1)

x = train_df.comment_text

x_test = test_df.comment_text
x.shape
x_test.shape
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer





vect = TfidfVectorizer(max_features=5000,stop_words='english')

vect
x_train = vect.fit_transform(x)
x_test = vect.transform(x_test)
cols = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



rfc = RandomForestClassifier()



submission_binary = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')



#to get predictions specific to each columns in the dataset, binary relevance based approach

for labels in cols:

    print('Started with {}'.format(labels))

    y = train_df[labels]

    rfc.fit(x_train,y)

    

    y_preds = rfc.predict(x_train)

    print('Validation accuracy is {}'.format(accuracy_score(y, y_preds)))

    # compute the predicted probabilities for x_test

    y_prob = rfc.predict_proba(x_test)[:,1]

    submission_binary[labels] = y_prob
rc = roc_auc(y_preds, y)

rc
cf = classification_report(y_preds, y)

print('The Classification Report {} \n '.format(cf))
submission_binary.head()
submission_binary.to_csv('submission_binary',index=False)

print('Submission file is successfully created!!')