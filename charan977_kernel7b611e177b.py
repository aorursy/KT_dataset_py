# importing required libraries

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from keras.layers import Activation,Input,CuDNNLSTM,CuDNNGRU, Embedding, LSTM, Dropout, BatchNormalization, Dense, concatenate, Flatten, Conv1D, MaxPool1D, LeakyReLU, ELU, SpatialDropout1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.utils import to_categorical,plot_model

from keras.preprocessing.text import Tokenizer, one_hot

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, load_model,Sequential

from keras import regularizers

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from keras.initializers import he_normal,Orthogonal

from keras.regularizers import l2

from keras.constraints import max_norm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix,classification_report

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import re

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import pickle

import seaborn as sns

train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
train.columns[train.isna().any()].tolist()
print("Target column has", train['target'].isna().sum(), "missing values")

print("comment_text column has", train['comment_text'].isna().sum(), "missing values")
# see distribution of the comment_text length

train['comment_text'].str.len().hist()
test.columns[test.isna().any()].tolist()
print("comment_text column has",test['comment_text'].isna().sum(), "missing values")
test.comment_text.str.len().hist()
train.head()
data = train.sample(frac=0.5)

print("Mean value for the target column:  ", data["target"].mean())

print("Number of targets higher than 0.5: ", data[(data['target'] >0.5)]["target"].count())

print("Number of targets higher than 0.0: ", data[(data['target'] >0.0)]["target"].count())

print("Number of comments:                ", len(data))
data['label'] = np.where(data['target']>0,1,0)

label_list = list(data['label'].unique())

label_list

print("Number of targets higher than 0.0: ", data[(data['target'] >0.0)]["target"].count())
# this is a fraction of the sample fraction. Essentially we should always keep this between 0.7-0.9

training_frac = 0.8

train_len = int(len(data)*training_frac)

valid_len = int(len(data)*(1.0-training_frac))



train = data.iloc[:train_len, :]

valid = data.iloc[:valid_len, :]
train0 = train[train["label"]==0]

train1 = train[train["label"]==1]

train0["count"] = train0['comment_text'].str.split().str.len()

train1["count"] = train1['comment_text'].str.split().str.len()
import seaborn as sns

x = pd.Series(train0["count"])

ax = sns.distplot(x, kde=False)
import seaborn as sns

x = pd.Series(train1["count"])

ax = sns.distplot(x, kde=False)
total_comments = train.shape[0]

n_unique_comments = train['comment_text'].nunique()

n_comments_both = len(set(train['comment_text'].unique()) & set(test['comment_text'].unique()))

print('Train set: %d rows and %d columns.' % (total_comments, train.shape[1]))

display(train.head())

display(train.describe())


print('Test set: %d rows and %d columns.' % (test.shape[0], test.shape[1]))

display(test.head())

print('Number of unique comments: %d or %.2f%%'% (n_unique_comments, (n_unique_comments / total_comments * 100)))

print('Number of comments that are both in train and test sets: %d'% n_comments_both)
train['comment_length'] = train['comment_text'].apply(lambda x : len(x))

test['comment_length'] = test['comment_text'].apply(lambda x : len(x))

train['word_count'] = train['comment_text'].apply(lambda x : len(x.split(' ')))

test['word_count'] = test['comment_text'].apply(lambda x : len(x.split(' ')))

bin_size = max(train['comment_length'].max(), test['comment_length'].max())//10



plt.figure(figsize=(20, 6))

sns.distplot(train['comment_length'], bins=bin_size)

sns.distplot(test['comment_length'], bins=bin_size)

plt.show()
bin_size = max(train['word_count'].max(), test['word_count'].max())//10

plt.figure(figsize=(20, 6))

sns.distplot(train['word_count'], bins=bin_size)

sns.distplot(test['word_count'], bins=bin_size)

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

sns.distplot(train['target'], bins=20, ax=ax1).set_title("Complete set")

sns.distplot(train[train['target'] > 0]['target'].values, bins=20, ax=ax2).set_title("Only toxic comments")

plt.show()
train['is_toxic'] = train['target'].apply(lambda x : 1 if (x > 0.5) else 0)

plt.figure(figsize=(8, 6))

sns.countplot(train['is_toxic'])

plt.show()
# Lets also see how many missing values (in percentage) we are dealing with

miss_val_train_df = train.isnull().sum(axis=0) / train_len

miss_val_train_df = miss_val_train_df[miss_val_train_df > 0] * 100

miss_val_train_df
# lets create a list of all the identities tagged in this dataset. This list given in the data section of this competition. 

identities = ['male','female','transgender','other_gender','heterosexual','homosexual_gay_or_lesbian',

              'bisexual','other_sexual_orientation','christian','jewish','muslim','hindu','buddhist',

              'atheist','other_religion','black','white','asian','latino','other_race_or_ethnicity',

              'physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness',

              'other_disability']

# getting the dataframe with identities tagged

train_labeled = train.loc[:, ['target'] + identities ].dropna()

# lets define toxicity as a comment with a score being equal or .5

# in that case we divide it into two dataframe so we can count toxic vs non toxic comment per identity

toxic = train_labeled[train_labeled['target'] >= .5][identities]

non_toxic = train_labeled[train_labeled['target'] < .5][identities]
# at first, we just want to consider the identity tags in binary format. So if the tag is any value other than 0 we consider it as 1.

toxic_count = toxic.where(train_labeled == 0, other = 1).sum()

non_toxic_count = non_toxic.where(train_labeled == 0, other = 1).sum()
toxic_vs_non_toxic = pd.concat([toxic_count, non_toxic_count], axis=1)

toxic_vs_non_toxic = toxic_vs_non_toxic.rename(index=str, columns={1: "non-toxic", 0: "toxic"})

# here we plot the stacked graph but we sort it by toxic comments to (perhaps) see something interesting

toxic_vs_non_toxic.sort_values(by='toxic').plot(kind='bar', stacked=True, figsize=(30,10), fontsize=20).legend(prop={'size': 20})
# First we multiply each identity with the target

weighted_toxic = train_labeled.iloc[:, 1:].multiply(train_labeled.iloc[:, 0], axis="index").sum() 

# changing the value of identity to 1 or 0 only and get comment count per identity group

identity_label_count = train_labeled[identities].where(train_labeled == 0, other = 1).sum()

# then we divide the target weighted value by the number of time each identity appears

weighted_toxic = weighted_toxic / identity_label_count

weighted_toxic = weighted_toxic.sort_values(ascending=False)

# plot the data using seaborn like before

plt.figure(figsize=(30,20))

sns.set(font_scale=3)

ax = sns.barplot(x = weighted_toxic.values , y = weighted_toxic.index, alpha=0.8)

plt.ylabel('Demographics')

plt.xlabel('Weighted Toxicity')

plt.title('Weighted Analysis of Most Frequent Identities')

plt.show()
# lets take the dataset with identitiy tags, created date, and target column

with_date = train.loc[:, ['created_date', 'target'] + identities].dropna()

# next we will create a weighted dataframe for each identity tag (like we did before)

# first we divide each identity tag with the total value it has in the dataset

weighted = with_date.iloc[:, 2:] / with_date.iloc[:, 2:].sum()

# then we multiplty this value with the target 

target_weighted = weighted.multiply(with_date.iloc[:, 1], axis="index")

# lets add a column to count the number of comments

target_weighted['comment_count'] = 1

# now we add the date to our newly created dataframe (also parse the text date as datetime)

target_weighted['created_date'] = pd.to_datetime(with_date['created_date'].apply(lambda dt: dt[:10]))

# now we can do a group by of the created date to count the number of times a identity appears for that date

identity_weight_per_date = target_weighted.groupby(['created_date']).sum().sort_index()
# lets group most of the identities into three major categories as follows for simplified analysis

races = ['black','white','asian','latino','other_race_or_ethnicity']

religions = ['atheist', 'buddhist', 'christian', 'hindu', 'muslim', 'jewish','other_religion']

sexual_orientation = ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']
# lets create a column to aggregate our weighted toxicity score per identity group

identity_weight_per_date['races_total'] = identity_weight_per_date[races].sum(axis=1)

identity_weight_per_date['religions_total'] = identity_weight_per_date[religions].sum(axis=1)

identity_weight_per_date['sexual_orientation_total'] = identity_weight_per_date[sexual_orientation].sum(axis=1)

# and then plot a time-series line plot per identity group

identity_weight_per_date[['races_total', 'religions_total', 'sexual_orientation_total']].plot(figsize=(15,7), linewidth=1, fontsize=15) 

plt.legend(loc=2, prop={'size': 15})

plt.xlabel('Comment Date', fontsize=15)

plt.ylabel('Weighted Toxic Score', fontsize=15)
identity_weight_per_date['comment_count'].plot(figsize=(15,7), linewidth=1, fontsize=15)

plt.xlabel('Comment Date', fontsize = 15)

plt.ylabel('Total Comments', fontsize = 15)
# lets divide by the comment count for the date to get a relative weighted toxic score

identity_weight_per_date['races_rel'] = identity_weight_per_date['races_total'] / identity_weight_per_date['comment_count']

identity_weight_per_date['religions_rel'] = identity_weight_per_date['religions_total'] / identity_weight_per_date['comment_count']

identity_weight_per_date['sexual_orientation_rel'] = identity_weight_per_date['sexual_orientation_total']  / identity_weight_per_date['comment_count']

# now lets plot the data

identity_weight_per_date[['races_rel', 'religions_rel', 'sexual_orientation_rel']].plot(figsize=(15,7), linewidth=1, fontsize=20) 

plt.legend(loc=2, prop={'size': 15})

plt.xlabel('Comment Date', fontsize=15)

plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets plot relative weighted toxic score for each identity of races

identity_weight_per_date[races].div(identity_weight_per_date['comment_count'], axis=0).plot(figsize=(15,7), linewidth=1, fontsize=15)

plt.legend(loc=2, prop={'size': 15})

plt.xlabel('Comment Date', fontsize=15)

plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets plot relative weighted toxic score for each identity of religions

identity_weight_per_date[religions].div(identity_weight_per_date['comment_count'], axis=0).plot(figsize=(15,7), linewidth=1, fontsize=15)

plt.legend(loc=2, prop={'size': 15})

plt.xlabel('Comment Date', fontsize=15)

plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets plot relative weighted toxic score for each identity of sexual orientation

identity_weight_per_date[sexual_orientation].div(identity_weight_per_date['comment_count'], axis=0).plot(figsize=(15,7), linewidth=1, fontsize=20)

plt.legend(loc=2, prop={'size': 15})

plt.xlabel('Comment Date', fontsize=15)

plt.ylabel('Relative Weighted Toxic Score', fontsize=15)




# We clean the essay text data

# For this task, we have defined some helper functions

# The same function and code snippet will be used to clean project title

# https://stackoverflow.com/a/47091490/4084039



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



# https://gist.github.com/sebleier/554280



stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]

# Cleaning Text feature

preprocessed_text = []

# tqdm is for printing the status bar

for sentance in tqdm(train['comment_text'].values):

    sent = decontracted(sentance)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split() if e not in stopwords)

    preprocessed_text.append(sent.lower().strip())

train["clean_text"] = preprocessed_text
# Combining all the above statemennts 

preprocessed_comments_test = []

# tqdm is for printing the status bar

for sentence in tqdm(test['comment_text'].values):

    sent = decontracted(sentence)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split() if e not in stopwords)

    preprocessed_comments_test.append(sent.lower().strip())
test['comment_text'] = preprocessed_comments_test
train_len = len(train.index)
miss_val_train_df = train.isnull().sum(axis=0) / train_len

miss_val_train_df = miss_val_train_df[miss_val_train_df > 0] * 100

miss_val_train_df
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
for column in identity_columns + ['target']:

    train[column] = np.where(train[column] >= 0.5, True, False)
y = train['target'].values
# We split our dataset into train,cross-validation and test set

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train,y,test_size=0.30,random_state=43)



print(x_train.shape)

print(x_valid.shape)
MAX_VOCAB_SIZE = 100000

TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'

MAX_SEQUENCE_LENGTH = 300



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(train[TEXT_COLUMN])



# All comments must be truncated or padded to be the same length.

def padding_text(texts, tokenizer):

    return sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

train_text = padding_text(x_train[TEXT_COLUMN], tokenizer)

train_y = to_categorical(x_train[TOXICITY_COLUMN])

validate_text = padding_text(x_valid[TEXT_COLUMN], tokenizer)

validate_y = to_categorical(x_valid[TOXICITY_COLUMN])

# for submission purpose

test_text = padding_text(test[TEXT_COLUMN], tokenizer)
import os

print(os.listdir("../input"))

import os

print(os.listdir("../input/fasttext-crawl-300d-2m"))
embeddings_index = {}

with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec' ,encoding='utf8') as f:

  for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

len(tokenizer.word_index)
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,300))

num_words_in_embedding = 0

for word, i in tokenizer.word_index.items():

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:

    num_words_in_embedding += 1

    # words not found in embedding index will be all-zeros.

    embedding_matrix[i] = embedding_vector
embedding_matrix.shape
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import pickle

from tqdm import tqdm

from wordcloud import WordCloud





from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn import metrics

from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix,classification_report

%matplotlib inline

warnings.filterwarnings("ignore")

from IPython.display import Image,YouTubeVideo,HTML

from keras.models import Sequential, Model

from keras.utils import to_categorical,plot_model

from keras.layers import Dense, Activation

from keras.layers.normalization import BatchNormalization

from keras.initializers import he_normal

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.layers import Dropout

from keras.layers import Embedding,CuDNNLSTM,CuDNNGRU, Flatten, Input, concatenate, Conv1D, GlobalMaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D, Bidirectional

from keras.regularizers import l2

from keras.optimizers import Adam

from keras.initializers import Orthogonal

from keras.preprocessing.text import one_hot

from keras.constraints import max_norm
input_text_bgru = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')

embedding_layer_bgru = Embedding(len(tokenizer.word_index) + 1,

                                    300,

                                    weights=[embedding_matrix],

                                    input_length=MAX_SEQUENCE_LENGTH,

                                    trainable=False)

g = embedding_layer_bgru(input_text_bgru)

g = SpatialDropout1D(0.4)(g)

g = Bidirectional(CuDNNGRU(64, return_sequences=True))(g)

g = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(g)

avg_pool = GlobalAveragePooling1D()(g)

max_pool = GlobalMaxPooling1D()(g)

g = concatenate([avg_pool, max_pool])

g = Dense(128, activation='relu')(g)

bgru_output = Dense(2, activation='softmax')(g)

model = Model(inputs=[input_text_bgru], outputs=[bgru_output])
from sklearn.metrics import roc_auc_score

def auc1(y_true, y_pred):

    if len(np.unique(y_true[:,1])) == 1:

        return 0.5

    else:

        return roc_auc_score(y_true, y_pred)



def auroc(y_true, y_pred):

    return tf.py_func(auc1, (y_true, y_pred), tf.double)



model.compile(loss='categorical_crossentropy',

                  optimizer='adam',

                  metrics=[auroc])
model.summary()
history = model.fit(train_text,train_y,

              batch_size=1024,

              epochs=10,

              validation_data=(validate_text, validate_y))
plt.plot(history.history['auroc'], 'g')

plt.plot(history.history['val_auroc'], 'r')

plt.legend({'Train ROCAUC': 'g', 'Test ROCAUC':'r'})

plt.show()





plt.plot(history.history['loss'], 'g')

plt.plot(history.history['val_loss'], 'r')

plt.legend({'Train Loss': 'g', 'Test Loss':'r'})

plt.show()
predictions = model.predict(test_text)[:, 1]
submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

submit.prediction = predictions

submit.to_csv('submission.csv', index=False)