# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import required packages

#basics

import pandas as pd 

import numpy as np



#misc

import gc

import time

import warnings



#stats

from scipy.misc import imread

from scipy import sparse

import scipy.stats as ss



#viz

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn



#nlp

import string

import re    #for regex

import nltk

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer   





#FeatureEngineering

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.model_selection import train_test_split



import tensorflow as tf 





#settings

start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

eng_stopwords = set(stopwords.words("english"))

warnings.filterwarnings("ignore")



lem = WordNetLemmatizer()

tokenizer=TweetTokenizer()





import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

import scipy as sp

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import OneHotEncoder

## Common Variables for Notebook 

ROOT = '/kaggle/input/google-quest-challenge/'
## load the data 

train = pd.read_csv(ROOT+'train.csv')

test = pd.read_csv(ROOT+'test.csv')

sub = pd.read_csv(ROOT+'sample_submission.csv')
## Quick look at the train

train.head()
#Quick look at the test 

test.head()
## Quick look at the sample data 

sub.head()

## Get the shape of the data

train_len, test_len ,sub_len = len(train.index), len(test.index),len(sub.index)

print(f'train size: {train_len}, test size: {test_len} , sample size: {sub_len}')
## Count the missing values 

miss_val_train = train.isnull().sum(axis=0) / train_len

miss_val_train = miss_val_train[miss_val_train> 0] * 100

miss_val_train
## Number of train columns

len(list(train.columns))
## Check the scoring for questions

all_train_columns = list(train.columns)

question_answer_cols = all_train_columns[:11]

question_target_cols = all_train_columns[11:32]

answer_target_cols  = all_train_columns[32:41]
## Check one question and answer  

questiont = train["question_title"][0]

questionb = train["question_body"][0]

answer1 = train["answer"][0]



print(f"The First Question Topic  is : {questiont}\n\n ")

print(f"The First Question Details are :  \n\n {questionb}\n\n ")

print(f"The First answer is :\n\n {answer1}\n\n ")
## Check target scoring for question



train[question_target_cols].loc[0]
## Check target scoring for answer

train[answer_target_cols].loc[0]
## How many distinct users asked more than 10 questions  ?

user_q_grp = train.question_user_name.value_counts()

user_q_grp.loc[user_q_grp>10].plot(kind='bar', figsize=(30,10), fontsize=10).legend(prop={'size': 20})
## How many distinct users have answered more than 10 questions?

user_a_grp = train.question_user_name.value_counts()

user_a_grp.loc[user_a_grp>10].plot(kind='bar', figsize=(30,10), fontsize=10).legend(prop={'size': 20})
##Lets see what kind of quesitons Mike asked

print( f'First Question Asked by Mike : \n\n {train.loc[train.question_user_name =="Mike"]["question_body"].values[1]}')
## Another question asked by Mike 

print( f'Second Question Asked by Mike : \n\n {train.loc[train.question_user_name =="Mike"]["question_body"].values[2]}')
## How Many Mike ?



train.loc[train.question_user_name =="Mike"]["question_user_page"].values
## What is the distribution of all question ranking columns 



train[question_target_cols]
## lets see some distributions of questions targets

plt.figure(figsize=(20, 5))



sns.distplot(train[question_target_cols[0]], hist= False , rug= False ,kde=True, label =question_target_cols[0],axlabel =False )

sns.distplot(train[question_target_cols[1]], hist= False , rug= False,label =question_target_cols[1],axlabel =False)

sns.distplot(train[question_target_cols[2]], hist= False , rug= False,label =question_target_cols[2],axlabel =False)

sns.distplot(train[question_target_cols[3]], hist= False , rug= False,label =question_target_cols[3],axlabel =False)

sns.distplot(train[question_target_cols[4]], hist= False , rug= False,label =question_target_cols[4],axlabel =False)

plt.show()
## lets see some distributions of answer targets

plt.figure(figsize=(20, 5))



sns.distplot(train[answer_target_cols[0]], hist= False , rug= False ,kde=True, label =answer_target_cols[0],axlabel =False )

sns.distplot(train[answer_target_cols[1]], hist= False , rug= False,label =answer_target_cols[1],axlabel =False)

#sns.distplot(train[answer_target_cols[2]], hist= False , rug= False,label =answer_target_cols[2],axlabel =False)

#sns.distplot(train[answer_target_cols[3]], hist= False , rug= False,label =answer_target_cols[3],axlabel =False)

sns.distplot(train[answer_target_cols[4]], hist= False , rug= False,label =answer_target_cols[4],axlabel =False)

plt.show()

## Removed two columns as value was quite high and other graphs were not visible .
# Lets see how the mean value of one target feature for questions changes based on category

for idx in range(20):

    df = train.groupby('category')[question_target_cols[idx]].mean()

        

    fig, axes = plt.subplots(1, 1, figsize=(10,10))

    axes.set_title(question_target_cols[idx])

    df.plot(label=question_target_cols[idx])

    plt.show()



## Lets see the words of first question



plt.figure(figsize=(20, 5))



text = train.question_body[0]



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
### Lets see the words of first answer

plt.figure(figsize=(20, 5))



text = train.answer[0]



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
##How many words are there in all questions ? 



text = " ".join(question_body for question_body in train.question_body)

print ("There are {} words in the combination of all questions.".format(len(text)))
## Load all questions in word cloud 



stopwords = set(STOPWORDS)

stopwords.update(["gt", "lt", "one", "use", "will","using"]) ## I found this list by first time running this cell without stopwords



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:

# the matplotlib way:

plt.figure(figsize=(20, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
##Lets do some unnecessary but nice masking

import cv2

im = cv2.imread("../input/worldcloud2/question_mark_col.png")

img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Generate a word cloud image

wc = WordCloud(background_color="black", max_words=1000, mask=img,

               stopwords=stopwords,max_font_size=90, random_state=42)

wc.generate(text)

image_colors = ImageColorGenerator(img)

# Display the generated image:

# the matplotlib way:

plt.figure(figsize=[12,12])

plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")

_=plt.show()
#clean comments

clean_mask=np.array(Image.open("../input/wordcloud3/Answer.jpg"))

clean_mask=clean_mask[:,:,1]

#wordcloud for clean comments

ans = " ".join(answer for answer in train.answer)

wc= WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=stopwords)

wc.generate(ans)

plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Words frequented in Answers", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()

import cv2

im = cv2.imread("../input/wordcloud3/Answer.jpg")

img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

ans = " ".join(answer for answer in train.answer)

# Generate a word cloud image

wc = WordCloud(background_color="white",max_words=1000, mask=img,

               stopwords=stopwords,max_font_size=90, random_state=42)

wc.generate(ans)

image_colors = ImageColorGenerator(img)

# Display the generated image:

# the matplotlib way:

plt.figure(figsize=[12,12])

plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")

_=plt.show()
# https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda/data as reference 
plt.figure(figsize=(20,20))

sns.heatmap(train[question_target_cols].corr(),vmin=-1,cmap='coolwarm')
## add category as label

train['cat_label'] = train['category'].rank(method='dense', ascending=False).astype(int)
## lets see if its correlated to any category 

question_target_cols.append('cat_label')



plt.figure(figsize=(20,20))

sns.heatmap(train[question_target_cols].corr(),vmin=-1,cmap='coolwarm')
##Lets do the same for answers 

plt.figure(figsize=(20,20))

sns.heatmap(train[answer_target_cols].corr(),vmin=-1,cmap='coolwarm')
##Baseline from https://www.kaggle.com/ryches/mean-of-categories-benchmark]

target_cols =question_target_cols+answer_target_cols

train["cat_host"]= train["category"]+train["host"]

category_means_map = train.groupby(["cat_host"])[target_cols].mean().T.to_dict()

preds = train["cat_host"].map(category_means_map).apply(pd.Series)
category_means_map.keys()
from scipy.stats import spearmanr

overall_score = 0

for col in target_cols:

    overall_score += spearmanr(preds[col], train[col]).correlation/len(target_cols)

    print(col, spearmanr(preds[col], train[col]).correlation)
overall_score
##Baseline from https://www.kaggle.com/ryches/mean-of-categories-benchmark

target_cols =question_target_cols+answer_target_cols

test["cat_host"]= test["category"]+test["host"]

#category_means_map = train.groupby(["cat_host"])[target_cols].mean().T.to_dict()

#preds = train["cat_host"].map(category_means_map).apply(pd.Series)
test_preds = test["cat_host"].map(category_means_map).apply(pd.Series)
test_preds
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

for col in target_cols:

    sub[col] = test_preds[col]
sub.fillna(value=0.000000,inplace=True)
sub.to_csv("submission.csv", index = False)
sub.describe()
sub.loc[sub['question_asker_intent_understanding'].isna() == True]
train.iloc[:,1:3]
#train_test_merge - analyse questions

merge=pd.concat([train.iloc[:,0:3],test.iloc[:,0:3]])

df_q=merge.reset_index(drop=True)
## Indirect features



#Sentense count in each comment:

    #  '\n' can be used to count the number of sentences in each comment

df_q['count_sent']=df_q["question_body"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:

df_q['count_word']=df_q["question_body"].apply(lambda x: len(str(x).split()))

#Unique word count

df_q['count_unique_word']=df_q["question_body"].apply(lambda x: len(set(str(x).split())))

#Letter count

df_q['count_letters']=df_q["question_body"].apply(lambda x: len(str(x)))

#punctuation count

df_q["count_punctuations"] =df_q["question_body"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count

df_q["count_words_upper"] = df_q["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count

df_q["count_words_title"] = df_q["question_body"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords

df_q["count_stopwords"] = df_q["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words

df_q["mean_word_len"] = df_q["question_body"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#derived features

#Word count percent in each comment:

df_q['word_unique_percent']=df_q['count_unique_word']*100/df_q['count_word']

#derived features

#Punct percent in each comment:

df_q['punct_percent']=df_q['count_punctuations']*100/df_q['count_word']
train.iloc[:,2:]
#serperate train and test features

train_feats=df_q.iloc[0:len(train),]

test_feats=df_q.iloc[len(train):,]

len(train)
train[question_target_cols]
question_target_cols.append('qa_id')
train_q = pd.merge(train_feats,train[question_target_cols], on='qa_id',how='left')

train_q
from matplotlib.ticker import StrMethodFormatter





train_q['count_sent'].loc[train_q['count_sent']>10] = 10 

plt.figure(figsize=(12,6))

## sentenses

plt.subplot(121)

plt.suptitle("Are longer questions more clear?",fontsize=20)

sns.violinplot(y='count_sent',x='question_well_written', data=train_q,split=True)

plt.xlabel('Clear?', fontsize=12)

plt.ylabel('# of sentences', fontsize=12)

plt.title("Number of sentences in each question", fontsize=15)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

# words

train_q['count_word'].loc[train_q['count_word']>1000] = 1000

plt.subplot(122)

sns.violinplot(y='count_word',x='question_well_written', data=train_q,split=True,inner="quart")

plt.xlabel('Clear?', fontsize=12)

plt.ylabel('# of words', fontsize=12)

plt.title("Number of words in each question", fontsize=15)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

#plt.show()
train_q['count_word']
#https://github.com/ahmedbesbes/Quora-Insincere-Questions-Classification

# import keras tokenizing utilities 

from keras.preprocessing import text, sequence



# import tensorboardX in case we want to log metrics to tensorboard (requires tensorflow installed - optional)

# import spacy for tokenization

import spacy

from tqdm import tqdm_notebook

tqdm_notebook().pandas()

# fastText is a library for efficient learning of word representations and sentence classification

# https://github.com/facebookresearch/fastText/tree/master/python

# I use it with a pre-trained english embedding that you can fetch from the official website

#import fastText

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
train = pd.read_csv(ROOT+'train.csv')

test = pd.read_csv(ROOT+'test.csv')

sub = pd.read_csv(ROOT+'sample_submission.csv')
def decontract(text):

    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)

    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)

    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)

    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)

    text = re.sub(r"(A|a)isn(\'|\’)t ", "is not ", text)

    text = re.sub(r"n(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)re ", " are ", text)

    text = re.sub(r"(\'|\’)d ", " would ", text)

    text = re.sub(r"(\'|\’)ll ", " will ", text)

    text = re.sub(r"(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)ve ", " have ", text)

    return text



def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '12345', x)

    x = re.sub('[0-9]{4}', '1234', x)

    x = re.sub('[0-9]{3}', '123', x)

    x = re.sub('[0-9]{2}', '12', x)

    return x



def preprocess(x):

    x= decontract(x)

    x=clean_text(x)

    x=clean_numbers(x)

    return x



train['question_body'] = train['question_body'].progress_map(lambda q: preprocess(q))

train['answer'] = train['answer'].progress_map(lambda q: preprocess(q))

train['question_title'] = train['question_title'].progress_map(lambda q: preprocess(q))





test['question_body'] = test['question_body'].progress_map(lambda q: preprocess(q))

test['answer'] = test['answer'].progress_map(lambda q: preprocess(q))

test['question_title'] = test['question_title'].progress_map(lambda q: preprocess(q))
# define tokenization parameters 



MAX_WORDS = 40000

MAX_LEN = 500
all_questions = (train['question_body']+'' +train['answer']).tolist() + (test['question_body']+''+test['answer']).tolist()

len(all_questions)
from keras.preprocessing.text import Tokenizer



train['text'] = train['question_body'] + ' ' + train['answer'] +' '+ train['question_title']

test['text'] = test['question_body'] + ' ' + test['answer'] +' '+ test['question_title']





full_text = list(train['text'].values) + list(test['text'].values)



tk = Tokenizer(lower = False, filters='')

tk.fit_on_texts(full_text)

print(f'Number of words in the dictionary: {len(tk.word_index)}')



train_df, valid_df, y_train, y_valid = train_test_split(train, train[sub.columns[1:]], test_size=0.1)

train_tokenized = tk.texts_to_sequences(train_df['question_body'] + ' ' + train_df['answer']+ ' '+ + train_df['question_title'])

valid_tokenized = tk.texts_to_sequences(valid_df['question_body'] + ' ' + valid_df['answer']+ ' '+ valid_df['question_title'])

test_tokenized = tk.texts_to_sequences(test['text'])



max_len = 500

X_train = pad_sequences(train_tokenized, maxlen = max_len)

X_valid = pad_sequences(valid_tokenized, maxlen = max_len)

X_test = pad_sequences(test_tokenized, maxlen = max_len)

!ls '../input/wikinews300d1mvec/wiki-news-300d-1M.vec'
def get_embedding_path(embedding):

    embedding_zoo = {"crawl": "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec","crawl_sub":"../input/fasttext-crawl-300d-2m-with-subword/crawl-300d-2m-subword/crawl-300d-2M-subword.vec",

                "glove": "../input/glove840b/glove.840B.300d.txt","paragram":"../input/paragram-300-sl999/paragram_300_sl999.txt", "wikinews":"../input/wikinews300d1mvec/wiki-news-300d-1M.vec" }

    return embedding_zoo.get(embedding)



embed_size = 300

max_features = 100000

def get_coefs(word,*arr):

    return word, np.asarray(arr, dtype='float32')



def build_matrix(embedding, tokenizer):

    embedding_path= get_embedding_path(embedding)

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding='utf-8'))



    word_index = tk.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.zeros((nb_words + 1, embed_size))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix







def sigmoid(x):

    return 1 / (1 + np.exp(-x))
"""%%time

embedding_matrix_1 = build_matrix("crawl_sub", tk)

#embedding_matrix_2 = build_matrix("paragram", tk)

embedding_matrix_3 = build_matrix("glove", tk)

#embedding_matrix_4 = build_matrix("wikinews", tk)



#embedding_matrix = np.mean((1.28*embedding_matrix_1, 0.72*embedding_matrix_2,2*embedding_matrix_3), axis=0)

embedding_matrix =np.mean((embedding_matrix_1, embedding_matrix_3), axis=0)

#del embedding_matrix_2,

del embedding_matrix_1,embedding_matrix_3

gc.collect()

np.shape(embedding_matrix)"""
%%time

embedding_matrix_crawl = build_matrix("crawl", tk)
"""import pickle

GLOVE_EMBEDDING_PATH = '/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl' 

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.zeros((nb_words + 1, embed_size))

    unknown_words = []

    

    for word, i in word_index.items():

        if i >= max_features:

            continue

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words

"""
#tic = time.time()

#embedding_matrix_glove,_ = build_matrix(tk.word_index,GLOVE_EMBEDDING_PATH)
#embedding_matrix =np.mean((embedding_matrix_crawl, embedding_matrix_glove), axis=0)

embedding_matrix = embedding_matrix_crawl

del embedding_matrix_crawl#,embedding_matrix_glove

gc.collect()
"""From built-in optimizer classes.

"""

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import six

import copy

from six.moves import zip



from keras import backend as K

from keras.utils.generic_utils import serialize_keras_object

from keras.utils.generic_utils import deserialize_keras_object

from keras.legacy import interfaces



from keras.optimizers import Optimizer



class AdamW(Optimizer):

    """AdamW optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments

        lr: float >= 0. Learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: float >= 0. Weight decay (L2 penalty) (default: 0.025).

        batch_size: integer >= 1. Batch size used during training.

        samples_per_epoch: integer >= 1. Number of samples (training points) per epoch.

        epochs: integer >= 1. Total number of epochs for training. 

    # References

        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., weight_decay=0.025, 

                 batch_size=1, samples_per_epoch=1, 

                 epochs=1, **kwargs):

        super(AdamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

            self.weight_decay = K.variable(weight_decay, name='weight_decay')

            self.batch_size = K.variable(batch_size, name='batch_size')

            self.samples_per_epoch = K.variable(samples_per_epoch, name='samples_per_epoch')

            self.epochs = K.variable(epochs, name='epochs')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1

        '''Bias corrections according to the Adam paper

        '''

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            

            '''Schedule multiplier eta_t = 1 for simple AdamW

            According to the AdamW paper, eta_t can be fixed, decay, or 

            also be used for warm restarts (AdamWR to come). 

            '''

            eta_t = 1.

            p_t = p - eta_t*(lr_t * m_t / (K.sqrt(v_t) + self.epsilon))

            if self.weight_decay != 0:

                '''Normalized weight decay according to the AdamW paper

                '''

                w_d = self.weight_decay*K.sqrt(self.batch_size/(self.samples_per_epoch*self.epochs))

                p_t = p_t - eta_t*(w_d*p) 



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'weight_decay': float(K.get_value(self.weight_decay)),

                  'batch_size': int(K.get_value(self.batch_size)),

                  'samples_per_epoch': int(K.get_value(self.samples_per_epoch)),

                  'epochs': int(K.get_value(self.epochs)),

                  'epsilon': self.epsilon}

        base_config = super(AdamW, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        """

        Keras Layer that implements an Attention mechanism for temporal data.

        Supports Masking.

        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, features)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Example:

            model.add(LSTM(64, return_sequences=True))

            model.add(Attention())

        """

        self.supports_masking = True

        #self.init = initializations.get('glorot_uniform')

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight(shape=(input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight(shape=(input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        # eij = K.dot(x, self.W) TF backend doesn't support it



        # features_dim = self.W.shape[0]

        # step_dim = x._keras_shape[1]



        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

    #print weigthted_input.shape

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        #return input_shape[0], input_shape[-1]

        return input_shape[0],  self.features_dim

from keras.layers import *

from keras.models import *

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import *

from keras.optimizers import *

import keras.backend as K

from keras.callbacks import *
class AttentionWeightedAverage(Layer):

    """

    Computes a weighted average of the different channels across timesteps.

    Uses 1 parameter pr. channel to compute the attention value for a single timestep.

    """



    def __init__(self, return_attention=False, **kwargs):

        self.init = initializers.RandomUniform(seed=10000)

        self.supports_masking = True

        self.return_attention = return_attention

        super(AttentionWeightedAverage, self).__init__(** kwargs)



    def build(self, input_shape):

        self.input_spec = [InputSpec(ndim=3)]

        assert len(input_shape) == 3



        self.W = self.add_weight(shape=(input_shape[2], 1),

                                 name='{}_W'.format(self.name),

                                 initializer=self.init)

        self.trainable_weights = [self.W]

        super(AttentionWeightedAverage, self).build(input_shape)



    def call(self, x, mask=None):

        # computes a probability distribution over the timesteps

        # uses 'max trick' for numerical stability

        # reshape is done to avoid issue with Tensorflow

        # and 1-dimensional weights

        logits = K.dot(x, self.W)

        x_shape = K.shape(x)

        logits = K.reshape(logits, (x_shape[0], x_shape[1]))

        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))



        # masked timesteps have zero weight

        if mask is not None:

            mask = K.cast(mask, K.floatx())

            ai = ai * mask

        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())

        weighted_input = x * K.expand_dims(att_weights)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:

            return [result, att_weights]

        return result



    def get_output_shape_for(self, input_shape):

        return self.compute_output_shape(input_shape)



    def compute_output_shape(self, input_shape):

        output_len = input_shape[2]

        if self.return_attention:

            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]

        return (input_shape[0], output_len)



    def compute_mask(self, input, input_mask=None):

        if isinstance(input_mask, list):

            return [None] * len(input_mask)

        else:

            return None

class AdamW(Optimizer):

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)

                 epsilon=1e-8, decay=0., **kwargs):

        super(AdamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.learning_rate = K.variable(learning_rate, name='learning_rate')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)

        self.epsilon = epsilon

        self.initial_decay = decay



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]

        wd = self.wd # decoupled weight decay (3/4)



        learning_rate = self.learning_rate

        if self.initial_decay > 0:

            learning_rate *= (1. / (1. + self.decay * K.cast(self.iterations,

                                                  K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1

        lr_t = learning_rate * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - learning_rate * wd * p # decoupled weight decay (4/4)



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'learning_rate': float(K.get_value(self.learning_rate)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'weight_decay': float(K.get_value(self.wd)),

                  'epsilon': self.epsilon}

        base_config = super(AdamW, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
max_len = 500

max_features =embedding_matrix.shape[0]

embedding_matrix.shape
# Compatible with tensorflow backend

class SpearmanRhoCallback(Callback):

    def __init__(self, training_data, validation_data, patience, model_name):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]

        

        self.patience = patience

        self.value = -1

        self.bad_epochs = 0

        self.model_name = model_name



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred_val = self.model.predict(self.x_val)

        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])

        if rho_val >= self.value:

            self.value = rho_val

        else:

            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:

            print("Epoch %05d: early stopping Threshold" % epoch)

            self.model.stop_training = True

            #self.model.save_weights(self.model_name)

        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')

        return rho_val



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
def build_model_zoo(X_train, y_train, X_valid, y_valid, embedding_matrix, lr=0.0,lr_d=0.0, units=0, spatial_dr=0.0, dense_units=128,

                dr=0.1, epochs=5, use_attention=True,model_type = 'bigrucnn',batch_size=256):



    spatialdropout=0.20

    rnn_units=64

    weight_decay=0.07

    filters=[100, 80, 30, 12]

    file_path = f'{MODEL_TYPE}_best_model.hdf5'

    

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    #early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 30)

    #spearman = 

    scheduler = ReduceLROnPlateau(patience=3)

    if model_type == 'bigruatt':

        inp = Input(shape = (max_len,))

        x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)

        x1 = SpatialDropout1D(spatial_dr)(x)



        x_gru = Bidirectional(GRU(units * 2, return_sequences = True))(x1)

        if use_attention:

            x_att = Attention(max_len)(x_gru)

            x = Dropout(dr)(Dense(dense_units, activation='relu') (x_att))

        else:

            x_att = Flatten() (x_gru)

            x = Dropout(dr)(Dense(dense_units, activation='relu') (x_att))



        x = BatchNormalization()(x)

    #x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))

        x = Dense(30, activation = "sigmoid")(x)

        model = Model(inputs = inp, outputs = x)

    elif model_type == 'bigrucnn':

        inp = Input(shape=(max_len,))

        x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

        x = SpatialDropout1D(spatialdropout)(x)

        x = Bidirectional(GRU(rnn_units, return_sequences=True))(x)



        x1 = Conv1D(filters=filters[0], activation='relu', kernel_size=1, 

                    padding='same', kernel_initializer=glorot_uniform(seed=110000))(x)

        x2 = Conv1D(filters=filters[1], activation='relu', kernel_size=2, 

                    padding='same', kernel_initializer=glorot_uniform(seed=120000))(x)

        x3 = Conv1D(filters=filters[2], activation='relu', kernel_size=3, 

                    padding='same', kernel_initializer=glorot_uniform(seed=130000))(x)

        x4 = Conv1D(filters=filters[3], activation='relu', kernel_size=5, 

                    padding='same', kernel_initializer=glorot_uniform(seed=140000))(x)



    

        x1 = GlobalMaxPool1D()(x1)

        x2 = GlobalMaxPool1D()(x2)

        x3 = GlobalMaxPool1D()(x3)

        x4 = GlobalMaxPool1D()(x4)



        c = concatenate([x1, x2, x3, x4])

        x = Dense(200, activation='relu', kernel_initializer=glorot_uniform(seed=111000))(c)

        x = Dropout(0.2, seed=10000)(x)

        x = BatchNormalization()(x)

        x = Dense(30, activation="sigmoid", kernel_initializer=glorot_uniform(seed=110000))(x)

        model = Model(inputs=inp, outputs=x)

    elif model_type == 'poolrnn':

        inp = Input(shape=(max_len,))

        embedding_layer = Embedding(max_features,

                               embed_size,

                                weights=[embedding_matrix],

                                input_length=max_len,

                                trainable=False)(inp)

        embedding_layer = SpatialDropout1D(spatialdropout, seed=1024)(embedding_layer)



        rnn_1 = Bidirectional(GRU(rnn_units, return_sequences=True, 

                                   kernel_initializer=glorot_uniform(seed=10000), 

                                   recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(embedding_layer)      



        last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)

        maxpool = GlobalMaxPooling1D()(rnn_1)

        attn = AttentionWeightedAverage()(rnn_1)

        average = GlobalAveragePooling1D()(rnn_1)



        c = concatenate([last, maxpool, attn], axis=1)

        c = Reshape((3, -1))(c)

        c = Lambda(lambda x:K.sum(x, axis=1))(c)

        x = BatchNormalization()(c)

        x = Dense(200, activation='relu', kernel_initializer=glorot_uniform(seed=111000))(x)

        x = Dropout(0.2, seed=1024)(x)

        x = BatchNormalization()(x)

        output_layer = Dense(30, activation="sigmoid", kernel_initializer=glorot_uniform(seed=111000))(x)

        model = Model(inputs=inp, outputs=output_layer)

        

        

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_valid, y_valid), 

                        verbose = 0, callbacks = [check_point,SpearmanRhoCallback(training_data=(X_train, y_train), validation_data=(X_valid, y_valid),

                                       patience=5, model_name=f'best_model_batch.h5')])

    #model = load_model(file_path)

    return model

kfold = KFold(n_splits=3, random_state=42, shuffle=True)

bestscore = []

bestloss = []

y_test = np.zeros((X_test.shape[0], ))

oof = np.zeros((X_valid.shape[0], ))

predict_list = []

epochs = [15, 20, 17, 18]

val_list = []

val_pred_list = []

for i, (train_index, valid_index) in enumerate(kfold.split(X_train,y_train.values)):

    print(len(train_index))

    print(len(valid_index))

    val_list += list(valid_index)

    print('FOLD%s'%(i+1))

    if i ==0 :

        MODEL_TYPE='bigruatt'

        BATCH_SIZE=64

    elif i ==1 :

        MODEL_TYPE='bigrucnn'

        BATCH_SIZE = 64

    else:

        MODEL_TYPE = 'poolrnn'

        BATCH_SIZE = 64

        

    X_tr, X_val, Y_tr, Y_val = X_train[train_index], X_train[valid_index], y_train.values[train_index], y_train.values[valid_index]

    model = build_model_zoo(X_tr, Y_tr, X_val, Y_val, embedding_matrix, lr = 1e-2, units = 64,

                    spatial_dr = 0.1, dense_units=128, dr=0.1, epochs=30,model_type=MODEL_TYPE,batch_size=BATCH_SIZE) ##There was a bug in earlier Kernel

    

    valid_pred = model.predict(X_valid, batch_size = 256, verbose = 1)

    score = 0

    for j, col in enumerate(train[sub.columns[1:]]):

        score += np.nan_to_num(spearmanr(y_valid[col], valid_pred[:, j]).correlation)

    print(score / (j + 1))    

    prediction = np.nan_to_num(model.predict(X_test, batch_size = 256, verbose = 1))

    predict_list.append(prediction)

    val_pred_list.append(valid_pred)
valid_pred = model.predict(X_valid, batch_size = 1024, verbose = 1)

score = 0

for j, col in enumerate(train[sub.columns[1:]]):

        score += np.nan_to_num(spearmanr(y_valid[col], valid_pred[:, j]).correlation)

print(score / (j + 1))    
val_pred_mean = 0.4*val_pred_list[0]+0.3*val_pred_list[1]+0.3*val_pred_list[2]

score = 0

for i, col in enumerate(train[sub.columns[1:]]):

    score += np.nan_to_num(spearmanr(y_valid[col], val_pred_mean[:, i]).correlation)

    print(score)
val_pred_mean
print(score / (i + 1))
#prediction = np.nan_to_num(model.predict(X_test, batch_size = 1024, verbose = 1))

prediction_mean = 0.33*predict_list[0]+0.33*predict_list[1]+0.34*predict_list[2]
sub[sub.columns[1:]] = sigmoid(prediction_mean)

sub.to_csv('submission.csv', index=False)