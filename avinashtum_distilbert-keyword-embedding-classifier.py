import numpy as np # linear algebra

from tqdm import tqdm_notebook

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import pdb
!pip install transformers
import plotly.graph_objs as go

from matplotlib import pyplot as plt

import plotly.offline as py

import regex as re

from bs4 import BeautifulSoup

import string

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from nltk.tokenize import word_tokenize

from gensim.models import word2vec


import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
import nltk

nltk.download('punkt')
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import torch

import transformers as ppb

import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
def get_embedding(df,col):

    

    corpus = create_corpus(df,col)

    w2v_model = word2vec.Word2Vec([corpus], size=50, window=10, min_count=1, workers=4)

    embedding_dict={}

    for word in corpus:

        embedding_dict[word] = w2v_model.wv[word]



    num_words=len(corpus)

    embedding_matrix=np.zeros((num_words,50))



    for i,word in tqdm_notebook(enumerate(df[col].unique())):

        if i > num_words:

            continue

        emb_vec=embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec

    return embedding_matrix



def create_corpus(df,col):

    corpus=[]

    for keyword in tqdm_notebook(df[col].unique()):

        keyword=keyword.lower()

        corpus.append(keyword)

    return corpus
def combine_features(df,features,cat_cols):

    

    df_copy=df.copy()



    for col in cat_cols:

        embedding = get_embedding(df_copy,col)

        vec = {val:embedding[i]  for i, val in enumerate(df_copy[col].unique())}

        for key,value in vec.items():

            df_copy[col] = df_copy[col].map(lambda x: value if x == key else x)

        

        embed_array = np.stack(df_copy[col].values,axis=0)

        features = np.concatenate((features,embed_array),axis=1)

    return features
train.head()
# lets understand the target distribution

target_cnt=train.target.value_counts()



labels = (np.array(target_cnt.index))

sizes = (np.array((target_cnt / target_cnt.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(

    title='Target distribution',

    font=dict(size=10),

    width=300,

    height=300,

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="usertype")
# fill NaN with empty string

train['keyword'] = train.keyword.fillna(value='None')

train['location'] = train.location.fillna(value='None')



test['keyword'] = test.keyword.fillna(value='None')

test['location'] = test.location.fillna(value='None')
train.head()


def get_features(data, batch_size=2500):

    # Use DistilBERT as feature extractor:

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # Load pretrained model/tokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    model = model_class.from_pretrained(pretrained_weights)

    model.to(device)

    model = nn.DataParallel(model)

    

    # tokenize,padding and masking

    tokenized = data["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 0

    for i in tokenized.values:

        if len(i) > max_len:

            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])



    attention_mask = np.where(padded != 0, 1, 0)

    

    last_hidden_states=[]

    no_batch = data.shape[0]//batch_size

    start_index=0

    end_index=1

    for i in tqdm_notebook(range(1,no_batch+2)):



        if  data.shape[0]>batch_size*i:

                end_index=batch_size*i

        else:

            end_index=train.shape[0]



        input_ids = torch.tensor(padded[start_index:end_index])  

        batch_attention_mask = torch.tensor(attention_mask[start_index:end_index])



        input_ids = input_ids.to(device)

        batch_attention_mask = batch_attention_mask.to(device)



        with torch.no_grad():

            batch_hidden_state = model(input_ids, attention_mask=batch_attention_mask)

            print("Batch {} is completed sucessfully".format(i))

            last_hidden_states.append(batch_hidden_state[0])



        start_index=batch_size*i

        end_index=batch_size*i

    fin_features = torch.cat(last_hidden_states,0)

    clf_features = fin_features[:,0,:].cpu().numpy()

    return clf_features
gc.collect()

features = get_features(train,batch_size=2500)

test_distil_features = get_features(test,batch_size=2500)
cat_cols = ['keyword']
train_features = combine_features(train,features,cat_cols)

test_features = combine_features(test,test_distil_features,cat_cols)
labels = train["target"]
## Use features from previous modle and train a Logistic regression model

# labels = train["target"]

# train model

lr_clf = LogisticRegression()

lr_clf.fit(train_features, labels)
test_pred = lr_clf.predict(test_features)
submission['target'] = test_pred

submission.to_csv('submission.csv', index=False)