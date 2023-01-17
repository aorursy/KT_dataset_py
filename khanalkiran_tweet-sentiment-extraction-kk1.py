# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os

import re

import nltk

import string

import warnings

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import conll2000

from nltk.corpus import brown

from nltk.stem.wordnet import WordNetLemmatizer

import string 

import plotly.graph_objs as go

import plotly.express as px

import matplotlib.pyplot as pt

import seaborn as sns



%matplotlib inline



from collections import defaultdict

from collections import Counter

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import transformers

from tqdm import tqdm

import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch



import torch 

from torch import nn

from torch.nn import functional as F

import torch.optim as optim



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from plotly import tools

from plotly.subplots import make_subplots





import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold 

from transformers import *

import tokenizers





warnings.filterwarnings('ignore')
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print(train_data.shape, test_data.shape, submission_data.shape)

display(train_data.head())

display(test_data.head())

display(submission_data.head())
train_data.isna().any()
test_data.isna().any()
train_data.loc[train_data['selected_text'].isnull()]
train_data.loc[train_data['text'].isnull()]
# Removing the row from train data with missing value

train_data_new = train_data.drop([314])
train_data_new.isna().any()
train_data['sentiment'].unique() 
def count_senti(df):

    sum = df['sentiment'].value_counts()

    percent = df['sentiment'].value_counts(normalize = True)

    return pd.concat([sum, percent], axis = 1, keys = ['Sum', 'Percent'])
print("Sentiments for train data")

senti_train = count_senti(train_data)

display(senti_train)

print("Sentiments for test data")

senti_test = count_senti(test_data)

display(senti_test)
colors = ['purple', 'green', 'red']

fig = make_subplots(rows = 1, cols = 2, specs = [[{"type":"pie"}, {"type":"pie"}]])

fig.add_trace(go.Pie(labels = list(senti_train.index),

                     values = list(senti_train.Sum.values), hoverinfo = 'label+percent', 

                     textinfo = 'value+percent',

                     marker = dict(colors = colors)), row = 1, col = 1)

fig.add_trace(go.Pie(labels = list(senti_test.index), 

                     values = list(senti_test.Sum.values), hoverinfo = 'label+percent', 

                     textinfo = 'value+percent',

                     marker = dict(colors =colors)), row = 1, col = 2)

fig.update_layout(title_text = "Train and Test Sentiment Percentages", title_x = 0.5)
df_train = train_data.copy()

df_test = test_data.copy()
df_train[df_train['sentiment'] == 'positive']
df_train[df_train['sentiment'] == 'neutral']
df_train[df_train['sentiment'] == 'negative']
def text_cleaning(txt):

    """

    Convert given text to lower case, remove all non-word characters, digits, links.

    """

    txt = str(txt).lower()

    txt= re.sub('https?://\S+|www\.\S+', '', txt)

    txt = re.sub('\[.*?\]', '', txt)

    txt = re.sub('<.*?>+', '', txt)

    txt = re.sub('[%s]' % re.escape(string.punctuation), ' ', txt)

    txt = re.sub('\n', '', txt)

    txt = re.sub('\w*\d\w*', '', txt)

    return txt
def rm_stopword(text):

    text_tokens = word_tokenize(text)

    stop_list = stopwords.words('english')

    new_text = [word for word in text_tokens if word not in stop_list]

    final_text = ' '.join(new_text)

    return final_text
#Clean text and selected_text

df_train['clean_text'] = df_train['text'].apply(lambda x: text_cleaning(x))

df_train['clean_selected_text'] = df_train['selected_text'].apply(lambda x: text_cleaning(x))
df_train.head(50)
# Remove stopwords from both text and selected text

df_train['clean_text'] = df_train['clean_text'].apply(lambda x: rm_stopword(x))

df_train['clean_selected_text'] = df_train['clean_selected_text'].apply(lambda x: rm_stopword(x))
df_train.head()
def count_words(df,feature, senti):

    word_list = []

    for x in df[df['sentiment'] == senti][feature].str.split():

        for i in x:

            word_list.append(i)

    cnt = Counter()

    for word in word_list:

        cnt[word] +=1

    df_cnt = pd.DataFrame(cnt.most_common(10))

    df_cnt.columns = ['Freq_words', 'Freq']

    df_cnt.style.background_gradient(cmap = 'purple')

    return df_cnt 
positive_top10 = count_words(df_train,'clean_text','positive')

display(positive_top10)



fig = px.bar(positive_top10, x = 'Freq', y = 'Freq_words', title = 'Top 10 frequent postive words',

            orientation = 'h', width = 600, height = 600, color = 'Freq_words')

fig.show()
neutral_top10 = count_words(df_train,'clean_text','neutral')

display(neutral_top10)



fig = px.bar(neutral_top10, x = 'Freq', y = 'Freq_words', title = 'Top 10 frequent neutral words',

            orientation = 'h', width = 600, height = 600, color = 'Freq_words')

fig.show()
negative_top10 = count_words(df_train,'clean_text','negative')

display(negative_top10)



fig = px.bar(negative_top10, x = 'Freq', y = 'Freq_words', title = 'Top 10 frequent negative words',

            orientation = 'h', width = 600, height = 600, color = 'Freq_words')

fig.show()
def pos_freq(df,feature, senti):

    total_pos_count = []

    for x in df[df['sentiment'] == senti][feature].str.split():

        pos_count = nltk.pos_tag(x)

        total_pos_count.extend(pos_count) 

    tag_freq = nltk. FreqDist(tag for (word, tag) in total_pos_count)

    ans = tag_freq.most_common()[0:10]

    return ans 
pos_freq(df_train,'text', 'negative')
pos_freq(df_train,'text', 'positive')
MAX_LEN = 198

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 8

EPOCHS = 10

ROBERTA_PATH = "../input/roberta-base/"

MODEL_PATH = "model.bin"

TRAINING_FILE = "../input/train.csv"



TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{ROBERTA_PATH}/vocab.json", 

    merges_file=f"{ROBERTA_PATH}/merges.txt", 

    lowercase=True,

    add_prefix_space=True

)
class TweetModel(nn.Module):

    def __init__(self):

        super(TweetModel, self).__init__()

        self.bert = transformers.RobertaModel.from_pretrained(ROBERTA_PATH)

        self.l0 = nn.Linear(768, 2)

    

    def forward(self, ids, mask, token_type_ids):

        sequence_output, pooled_output = self.bert(

            ids, 

            attention_mask=mask

        )

        logits = self.l0(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits
device = torch.device("cuda")

model = TweetModel()

model.to(device)

model = nn.DataParallel(model)

model.load_state_dict(torch.load("../input/roberta-tweet-model/model.bin"))

model.eval()
class TweetDataset:

    def __init__(self, tweet, sentiment, selected_text):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.tokenizer = TOKENIZER

        self.max_len = MAX_LEN

    

    def __len__(self):

        return len(self.tweet)

    

    def __getitem__(self, item):

        # For Roberta, CLS = <s> and SEP = </s>

        # Multiple strings: '<s>hi, my name is abhishek!!!</s></s>whats ur name</s>'

        # id for <s>: 0

        # id for </s>: 2

    

        tweet = " " + " ".join(str(self.tweet[item]).split())

        selected_text = " " + " ".join(str(self.selected_text[item]).split())

        

        len_st = len(selected_text)

        idx0 = -1

        idx1 = -1

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):

            if tweet[ind: ind+len_st] == selected_text:

                idx0 = ind

                idx1 = ind + len_st

                break

        #print(f"idx0: {idx0}")

        #print(f"idx1: {idx1}")

        #print(f"len_st: {len_st}")

        #print(f"idxed tweet: {tweet[idx0: idx1]}")



        char_targets = [0] * len(tweet)

        if idx0 != -1 and idx1 != -1:

            for ct in range(idx0, idx1):

                # if tweet[ct] != " ":

                char_targets[ct] = 1



        #print(f"char_targets: {char_targets}")



        tok_tweet = self.tokenizer.encode(tweet)

        tok_tweet_tokens = tok_tweet.tokens

        tok_tweet_ids = tok_tweet.ids

        tok_tweet_offsets = tok_tweet.offsets

        

         #print(tweet)

        #print(selected_text)

        #print(tok_tweet_tokens)

        #print(f"tok_tweet.offsets= {tok_tweet.offsets}")

        

        targets = [0] * len(tok_tweet_ids)

        target_idx = []

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):

            #print("**************")

            #print(offset1, offset2)

            #print(tweet[offset1: offset2])

            #print(char_targets[offset1: offset2])

            #print("".join(tok_tweet_tokens)[offset1: offset2])

            #print("**************")

            if sum(char_targets[offset1: offset2]) > 0:

                targets[j] = 1

                target_idx.append(j)



        #print(f"targets= {targets}")

        #print(f"target_idx= {target_idx}")



        #print(tok_tweet_tokens[target_idx[0]])

        #print(tok_tweet_tokens[target_idx[-1]])

        

        targets_start = [0] * len(targets)

        targets_end = [0] * len(targets)



        non_zero = np.nonzero(targets)[0]

        if len(non_zero) > 0:

            targets_start[non_zero[0]] = 1

            targets_end[non_zero[-1]] = 1

        

        #print(targets_start)

        #print(targets_end)

        #print(tok_tweet_tokens)

        #print([x for jj, x in enumerate(tok_tweet_tokens) if targets_start[jj] == 1])

        #print([x for jj, x in enumerate(tok_tweet_tokens) if targets_end[jj] == 1])

        



        # check padding:

        # <s> pos/neg/neu </s> </s> tweet </s>

        if len(tok_tweet_tokens) > self.max_len - 5:

            tok_tweet_tokens = tok_tweet_tokens[:self.max_len - 5]

            tok_tweet_ids = tok_tweet_ids[:self.max_len - 5]

            targets_start = targets_start[:self.max_len - 5]

            targets_end = targets_end[:self.max_len - 5]

        

        # positive: 1313

        # negative: 2430

        # neutral: 7974

        

        sentiment_id = {

            'positive': 1313,

            'negative': 2430,

            'neutral': 7974

        }



        tok_tweet_ids = [0] + [sentiment_id[self.sentiment[item]]] + [2] + [2] + tok_tweet_ids + [2]

        targets_start = [0] + [0] + [0] + [0] + targets_start + [0]

        targets_end = [0] + [0] + [0] + [0] + targets_end + [0]

        token_type_ids = [0, 0, 0, 0] + [0] * (len(tok_tweet_ids) - 5) + [0]

        mask = [1] * len(token_type_ids)



        #print("Before padding")

        #print(f"len(tok_tweet_ids)= {len(tok_tweet_ids)}")

        #print(f"len(targets_start)= {len(targets_start)}")

        #print(f"len(targets_end)= {len(targets_end)}")

        #print(f"len(token_type_ids)= {len(token_type_ids)}")

        #print(f"len(mask)= {len(mask)}")



        padding_length = self.max_len - len(tok_tweet_ids)

        

        tok_tweet_ids = tok_tweet_ids + ([1] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        targets_start = targets_start + ([0] * padding_length)

        targets_end = targets_end + ([0] * padding_length)

        

        #print("After padding")

        #print(f"len(tok_tweet_ids)= {len(tok_tweet_ids)}")

        #print(f"len(targets_start)= {len(targets_start)}")

        #print(f"len(targets_end)= {len(targets_end)}")

        #print(f"len(token_type_ids)= {len(token_type_ids)}")

        #print(f"len(mask)= {len(mask)}")



        return {

            'ids': torch.tensor(tok_tweet_ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'targets_start': torch.tensor(targets_start, dtype=torch.float),

            'targets_end': torch.tensor(targets_end, dtype=torch.float),

            'padding_len': torch.tensor(padding_length, dtype=torch.long),

            'orig_tweet': self.tweet[item],

            'orig_selected': self.selected_text[item],

            'sentiment': self.sentiment[item]

        }

    
df1_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

df1_test.loc[:, "selected_text"] = df1_test.text.values
test_dataset = TweetDataset(

        tweet=df1_test.text.values,

        sentiment=df1_test.sentiment.values,

        selected_text=df1_test.selected_text.values

    )



data_loader = torch.utils.data.DataLoader(

    test_dataset,

    shuffle=False,

    batch_size=VALID_BATCH_SIZE,

    num_workers=1

)
all_outputs = []

fin_outputs_start = []

fin_outputs_end = []

fin_padding_lens = []

fin_orig_selected = []

fin_orig_sentiment = []

fin_orig_tweet = []

fin_tweet_token_ids = []



with torch.no_grad():

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        padding_len = d["padding_len"]

        sentiment = d["sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.float)

        targets_end = targets_end.to(device, dtype=torch.float)



        outputs_start, outputs_end = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())

        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())



        fin_orig_sentiment.extend(sentiment)

        fin_orig_selected.extend(orig_selected)

        fin_orig_tweet.extend(orig_tweet)

        

        



fin_outputs_start = np.vstack(fin_outputs_start)

fin_outputs_end = np.vstack(fin_outputs_end)

fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)

jaccards = []

threshold = 0.2

for j in range(fin_outputs_start.shape[0]):

    target_string = fin_orig_selected[j]

    padding_len = fin_padding_lens[j]

    sentiment_val = fin_orig_sentiment[j]

    original_tweet = fin_orig_tweet[j]



    if padding_len > 0:

        mask_start = fin_outputs_start[j, 4:-1][:-padding_len] >= threshold

        mask_end = fin_outputs_end[j, 4:-1][:-padding_len] >= threshold

        tweet_token_ids = fin_tweet_token_ids[j, 4:-1][:-padding_len]

    else:

        mask_start = fin_outputs_start[j, 4:-1] >= threshold

        mask_end = fin_outputs_end[j, 4:-1] >= threshold

        tweet_token_ids = fin_tweet_token_ids[j, 4:-1][:-padding_len]

        

        

        

    mask = [0] * len(mask_start)

    idx_start = np.nonzero(mask_start)[0]

    idx_end = np.nonzero(mask_end)[0]

    if len(idx_start) > 0:

        idx_start = idx_start[0]

        if len(idx_end) > 0:

            idx_end = idx_end[0]

        else:

            idx_end = idx_start

    else:

        idx_start = 0

        idx_end = 0



    for mj in range(idx_start, idx_end + 1):

        mask[mj] = 1



    output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]



    filtered_output = TOKENIZER.decode(output_tokens)

    filtered_output = filtered_output.strip().lower()



    all_outputs.append(filtered_output.strip())



sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = all_outputs

sample.to_csv("submission.csv", index=False)
sample.head()