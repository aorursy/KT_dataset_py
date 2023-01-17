import nltk 

w1 = set('AI is our friend and it has been friendly'.lower().split())

w2 = set('AI and humans have always been friendly'.lower().split())

 

print ("Jaccard similarity of above two sentences is",1-nltk.jaccard_distance(w1, w2))
w1 = set('Kaggle is awesome'.lower().split())

w2 = set('kaggle is great way of learning DS'.lower().split())

print("The Jaccard similarity is:",1-nltk.jaccard_distance(w1, w2))
!pip install "/kaggle/input/chart-studio/chart_studio-1.0.0-py3-none-any.whl"

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

import re

import string



import matplotlib.pyplot as plt

from plotly import tools

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



import os

import tokenizers

import torch

import transformers

import torch.nn as nn

from tqdm import tqdm







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# word level jaccard score: https://www.kaggle.com/c/tweet-sentiment-extraction/overview/evaluation

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train['text'] = train['text'].str.replace('[{}]'.format(string.punctuation), '')

test['text'] = test['text'].str.replace('[{}]'.format(string.punctuation), '')
train.head(3)
test.head(3)
print('Sentiment of text : {} \nOur training text :\n{}\nSelected text which we need to predict:\n{}'.format(train['sentiment'][0],train['text'][0],train['selected_text'][0]))
# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=600, 

                    height=300,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train.loc[train['sentiment'] == 'neutral', 'text'].append(test.loc[test['sentiment'] == 'neutral', 'text']), title="Word Cloud of Neutral tweets",color = 'white')
plot_wordcloud(train.loc[train['sentiment'] == 'positive', 'text'].append(test.loc[test['sentiment'] == 'positive', 'text']), title="Word Cloud of Positive tweets",color = 'green')
plot_wordcloud(train.loc[train['sentiment'] == 'negative', 'text'].append(test.loc[test['sentiment'] == 'negative', 'text']), title="Word Cloud of negative tweets",color = 'red')
from collections import defaultdict

train0_df = train[train["sentiment"]=='positive'].dropna().append(test[test["sentiment"]=='positive'].dropna())

train1_df = train[train["sentiment"]=='neutral'].dropna().append(test[test["sentiment"]=='neutral'].dropna())

train2_df = train[train["sentiment"]=='negative'].dropna().append(test[test["sentiment"]=='negative'].dropna())



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'red')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'green')



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train2_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of positive tweets", "Frequent words of neutral tweets",

                                          "Frequent words of negative tweets"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'gray')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



freq_dict = defaultdict(int)

for sent in train2_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'brown')







# Creating two subplots

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,

                          subplot_titles=["Bigram plots of Positive tweets", 

                                          "Bigram plots of Neutral tweets",

                                          "Bigram plots of Negative tweets"

                                          ])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)





fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Bigram Plots")

iplot(fig, filename='word-plots')
for sent in train0_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'blue')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'green')



freq_dict = defaultdict(int)

for sent in train2_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'violet')









# Creating two subplots

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04, horizontal_spacing=0.05,

                          subplot_titles=["Tri-gram plots of Positive tweets", 

                                          "Tri-gram plots of Neutral tweets",

                                          "Tri-gram plots of Negative tweets"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

iplot(fig, filename='word-plots')
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))

train['select_num_words'] = train["selected_text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))

train['select_num_unique_words'] = train["selected_text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))

train['select_num_chars'] = train["selected_text"].apply(lambda x: len(str(x)))
fig = go.Figure()

fig.add_trace(go.Histogram(x=train['num_words'],name = 'Number of words in text of train data'))

fig.add_trace(go.Histogram(x=test['num_words'],name = 'Number of words in text of test data'))

fig.add_trace(go.Histogram(x=train['select_num_words'],name = 'Number of words in selected text'))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig_ = go.Figure()

fig_.add_trace(go.Histogram(x=train['num_chars'],name = 'Number of characters in text of train data',marker = dict(color = 'rgba(222, 111, 33, 0.8)')))

fig_.add_trace(go.Histogram(x=test['num_chars'],name = 'Number of characters in text of test data',marker = dict(color = 'rgba(33, 1, 222, 0.8)')))

fig_.add_trace(go.Histogram(x=train['select_num_chars'],name = 'Number of characters in selected text',marker = dict(color = 'rgba(108, 25, 7, 0.8)')))



# Overlay both histograms

fig_.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig_.update_traces(opacity=0.75)

fig_.show()
fig_ = go.Figure()

fig_.add_trace(go.Histogram(x=train['num_unique_words'],name = 'Number of unique words in text of train data',marker = dict(color = 'rgba(222, 1, 3, 0.8)')))

fig_.add_trace(go.Histogram(x=test['num_unique_words'],name = 'Number of unique words in text of test data',marker = dict(color = 'rgba(3, 221, 2, 0.8)')))

fig_.add_trace(go.Histogram(x=train['select_num_unique_words'],name = 'Number of unique words in selected text',marker = dict(color = 'rgba(1, 2, 237, 0.8)')))



# Overlay both histograms

fig_.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig_.update_traces(opacity=0.75)

fig_.show()
MAX_LEN = 128

VALID_BATCH_SIZE = 8

BERT_PATH = "../input/roberta-base/"

MODEL_PATH = "model.bin"

TRAINING_FILE = "../input/train.csv"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{BERT_PATH}/vocab.json", 

    merges_file=f"{BERT_PATH}/merges.txt", 

    lowercase=True,

    add_prefix_space=True

)
class TweetModel(nn.Module):

    def __init__(self):

        super(TweetModel, self).__init__()

        self.bert = transformers.RobertaModel.from_pretrained(BERT_PATH)

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

model.load_state_dict(torch.load("../input/roberta-weights/roberta_model_1.bin"))

model.eval()



model1 = TweetModel()

model1.to(device)

model1 = nn.DataParallel(model1)

model1.load_state_dict(torch.load("../input/roberta-weights/roberta_model_2.bin"))

model1.eval()
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

        

        targets = [0] * len(tok_tweet_ids)

        target_idx = []

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):

            if sum(char_targets[offset1: offset2]) > 0:

                targets[j] = 1

                target_idx.append(j)



        

        targets_start = [0] * len(targets)

        targets_end = [0] * len(targets)



        non_zero = np.nonzero(targets)[0]

        if len(non_zero) > 0:

            targets_start[non_zero[0]] = 1

            targets_end[non_zero[-1]] = 1



        # check padding:

        # <s> pos/neg/neu </s> </s> tweet </s>

        if len(tok_tweet_tokens) > self.max_len - 5:

            tok_tweet_tokens = tok_tweet_tokens[:self.max_len - 5]

            tok_tweet_ids = tok_tweet_ids[:self.max_len - 5]

            targets_start = targets_start[:self.max_len - 5]

            targets_end = targets_end[:self.max_len - 5]





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



        padding_length = self.max_len - len(tok_tweet_ids)

        

        tok_tweet_ids = tok_tweet_ids + ([1] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        targets_start = targets_start + ([0] * padding_length)

        targets_end = targets_end + ([0] * padding_length)



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

df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

df_test.loc[:, "selected_text"] = df_test.text.values
test_dataset = TweetDataset(

        tweet=df_test.text.values,

        sentiment=df_test.sentiment.values,

        selected_text=df_test.selected_text.values

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

fin_outputs_start1 = []

fin_outputs_end1 = []

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

        

        outputs_start1, outputs_end1 = model1(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        



        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())

        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

        

        fin_outputs_start1.append(torch.sigmoid(outputs_start1).cpu().detach().numpy())

        fin_outputs_end1.append(torch.sigmoid(outputs_end1).cpu().detach().numpy())

        

        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())



        fin_orig_sentiment.extend(sentiment)

        fin_orig_selected.extend(orig_selected)

        fin_orig_tweet.extend(orig_tweet)



fin_outputs_start = np.vstack(fin_outputs_start)

fin_outputs_end = np.vstack(fin_outputs_end)



fin_outputs_start1 = np.vstack(fin_outputs_start1)

fin_outputs_end1 = np.vstack(fin_outputs_end1)



fin_outputs_start = (fin_outputs_start + fin_outputs_start1) / 2

fin_outputs_end = (fin_outputs_end + fin_outputs_end1) / 2





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



    if sentiment_val == "neutral" or len(original_tweet.split()) < 4:

        filtered_output = original_tweet



    all_outputs.append(filtered_output.strip())
sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = all_outputs

sample.to_csv("submission.csv", index=False)
sample.head()