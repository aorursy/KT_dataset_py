# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from plotly import tools

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import matplotlib

from tqdm import tqdm

from statistics import *

import plotly.express as px

from plotly.subplots import make_subplots

import nltk

from nltk.corpus import stopwords

import string

import re

from nltk.tokenize import word_tokenize

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import seaborn as sns

import json

from statistics import *

from sklearn.feature_extraction.text import CountVectorizer

from copy import deepcopy

nltk.download('stopwords')

stop=set(stopwords.words('english'))
def read_train():

    train=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

    train['text']=train['text'].astype(str)

    train['selected_text']=train['selected_text'].astype(str)

    return train



def read_test():

    test=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

    test['text']=test['text'].astype(str)

    return test



def read_submission():

    test=pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

    return test

    

train_df = read_train()

test_df = read_test()

submission_df = read_submission()

def remove_stopwords(text):

        if text is not None:

            tokens = [x for x in word_tokenize(text) if x not in stop]

            return " ".join(tokens)

        else:

            return None



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)





def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)





def remove_punct(text):

    exclude = set(string.punctuation)

    s = ''.join(ch for ch in text if ch not in exclude)

    return s



def clean_df(df, train=True):

    df["dirty_text"] = df['text']

    

    

    df["text"] = df['text'].apply(lambda x : x.lower())

    

    df['text']=df['text'].apply(lambda x: remove_emoji(x))

        

    df['text']=df['text'].apply(lambda x : remove_URL(x))

        

    df['text']=df['text'].apply(lambda x : remove_html(x))

        

    df['text'] =df['text'].apply(lambda x : remove_stopwords(x)) 

    

    df['text']=df['text'].apply(lambda x : remove_punct(x))

    

    df.text = df.text.replace('\s+', ' ', regex=True)

    

    if train:

        df["selected_text"] = df['selected_text'].apply(lambda x : x.lower())

        df['selected_text']=df['selected_text'].apply(lambda x: remove_emoji(x))

        df['selected_text']=df['selected_text'].apply(lambda x : remove_URL(x))

        df['selected_text']=df['selected_text'].apply(lambda x : remove_html(x))

        df['selected_text'] =df['selected_text'].apply(lambda x : remove_stopwords(x))

        df['selected_text']=df['selected_text'].apply(lambda x : remove_punct(x))

        df.selected_text = df.selected_text.replace('\s+', ' ', regex=True)

    

    return df



#train_df = clean_df(train_df)

#test_df = clean_df(test_df, train=False)
def plot_distrib_train_test(train, test):

    fig=make_subplots(1,2,subplot_titles=('Training set','Test set'))

    x=train.sentiment.value_counts()

    fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['orange','green','red'],name=''),row=1,col=1)

    x=test.sentiment.value_counts()

    fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['orange','green','red'],name=''),row=1,col=2)

    fig.show()



def show_word_distrib(df, target="positive", field="text", top_N=10, selected=True):

    fig = plt.figure()

    

    txt = df[df['sentiment']==target][field].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')

    words = nltk.tokenize.word_tokenize(txt)

    words_except_stop_dist = nltk.FreqDist(words) 

    

    rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),

                        columns=['Word', 'Frequency Text']).set_index('Word')

    print(rslt)

    #matplotlib.style.use('ggplot')

    ax1 = fig.add_subplot()

    rslt.plot.bar(rot=0, ax=ax1)

    

    if selected:

        txt = df[df['sentiment']==target]["selected_text"].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')

        words = nltk.tokenize.word_tokenize(txt)

        words_except_stop_dist = nltk.FreqDist(words) 



        rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),

                            columns=['Word', 'Frequency Selected Text']).set_index('Word')

        

        print(rslt)

        #matplotlib.style.use('ggplot')

        ax2 = fig.add_subplot()

        rslt.plot.bar(rot=0, ax=ax2)

    plt.show()

    

    

def general_stats(train, test):

    n_train = len(train)

    n_test = len(test)

    

    print("Number of train samples: {}".format(n_train))

    print("Number of test samples: {}".format(n_test))

    plot_distrib_train_test(train, test)

    

    print("Word distribution of 'text' and 'selected_text' in Training set for positive samples")

    show_word_distrib(train, target="positive")

    

    print("Word distribution of 'text' and 'selected_text' in Training set for neutral samples")

    show_word_distrib(train, target="neutral")

    

    print("Word distribution of 'text' and 'selected_text' in Training set for negative samples")

    show_word_distrib(train, target="negative")

    

    

    print("Word distribution of 'text' in Test set for positive samples")

    show_word_distrib(test, target="positive", selected=False)

    

    print("Word distribution of 'text' in Test set for neutral samples")

    show_word_distrib(test, target="neutral", selected=False)

    

    print("Word distribution of 'text' in Test set for negative samples")

    show_word_distrib(test, target="negative", selected=False)

    



def plot_hist_classes(df, to_plot, _header, col="text"):

    fig,(ax1,ax2, ax3)=plt.subplots(1,3,figsize=(10,5))

    

    df_len = to_plot(df, "negative", col=col)

    ax1.hist(df_len,color='red')

    ax1.set_title('Negative Tweets')

    

    df_len = to_plot(df, "positive", col=col)

    ax2.hist(df_len,color='green')

    ax2.set_title('Positive Tweets')

    

    df_len = to_plot(df, "neutral", col=col)

    ax3.hist(df_len,color='orange')

    ax3.set_title('Neutral Tweets')

    

    

    fig.suptitle(_header)

    plt.show()

    plt.close()

    



def average_word_lenght(df,col="text"):

    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,5))

    

    word=df[df['sentiment']=="negative"][col].str.split().apply(lambda x : [len(i) for i in x])

    sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')

    ax1.set_title('Negative')

    

    word=df[df['sentiment']=="positive"][col].str.split().apply(lambda x : [len(i) for i in x])

    sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')

    ax2.set_title('Positive')

    

    word=df[df['sentiment']=="neutral"][col].str.split().apply(lambda x : [len(i) for i in x])

    sns.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='orange')

    ax2.set_title('Neutral')

    

    fig.suptitle('Average word length in each tweet')

    plt.show()

    



def unique_words(df, col="text", title="Distribution of number of unique words"):

    fig,ax=plt.subplots(1,3,figsize=(12,7))

    colors = {

        "positive": "green",

        "negative": "red",

        "neutral": "orange"

    }

    for _, i in enumerate(["positive", "negative", "neutral"]):

        new=df[df['sentiment']==i][col].map(lambda x: len(set(x.split())))

        sns.distplot(new.values,ax=ax[_],color=colors[i])

        ax[_].set_title(i)

    fig.suptitle(title)

    fig.show()

    

def get_top_ngram(corpus, n=None):

        vec = CountVectorizer(ngram_range=(n, n),stop_words=stop).fit(corpus)

        bag_of_words = vec.transform(corpus)

        sum_words = bag_of_words.sum(axis=0) 

        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

        return words_freq[:20]

    

def plot_n_grams(df, size=2, 

                 title="Common bigrams in selected text",

                 column="text"):

    colors = {

        "positive": "green",

        "negative": "red",

        "neutral": "orange"

    }

    

    fig,ax=plt.subplots(1,3,figsize=(15,10))

    for _, i in enumerate(["positive", "negative", "neutral"]):

        new=df[df['sentiment']==i][column]

        top_n_bigrams=get_top_ngram(new,size)[:20]

        x,y=map(list,zip(*top_n_bigrams))

        sns.barplot(x=y,y=x,ax=ax[_],color=colors[i])

        ax[_].set_title(i)

    

    fig.suptitle(title)

    fig.show()

   



general_stats(train_df, test_df)



def to_plot_chars(df, _target, col="text"):

    return df[df['sentiment']==_target][col].str.len()

plot_hist_classes(train_df, to_plot_chars, _header='Characters Lenght Distribution in Training Tweets "text"')

plot_hist_classes(test_df, to_plot_chars, _header='Characters Lenght Distribution in Test Tweets "text"')

plot_hist_classes(train_df, to_plot_chars, _header='Characters Lenght Distribution in Training Tweets "selected_text"', col='selected_text')



def to_plot_word(df, _target, col="text"):

    return df[df['sentiment']==_target][col].str.split().map(lambda x: len(x))

plot_hist_classes(train_df, to_plot_word, _header='Sentence Lenght Distribution in Training Tweets "text" column')

plot_hist_classes(train_df, to_plot_word, _header='Sentence Lenght Distribution in Training Tweets "selected_text" column', col='selected_text')

plot_hist_classes(test_df, to_plot_word, _header='Sentence Lenght Distribution in Test Tweets "text" column')



print("Average word lenght in Training Tweets 'text' column")

average_word_lenght(train_df)

print("Average word lenght in Training Tweets 'selected_text' column")

average_word_lenght(train_df, col="selected_text")

print("Average word lenght in Test Tweets 'text' column")

average_word_lenght(test_df)



unique_words(train_df, title="Distribution of number of unique words in Training samples for 'text' column")

unique_words(train_df, col="selected_text",title="Distribution of number of unique words in Training samples for 'selected_text' column")

unique_words(test_df, title="Distribution of number of unique words in Test samples for 'text' column")







plot_n_grams(train_df, size=2, 

                 title="Common bigrams in text for the training set",

                 column="text")



plot_n_grams(train_df, size=2, 

                 title="Common bigrams in selected text for the training set",

                 column="selected_text")



plot_n_grams(test_df, size=2, 

                 title="Common bigrams in text for the test set",

                 column="text")





plot_n_grams(train_df, size=3, 

                 title="Common tri-grams in text for the training set",

                 column="text")



plot_n_grams(train_df, size=3, 

                 title="Common tri-grams in selected_text for the training set",

                 column="selected_text")



plot_n_grams(test_df, size=3, 

                 title="Common tri-grams in text for the test set",

                 column="text")



def analyze_neutral_lenght(train_df):

    neutral_df = train_df[train_df["sentiment"] == "neutral"]

    equal_selected_with_text = neutral_df[neutral_df['selected_text'] == neutral_df["text"]]

    

    print("Total number of neutral: {}".format(len(neutral_df)))

    print("Neutral with text equal to selected text: {}".format(len(equal_selected_with_text)))

    print("Ratio: {}".format(len(equal_selected_with_text)/len(neutral_df)))

    

    count = train_df.text.str.count(' ') <= 2

    less_than_T = train_df.loc[count]

    equal_selected_with_text = less_than_T[less_than_T['selected_text'] == less_than_T["text"]]

    print("Total number of tweets with less than 3 words per text: {}".format(len(neutral_df)))

    print("Neutral with text equal to selected text: {}".format(len(equal_selected_with_text)))

    print("Ratio: {}".format(len(equal_selected_with_text)/len(less_than_T)))

   





def post_process(submission_df, test_df):



    index_to_selected_text = {}

    for i, row in test_df.iterrows():

        _id = row[0]

        text = row[1]

        sentiment = row[2]

        if len(text.split(" ")) <= 3 or sentiment == "neutral":

            index_to_selected_text[_id] = text

    

    submission_rows = submission_df.to_dict("records")

    new_rows = []

    for row in submission_rows:

        _id = row['textID']

        if _id in index_to_selected_text:

            new_row = deepcopy(row)

            new_row['selected_text'] = index_to_selected_text[_id]

        else:

            new_row = row

        

        new_rows.append(new_row)



    return pd.DataFrame(new_rows)





train_df = read_train()

test_df = read_test()

submission_df = read_submission()

analyze_neutral_lenght(train_df)

def analyze_cleaning(df, cleaning_lambda, field="selected_text"):

    def diff_strings(row):

        count = {}

        A = row[0]

        B = row[1]



        if A is None:

            A = "TEMP"

        

        if B is None:

            B = "TEMP"

        

        for word in A.split(): 

            count[word] = count.get(word, 0) + 1



        for word in B.split(): 

            count[word] = count.get(word, 0) + 1





        diff = [word for word in count if count[word] == 1]

        _max_len = max(len(A), len(B))

        return len(diff)/_max_len

        

    to_analyze = df[[field]]

    to_analyze['cleaned_selected_text'] = to_analyze[field]

    to_analyze["diff_ratio"] = None

    to_analyze['cleaned_selected_text']=to_analyze['cleaned_selected_text'].apply(lambda x: cleaning_lambda(x))

    

    to_analyze["diff_ratio"] = to_analyze.apply(lambda x: diff_strings(x), axis=1)

    print("Cleaning function used: {}".format(cleaning_lambda.__name__))

    print("Average difference ratio: {}".format(to_analyze["diff_ratio"].mean()))

    print("")





train_df = read_train()

test_df = read_test()

cleaning_functions = [remove_stopwords, remove_URL, remove_html, remove_emoji, remove_punct]

for func in cleaning_functions:

    analyze_cleaning(train_df, func, field="selected_text")
"""

from tqdm import tqdm





def expand(text, selected, steps=2):

    _text = text.split()

    _selected = selected.split()

    

    _start = 0

    for i,_t in enumerate(_text):

        to_check = " ".join(_text[i: i + len(_selected)])

        if to_check == selected:

            break

        _start += 1

        

    #_start = text.find(selected)

    _end = _start + len(_selected)

    

    substrings = set([text])

    _low = _start

    _high = _end

    for i in range(steps):

        _low = _low - i if _low - i > 0 else 0

        _high = _high + i if _high + 1 < (len(_text) - 1) else len(_text) - 1

        

        to_add = " ".join(_text[_low:_high])

        if to_add != selected:

            substrings.add(to_add)

    

    return list(substrings)





train_df = read_train()



rows = train_df.to_dict("records")

new_rows = []

tot_text = set ()

for row in tqdm(rows):

    if row["sentiment"] != "neutral" and len(row["text"].split()) > 3:

        _text = row['text']

        _sel = row["selected_text"]

        add = expand(_text, _sel)

        for a in add:

            if _sel in a:

                if a != _text:

                    new_row = deepcopy(row)

                    new_row["text"] = a

                    new_rows.append(new_row)

    new_rows.append(row)





train_df = pd.DataFrame(new_rows)

"""
"""

def whole_text_classifier(test):

    test["selected_text"] = test['text']

    test = test[["textID", "selected_text"]]

    test.to_csv('whole_text_submission.csv',index=False)

    

whole_text_classifier(test_df)

"""
import os

import gc

import numpy as np 

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import BertTokenizer,BertConfig,TFBertModel

from tqdm import tqdm

tqdm.pandas()

BERT_PATH = '/kaggle/input/bert-base-uncased-huggingface-transformer/'

def find_offset(x,y):

    """find offset in fail scenarios (only handles the start fail as of now)"""

    x_str = ' '.join(x)

    y_str = ' '.join(y)

    idx0=0

    ## code snippet from this https://www.kaggle.com/abhishek/text-extraction-using-bert-w-sentiment-inference

    for ind in (i for i, e in enumerate(x_str) if e == y_str[0]):

        if (x_str[ind: ind+len(y_str)] == y_str) or (x_str[ind: ind+len(y_str.replace(' ##',''))] == y_str.replace(' ##','')):

            idx0 = ind

            idx1 = ind + len(y_str) - 1

            break

    t = 0

    for offset,i in enumerate(x):

        if t +len(i)+1>idx0:

            break

        t = t+len(i)+1

    return offset



def create_targets(df, tokenizer):

    df['t_text'] = df['text'].apply(lambda x: tokenizer.tokenize(str(x)))

    df['t_selected_text'] = df['selected_text'].apply(lambda x: tokenizer.tokenize(str(x)))

    def func(row):

        x,y = row['t_text'],row['t_selected_text'][:]

        _offset = 0

        for offset in range(len(x)):

            _offset = offset

            d = dict(zip(x[offset:],y))

            #when k = v that means we found the offset

            check = [k==v for k,v in d.items()]

            if all(check)== True:

                break 

        targets = [0]*_offset + [1]*len(y) + [0]* (len(x)-_offset-len(y))

        

        ## should be same if not its a fail scenario because of  start or end issue 

        if len(targets) != len(x):

            offset = find_offset(x,y)

            targets = [0]*offset + [1]*len(y) + [0] * (len(x)-offset-len(y))

        return targets

    df['targets'] = df.apply(func,axis=1)

    return df



def _convert_to_transformer_inputs(text, tokenizer, max_sequence_length):

    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,

            add_special_tokens=True,

            max_length=length,

            truncation_strategy=truncation_strategy)

        

        input_ids =  inputs["input_ids"]

        input_masks = [1] * len(input_ids)

        input_segments = inputs["token_type_ids"]

        padding_length = length - len(input_ids)

        padding_id = tokenizer.pad_token_id

        input_ids = input_ids + ([padding_id] * padding_length)

        input_masks = input_masks + ([0] * padding_length)

        input_segments = input_segments + ([0] * padding_length)

        

        return [input_ids, input_masks, input_segments]

    

    input_ids, input_masks, input_segments = return_id(text, None, 'longest_first', max_sequence_length)

    return [input_ids, input_masks, input_segments]



def compute_input_arrays(df, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []

    for _, instance in tqdm(df.iterrows()):

        t = str(instance.text)



        ids, masks, segments= _convert_to_transformer_inputs(t,tokenizer, max_sequence_length)

        

        input_ids.append(ids)

        input_masks.append(masks)

        input_segments.append(segments)

        

    return [np.asarray(input_ids, dtype=np.int32), 

            np.asarray(input_masks, dtype=np.int32), 

            np.asarray(input_segments, dtype=np.int32)]



def compute_output_arrays(df, columns):

    return np.asarray(df[columns].values.tolist())



def create_model():

    id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    attn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = BertConfig() 

    config.output_hidden_states = True 

    

    bert_model = TFBertModel.from_pretrained(

        BERT_PATH+'bert-base-uncased-tf_model.h5', config=config)

    

    _,_, hidden_states = bert_model(id, attention_mask=mask, token_type_ids=attn)



    h12 = tf.reshape(hidden_states[-1][:,0],(-1,1,768))

    h11 = tf.reshape(hidden_states[-2][:,0],(-1,1,768))

    h10 = tf.reshape(hidden_states[-3][:,0],(-1,1,768))

    h09 = tf.reshape(hidden_states[-4][:,0],(-1,1,768))



    concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])

    x = tf.keras.layers.GlobalAveragePooling1D()(concat_hidden)

    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(MAX_TARGET_LEN, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[id, mask, attn], outputs=x)

    return model



def convert_pred_to_text(df,pred):

    temp_output = []

    for idx,p in enumerate(pred):

        indexes = np.where(p>0.5)

        current_text = df['t_text'][idx]

        if len(indexes[0])>0:

            start = indexes[0][0]

            end = indexes[0][-1]

        else:

            start = 0

            end = len(current_text)



        ### < was written previously but it should be > 

        ### (means model goes into padding tokens then restrict till the end of text)

        ### Thanks Davide Romano for pointing this out

        if end >= len(current_text):

            end = len(current_text)

        temp_output.append(' '.join(current_text[start:end+1]))

    return temp_output



def correct_op(row):

    placeholder = row['temp_output']

    for original_token in str(row['text']).split():

        token_str = ' '.join(tokenizer.tokenize(original_token))

        placeholder = placeholder.replace(token_str,original_token,1)

    return placeholder



def replacer(row):

    if row['sentiment'] == 'neutral':

        return row['text']

    else:

        return row['temp_output2']

""" 

tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-uncased-vocab.txt')

MAX_TARGET_LEN = MAX_SEQUENCE_LENGTH = 108

train_df = create_targets(train_df, tokenizer)

test_df['t_text'] = test_df['text'].apply(lambda x: tokenizer.tokenize(str(x)))

train_df['targets'] = train_df['targets'].apply(lambda x :x + [0] * (MAX_TARGET_LEN-len(x)))

outputs = compute_output_arrays(train_df,'targets')

inputs = compute_input_arrays(train_df, tokenizer, MAX_SEQUENCE_LENGTH)

test_inputs = compute_input_arrays(test_df, tokenizer, MAX_SEQUENCE_LENGTH)



train_inputs = inputs

train_outputs = outputs



del inputs,outputs

K.clear_session()

model = create_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(loss='binary_crossentropy', optimizer=optimizer)

if not os.path.exists('/kaggle/input/tweet-finetuned-bert-v2/finetuned_bert.h5'):

    # Training done in another private kernel

    model.fit(train_inputs, train_outputs, epochs=10, batch_size=32)

    model.save_weights(f'finetuned_bert.h5')

else:

    model.load_weights('/kaggle/input/tweet-finetuned-bert-v2/finetuned_bert.h5')

    

threshold = 0.5

predictions = model.predict(test_inputs, batch_size=32, verbose=1)

pred = np.where(predictions>threshold,1,0)

test_df['temp_output'] = convert_pred_to_text(test_df,pred)

gc.collect()

test_df['temp_output2'] = test_df.progress_apply(correct_op,axis=1)

submission_df['selected_text'] = test_df['temp_output2']

submission_df['selected_text'] = submission_df['selected_text'].str.replace(' ##','')

submission_df.to_csv('submission.csv',index=False)

submission_df.head(10)

"""


!mkdir -p data

!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q

from simpletransformers.question_answering import QuestionAnsweringModel

from copy import deepcopy



use_cuda = True



    

def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1



def do_qa_train(train):



    output = {}

    output['version'] = 'v1.0'

    output['data'] = []

    paragraphs = []

    for line in train:

        context = line[1]



        qas = []

        question = line[-1]

        qid = line[0]

        answers = []

        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:

            answers.append({'answer_start': answer_start, 'text': answer.lower()})

            break



        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

    return paragraphs



def do_qa_test(test):

    paragraphs = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-1]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    return paragraphs



train_df = read_train()

test_df = read_test()

submission_df_distil = read_submission()





train = np.array(train_df)

test = np.array(test_df)

qa_train = do_qa_train(train)





with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)



output = {}

output['version'] = 'v1.0'

output['data'] = []



qa_test = do_qa_test(test)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)

    

MODEL_PATH = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

model = QuestionAnsweringModel('distilbert', 

                               MODEL_PATH, 

                              args={"reprocess_input_data": True,

                               "overwrite_output_dir": True,

                               "learning_rate": 8e-05,

                               "num_train_epochs": 3,

                               "max_seq_length": 192,

                               "weight_decay": 0.001,

                               "doc_stride": 64,

                               "save_model_every_epoch": False,

                               "fp16": False,

                               "do_lower_case": True,

                                 'max_query_length': 8,

                               'max_answer_length': 150,

                                    "use_cuda":True

                                    },

                              use_cuda=use_cuda)



model.train_model('data/train.json')

predictions = model.predict(qa_test)

predictions_df = pd.DataFrame.from_dict(predictions)



submission_df_distil['selected_text'] = predictions_df['answer']

submission_df_distil = post_process(submission_df_distil, test_df)







#submission_df_distil.to_csv('submission.csv', index=False)



"""

def do_qa_test2(test):

    paragraphs = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-2]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    return paragraphs





train_df = read_train()



train_df.dropna(inplace=True)



train_size = int(0.70 * len(train_df))



sub_train_df = train_df[:train_size]

sub_test_df = train_df[train_size:]



sub_test_df["predicted"] = None



train = np.array(sub_train_df)

test = np.array(sub_test_df)

qa_train = do_qa_train(train)





with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)



output = {}

output['version'] = 'v1.0'

output['data'] = []



qa_test = do_qa_test2(test)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)

    

MODEL_PATH = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

use_cuda = True





parameters = {

    "1": {

                   "reprocess_input_data": True,

                   "overwrite_output_dir": True,

                   "learning_rate": 8e-05,

                   "num_train_epochs": 1,

                   "adam_epsilon": 1e-08,

                   "max_seq_length": 192,

                   "weight_decay": 0.01,

                   "doc_stride": 64,

                   "save_model_every_epoch": False,

                   "fp16": False,

                   "do_lower_case": True,

                    "warmup_steps": 200

                },

    "2": {

                   "reprocess_input_data": True,

                   "overwrite_output_dir": True,

                   "learning_rate": 8e-05,

                   "num_train_epochs": 1,

                   "adam_epsilon": 1e-08,

                   "max_seq_length": 192,

                   "weight_decay": 0.01,

                   "doc_stride": 64,

                   "save_model_every_epoch": False,

                   "fp16": False,

                   "do_lower_case": True,

                    "warmup_steps": 200,

                    "warmup_ratio": 0.01

                },

    "3": {

                   "reprocess_input_data": True,

                   "overwrite_output_dir": True,

                   "learning_rate": 8e-05,

                   "num_train_epochs": 1,

                   "adam_epsilon": 1e-08,

                   "max_seq_length": 192,

                   "weight_decay": 0.01,

                   "doc_stride": 64,

                   "save_model_every_epoch": False,

                   "fp16": False,

                   "do_lower_case": True,

                    "warmup_steps": 200,

                    "warmup_ratio": 0.1

                },

    "4": {

                   "reprocess_input_data": True,

                   "overwrite_output_dir": True,

                   "learning_rate": 8e-05,

                   "num_train_epochs": 1,

                   "adam_epsilon": 1e-08,

                   "max_seq_length": 192,

                   "weight_decay": 0.01,

                   "doc_stride": 64,

                   "save_model_every_epoch": False,

                   "fp16": False,

                   "do_lower_case": True,

                    "warmup_steps": 200,

                    "warmup_ratio": 0.5

                }

    }

def jaccard(row): 

    a = set(row[1].lower().split()) 

    b = set(row[2].lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



scores = {k: None for k in list(parameters.keys())}



def post_process2(submission_df, test_df):



    index_to_selected_text = {}

    for i, row in test_df.iterrows():

        _id = row[0]

        text = row[1]

        sentiment = row[3]

        if len(text.split(" ")) <= 3 or sentiment == "neutral":

            index_to_selected_text[_id] = text

    

    submission_rows = submission_df.to_dict("records")

    new_rows = []

    for row in submission_rows:

        _id = row['textID']

        if _id in index_to_selected_text:

            new_row = deepcopy(row)

            new_row['selected_text'] = index_to_selected_text[_id]

        else:

            new_row = row

        

        new_rows.append(new_row)



    return pd.DataFrame(new_rows)





def align_test_pred(predictions_df, test_df):

    

    pred_text = {}

    for i, row in predictions_df.iterrows():

        _id = row[0]

        _pred = row[1]

        pred_text[_id] = _pred

        

    test_rows = test_df.to_dict("records")

    new_rows = []

    for t in test_rows:

        _id = t['textID']

        if _id in pred_text:

            new_row = t

            new_row['predicted'] = pred_text[_id]

            new_rows.append(new_row)

    

    new_df = pd.DataFrame(new_rows)

    return new_df

    

for key, param in parameters.items():

    sub_test_df = train_df[train_size:]



    sub_test_df["predicted"] = None

    

    model = QuestionAnsweringModel('distilbert', 

                               MODEL_PATH, 

                              args=param,

                              use_cuda=use_cuda)



    model.train_model('data/train.json')

    predictions = model.predict(qa_test)

    predictions_df = pd.DataFrame.from_dict(predictions)

    

    sub_test_df = align_test_pred(predictions_df, sub_test_df)

    

    sub_test_df = sub_test_df[["textID","selected_text", "predicted", "sentiment"]]

    

    sub_test_df = post_process2(sub_test_df, sub_test_df)

    

    ## Based on discussion https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140235

    def f(selected):

         return " ".join(set(selected.lower().split()))



    sub_test_df.predicted = sub_test_df.predicted.map(f)

    sub_test_df['jaccard'] = sub_test_df.apply(lambda x: jaccard(x), axis=1)

    average_jaccard = sub_test_df['jaccard'].mean()

    scores[key] = average_jaccard

    print("Params tested: {} - {}".format(key, json.dumps(param, indent=3)))

    print("Jaccard: {}".format(average_jaccard))

    print(":::::")





print(scores)





"""


!mkdir -p data

!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q

from simpletransformers.question_answering import QuestionAnsweringModel

from copy import deepcopy



use_cuda = True

    

def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1



def do_qa_train(train):



    output = {}

    output['version'] = 'v1.0'

    output['data'] = []

    paragraphs = []

    for line in train:

        context = line[1]



        qas = []

        question = line[-1]

        qid = line[0]

        answers = []

        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:

            answers.append({'answer_start': answer_start, 'text': answer.lower()})

            break



        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

    return paragraphs



def do_qa_test(test):

    paragraphs = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-1]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    return paragraphs



train_df = read_train()

test_df = read_test()

submission_df_albert = read_submission()





train = np.array(train_df)

test = np.array(test_df)

qa_train = do_qa_train(train)





with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)



output = {}

output['version'] = 'v1.0'

output['data'] = []



qa_test = do_qa_test(test)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)

    

MODEL_PATH = '/kaggle/input/pretrained-albert-pytorch/albert-base-v2/'

model = QuestionAnsweringModel('albert', 

                               MODEL_PATH, 

                               args={"reprocess_input_data": True,

                               "overwrite_output_dir": True,

                               "learning_rate": 8e-05,

                               "num_train_epochs": 3,

                               "max_seq_length": 192,

                               "weight_decay": 0.001,

                               "doc_stride": 64,

                               "save_model_every_epoch": False,

                               "fp16": False,

                               "do_lower_case": True,

                                 'max_query_length': 8,

                               'max_answer_length': 150

                                },

                              use_cuda=use_cuda)



model.train_model('data/train.json')

predictions = model.predict(qa_test)

predictions_df = pd.DataFrame.from_dict(predictions)



submission_df_albert['selected_text'] = predictions_df['answer']

submission_df_albert = post_process(submission_df_albert, test_df)



#submission_df_albert.to_csv('submission.csv', index=False)





!mkdir -p data

!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q

from simpletransformers.question_answering import QuestionAnsweringModel

from copy import deepcopy



use_cuda = True

    

def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1



def do_qa_train(train):



    output = {}

    output['version'] = 'v1.0'

    output['data'] = []

    paragraphs = []

    for line in train:

        context = line[1]



        qas = []

        question = line[-1]

        qid = line[0]

        answers = []

        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:

            answers.append({'answer_start': answer_start, 'text': answer.lower()})

            break



        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

    return paragraphs



def do_qa_test(test):

    paragraphs = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-1]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    return paragraphs



train_df = read_train()

test_df = read_test()

submission_df_bert = read_submission()





train = np.array(train_df)

test = np.array(test_df)

qa_train = do_qa_train(train)





with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)



output = {}

output['version'] = 'v1.0'

output['data'] = []



qa_test = do_qa_test(test)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)

    

MODEL_PATH = '/kaggle/input/bert-base-uncased/'

model = QuestionAnsweringModel('bert', 

                               MODEL_PATH, 

                               args={"reprocess_input_data": True,

                               "overwrite_output_dir": True,

                               "learning_rate": 8e-05,

                               "num_train_epochs": 3,

                               "max_seq_length": 192,

                               "weight_decay": 0.001,

                               "doc_stride": 64,

                               "save_model_every_epoch": False,

                               "fp16": False,

                               "do_lower_case": True,

                                 'max_query_length': 8,

                               'max_answer_length': 150

                                },

                              use_cuda=use_cuda)



model.train_model('data/train.json')

predictions = model.predict(qa_test)

predictions_df = pd.DataFrame.from_dict(predictions)



submission_df_bert['selected_text'] = predictions_df['answer']

submission_df_bert = post_process(submission_df_bert, test_df)



#submission_df_albert.to_csv('submission.csv', index=False)





from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()



def merge_predictions(submission_distil, sumbission_albert, submission_bert):

    def merge(row):

        pred_1 = set(str(row[1]).split())

        pred_2 = set(str(row[2]).split())

        pred_3 = set(str(row[3]).split())

        res = []

        

        res = list(pred_1.intersection(pred_2, pred_3))

        if len(res) == 0:

            res = list(pred_1.union(pred_2, pred_3))

        

        return " ".join(res)

    

    def merge_compound(row):

        pred_1 = str(row[1])

        pred_2 = str(row[2])

        pred_3 = str(row[3])

        

        preds = [pred_1, pred_2, pred_3]

        

        comp_1 = abs(sentiment_analyzer.polarity_scores(pred_1).get("compound",0))

        comp_2 = abs(sentiment_analyzer.polarity_scores(pred_2).get("compound",0))

        comp_3 = abs(sentiment_analyzer.polarity_scores(pred_3).get("compound",0))

        

        comps = [comp_1, comp_2, comp_3]

        

        best_comp = comps.index(max(comps))

        return preds[best_comp]

    

    def capitalize_selected(df):

        rows = df.to_dict("records")

        new_rows = []

        

        for r in rows:

            text = r["text"]

            lower_text = text.lower()

            pred_distil = r["selected_text_distil"]

            pred_albert = r["selected_text_albert"]

            pred_bert = r["selected_text_bert"]

            

            new_pred_distil = text[lower_text.find(pred_distil):len(pred_distil)]

            new_pred_albert = text[lower_text.find(pred_albert):len(pred_albert)]

            new_pred_bert = text[lower_text.find(pred_bert):len(pred_bert)]

            

            new_row = {

                "textID": r['textID'],

                "selected_text_distil": new_pred_distil,

                "selected_text_albert": new_pred_albert,

                "selected_text_bert": new_pred_bert

            }

            new_rows.append(new_row)

        

        return pd.DataFrame(new_rows)

    

    sumbission_albert = sumbission_albert.rename({'selected_text': 'selected_text_albert'}, axis=1) 

    del sumbission_albert["textID"]

    submission_distil = submission_distil.rename({'selected_text': 'selected_text_distil'}, axis=1) 

    

    submission_bert = submission_bert.rename({'selected_text': 'selected_text_bert'}, axis=1) 

    del submission_bert["textID"]

    

    final_sub = pd.concat([submission_distil, sumbission_albert, submission_bert], axis=1)

    

    #final_sub = capitalize_selected(final_sub)

    final_sub["selected_text"] = final_sub.apply(lambda x: merge_compound(x), axis=1)

    

    del final_sub["selected_text_albert"]

    del final_sub["selected_text_distil"]

    del final_sub["selected_text_bert"]

    

    return final_sub



"""

submission_df_albert = read_submission()

submission_df_distil = read_submission()

submission_df_bert = read_submission()

test_df = read_test()



submission_df_albert.selected_text = [x.lower() for x in test_df.text]

submission_df_distil.selected_text = [x.lower() for x in test_df.text]

submission_df_bert.selected_text = [x.lower() for x in test_df.text]



"""



submission = merge_predictions(submission_df_distil, submission_df_albert, submission_df_bert)





submission.to_csv('submission.csv', index=False)
