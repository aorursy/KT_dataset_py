import re

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = '/kaggle/input/tweet-sentiment-extraction/train.csv'

test_path = '/kaggle/input/tweet-sentiment-extraction/test.csv'

submit_path = '/kaggle/input/tweet-sentiment-extraction/sample_submission.csv'
df = pd.read_csv(train_path)

df.columns
df.dropna(axis = 0, how ='any',inplace=True)



df.text = df.text.str.strip()

df.selected_text = df.selected_text.str.strip()
import nltk

from nltk.tokenize import TreebankWordTokenizer

# from nltk.tokenize import sent_tokenize, word_tokenize

twt = TreebankWordTokenizer()
df['span_list'] = df.text.apply(twt.span_tokenize)

df[['text', 'span_list']].head()
def get_iob(text, selected_text, twt=twt, sentiment=None):

    """

    :param text: text

    :param selected_text: selected_text

    :param twt: Tokenizer that has `span_tokenize()` function

    :param sentiment: add sentiment info to IB tag e.g. `B-positive`, `I-neutral`

    :returns: iob string

    """

    sentiment_dict = {'positive':'POS', 'negative':'NEG', 'neutral': 'NEU'}



    start, end = re.search(re.escape(selected_text), text).span()

    # list of (start_idx, stop_idx)

    span_list = twt.span_tokenize(text)

    

    iob_list = []

    for start_sp, end_sp in span_list:

        iob_tag = 'O'

        if start_sp == start:

            iob_tag = 'B'

        elif start < start_sp and end_sp <= end:

            iob_tag = 'I'

            

        if sentiment is not None and iob_tag!='O':

            iob_tag += '-{}'.format(sentiment_dict[sentiment])

        iob_list.append(iob_tag)

    return ' '.join(iob_list)

    



def get_iob_format_from_row(row, twt=twt, add_sentiment=False):

    if add_sentiment:

        return get_iob(row.text, row.selected_text, twt=twt, sentiment=row.sentiment)

    return get_iob(row.text, row.selected_text, twt=twt)
df.head().apply(get_iob_format_from_row,axis=1)
# iob with sentiment info

df.head().apply(lambda x:get_iob_format_from_row(x, add_sentiment=True), axis=1)
df_pn = df.query('sentiment!="neutral"').copy()

df_pn['iob'] = df_pn.apply(lambda x:get_iob_format_from_row(x, add_sentiment=True), axis=1)

df_pn[['text','iob']].head()
word_data = df_pn['text'].str.split()

iob_data = df_pn['iob'].str.split()
df_test = pd.read_csv(test_path)

df_test.columns
df_test.text = df_test.text.str.strip()

# df_test_pn = df_test.query('sentiment!="neutral"').copy()



# twt = TreebankWordTokenizer()

df_test['text_list'] = df_test.text.apply(lambda x: [x[start_i:end_i] for start_i, end_i in twt.span_tokenize(x)])



# df_test_pn['pos_list'] = df_test_pn.text.apply(lambda x: nltk.pos_tag([x[start_i:end_i] for start_i, end_i in twt.span_tokenize(x)]))

df_test.head()