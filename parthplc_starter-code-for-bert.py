def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
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
import numpy as np 

import pandas as pd 



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# XGBoost

import xgboost as xgb

from xgboost import XGBClassifier



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
#Training data

df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

df_sub = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

df_train.head()
df_test.head()
print(df_train.shape,df_test.shape)
df_train.info()
df_train.isnull().sum()

null_columns=df_train.columns[df_train.isnull().any()]

# print all rows with atleast one null values

print(df_train[df_train.isnull().any(axis=1)][null_columns])
df_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Moving on to test dataset.
df_test.isna().sum()
# Nothing to worry about. 
# Contains positive tweets 

# Here we have a glimpse of positive cases

df_pos = df_train[df_train['sentiment']=='positive']

df_pos['text'].head()
df_neg = df_train[df_train['sentiment']=='negative']

df_neg['text'].head()
df_neu = df_train[df_train['sentiment']=='neutral']

df_neu['text'].head()
# Now lets check out whether tha dataset is distributed equally or not
df_train['sentiment'].value_counts()

sns.barplot(df_train['sentiment'].value_counts().index,df_train['sentiment'].value_counts(),palette='rocket')
# Lets check out for tes dataset what is the proportions
df_test['sentiment'].value_counts()
sns.barplot(df_test['sentiment'].value_counts().index,df_test['sentiment'].value_counts(),palette='rocket')
import torch
from transformers import BertForQuestionAnswering



model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
def answer_question(question, answer_text):

    '''

    Takes a `question` string and an `answer_text` string (which contains the

    answer), and identifies the words within the `answer_text` that are the

    answer. Prints them out.

    '''

    # ======== Tokenize ========

    # Apply the tokenizer to the input text, treating them as a text-pair.

    input_ids = tokenizer.encode(question, answer_text)



    # Report how long the input sequence is.

    #print('Query has {:,} tokens.\n'.format(len(input_ids)))



    # ======== Set Segment IDs ========

    # Search the input_ids for the first instance of the `[SEP]` token.

    sep_index = input_ids.index(tokenizer.sep_token_id)



    # The number of segment A tokens includes the [SEP] token istelf.

    num_seg_a = sep_index + 1



    # The remainder are segment B.

    num_seg_b = len(input_ids) - num_seg_a



    # Construct the list of 0s and 1s.

    segment_ids = [0]*num_seg_a + [1]*num_seg_b



    # There should be a segment_id for every input token.

    assert len(segment_ids) == len(input_ids)



    # ======== Evaluate ========

    # Run our example question through the model.

    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.

                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text



    # ======== Reconstruct Answer ========

    # Find the tokens with the highest `start` and `end` scores.

    answer_start = torch.argmax(start_scores)

    answer_end = torch.argmax(end_scores)



    # Get the string versions of the input tokens.

    tokens = tokenizer.convert_ids_to_tokens(input_ids)



    # Start with the first token.

    answer = tokens[answer_start]



    # Select the remaining answer tokens and join them with whitespace.

    for i in range(answer_start + 1, answer_end + 1):

        

        # If it's a subword token, then recombine it with the previous token.

        if tokens[i][0:2] == '##':

            answer += tokens[i][2:]

        

        # Otherwise, add a space then the token.

        else:

            answer += ' ' + tokens[i]



    return answer
text ='Spent the entire morning in a meeting w/ a vendor, and my boss was not happy w/ them. Lots of fun.I had other plans for my morning'

question = 'What text is neutral?'
ans = answer_question(question, text)

print(ans)
# Initialise the text

df_train['Bert_answers'] = ''
df_train.head()
positive_question = 'What text is positive ?' # In case of positive sentiment

negative_question = 'What text is negative ?'  # In case of negative sentiment

neutral_question = 'What text is neutral?' # In case of neutral sentiment
df_test.shape[0]


df_test['selected_text'] = ''

i = 0

while(i!=df_test.shape[0]):

    if (df_test['sentiment'].iloc[i]== 'positive'):

        df_test['selected_text'].iloc[i] = answer_question(positive_question,df_test['text'][i])

    elif (df_test['sentiment'].iloc[i]== 'negative'):

        df_test['selected_text'].iloc[i] = answer_question(negative_question,df_test['text'][i])

    else :

        df_test['selected_text'].iloc[i] = answer_question(neutral_question,df_test['text'][i])

    print(df_test['selected_text'].iloc[i])

    print(i)

    i = i+1

    

     

df_test.head()
df_test = df_test.drop(['text','sentiment'],axis = 1)

df_test.head()
df_test.set_index('textID',inplace = True)

df_test.head()
df_test.to_csv('submission.csv')
# So we try to make efficent code
# def soc_iter(sentiment,text):

#     if (sentiment == 'positive'):

#         result = answer_question(positive_question,text)

#     elif (sentiment == 'negative'):

#         result = answer_question(negative_question,text)

#     else :

#         result = answer_question(neutral_question,text)

        

    
# %%timeit

# df_test['selected_text'] = ''

# draw_series = []

# for index, row in df_test.iterrows():

#     draw_series.append(soc_iter(row['sentiment'],row['text']))

#     print(index)

    

# df_test['selected_text'] = draw_series
# We can see this method even though 321 times faster but still is gonna take an hour to complete