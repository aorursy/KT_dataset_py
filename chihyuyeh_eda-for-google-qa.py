# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import json



from IPython.display import display



#local script

from tfutils_py import get_answer, read_sample



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
df = read_sample(n=10)

print(df['question_text'][1])

print(df['annotations'][1])
df = read_sample(n=5000)



def get_html_tag_from_long_answer_candidate(document, candidate):

    start_token = candidate['start_token']

    end_token = candidate['end_token']

    tokenized_document = document.split(' ')

    

    html_tag_head = tokenized_document[start_token].lower()

    html_tag_tail = tokenized_document[end_token-1].lower()

    

    if html_tag_head == '<h1>' and html_tag_tail == '</h1>':

        return 'HEADLINE'

    elif html_tag_head == '<p>' and html_tag_tail == '</p>':

        return 'PARAGRAPH'

    elif html_tag_head == '<h1>' and html_tag_tail == '</h1>':

        return 'HEADLINE'

    elif html_tag_head == '<table>' and html_tag_tail == '</table>':

        return 'TABLE'

    elif html_tag_head == '<tr>' and html_tag_tail == '</tr>':

        return 'TABLE_ROW'

    elif html_tag_head == '<ul>' and html_tag_tail == '</ul>':

        return 'UNORDERED_LIST'

    elif html_tag_head == '<ol>' and html_tag_tail == '</ol>':

        return 'ORDERED_LIST'

    elif html_tag_head == '<dl>' and html_tag_tail == '</dl>':

        return 'DEFINITION_LIST'

    elif html_tag_head == '<dd>' and html_tag_tail == '</dd>':

        return 'DEFINITION_DESCRIPTION'

    elif html_tag_head == '<dt>' and html_tag_tail == '</dt>':

        return 'DEFINITION_TERM'

    elif html_tag_head == '<li>' and html_tag_tail == '</li>':

        return 'LIST_ITEM'

    else:

        print(html_tag_head)

        print(html_tag_tail)

        return 'UNKNOWN'



def test_get_html_tag_from_long_answer_candidate(df):

    SUCCESS_FLAG = True

    for (i, document) in enumerate(df['document_text']):

        for long_answer_candidate in df['long_answer_candidates'][i]:

            if get_html_tag_from_long_answer_candidate(document, long_answer_candidate) == 'UNKNOWN':

                SUCCESS_FLAG = False

                break

        if not SUCCESS_FLAG:

            break



    if SUCCESS_FLAG:

        print("Test is successful")



test_get_html_tag_from_long_answer_candidate(df)
df = read_sample(n=10)

df.loc[0,'document_text']
df = read_sample(n=1000)

doc_text_words = df['document_text'].apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(12,6))

sns.distplot(doc_text_words.values,kde=True,hist=False).set_title('Distribution of text word count of 1000 docs')
df = read_sample(n=3)

df.long_answer_candidates[0][:5]
# sample answer

sample = df.iloc[0]

get_answer(sample.document_text, sample.long_answer_candidates[0])
def preprocess(n=10):

    df = read_sample(n=n,ignore_doc_text=True, offset=6000)

    df['yes_no'] = df.annotations.apply(lambda x: x[0]['yes_no_answer'])

    df['long'] = df.annotations.apply(lambda x: [x[0]['long_answer']['start_token'], x[0]['long_answer']['end_token']])

    df['short'] = df.annotations.apply(lambda x: x[0]['short_answers'])

    return df

df = preprocess(5000)
display(df.long.apply(lambda x: "Answer Doesn't exist" if x[0] == -1 else "Answer Exists").value_counts(normalize=True))
# let's keep a mask of the answers that exist

mask_answer_exists = df.long.apply(lambda x: "Answer Doesn't exist" if x[0] == -1 else "Answer Exists") == "Answer Exists"
yes_no_dist = df.yes_no.value_counts(normalize=True)

yes_no_dist
short_dist = df[mask_answer_exists].short.apply(lambda x: "Short answer exists" if len(x) > 0 else "Short answer doesn't exist").value_counts(normalize=True)

plt.figure(figsize=(8,6))

sns.barplot(x=short_dist.index,y=short_dist.values).set_title("Distribution of short answers in answerable questions")

print(short_dist.values)
short_size_dist = df[mask_answer_exists].short.apply(len).value_counts(normalize=True)

short_size_dist_pretty = pd.concat([short_size_dist.loc[[0,1,],], pd.Series(short_size_dist.loc[2:].sum(),index=['>=2'])])

short_size_dist_pretty = short_size_dist_pretty.rename(index={0: 'No Short answer',1:"1 Short answer",">=2":"More than 1 short answers"})

plt.figure(figsize=(12,6))

sns.barplot(x=short_size_dist_pretty.index,y=short_size_dist_pretty.values).set_title("Distribution of Number of Short Answers in answerable questions")

print(short_size_dist_pretty.values)