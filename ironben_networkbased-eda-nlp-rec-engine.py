import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)

from wordcloud import WordCloud,STOPWORDS

import warnings

warnings.filterwarnings('ignore')

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from IPython.display import Image

answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')

comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')

emails = pd.read_csv("../input/data-science-for-good-careervillage/emails.csv")

group_memberships = pd.read_csv('../input/data-science-for-good-careervillage/group_memberships.csv')

groups = pd.read_csv('../input/data-science-for-good-careervillage/groups.csv')

matches = pd.read_csv('../input/data-science-for-good-careervillage/matches.csv')

professionals = pd.read_csv("../input/data-science-for-good-careervillage/professionals.csv")

questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')

school_memberships = pd.read_csv('../input/data-science-for-good-careervillage/school_memberships.csv')

students = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')

tag_questions = pd.read_csv("../input/data-science-for-good-careervillage/tag_questions.csv")

tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')

tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')
df_list = [answers,comments,emails,group_memberships,groups,matches,professionals,questions,school_memberships,students,tag_questions,tag_users,tags]
Image(filename="../input/cv-graph-schema2/graph_schema_kaggle.png")
Image(filename="../input/temp-pics/new_schema.png")
Image(filename="../input/temp-pics/question_tags.png")
print(os.listdir("../input/temp-pics"))
Image(filename="../input/temp-pics/dab_q_tag.png")
Image(filename="../input/temp-pics/q_519_tag.png")
# Merge Question Title and Body

questions['questions_full_text'] = questions['questions_title'] +'\r\n\r\n'+ questions['questions_body']
import spacy

nlp = spacy.load('en')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')



# Spacy Tokenfilter for part-of-speech tagging

token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']



def nlp_preprocessing(data):

    """ Use NLP to transform the text corpus to cleaned sentences and word tokens



    """    

    def token_filter(token):

        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list



        """    

        return not token.is_stop and token.is_alpha and token.pos_ in token_pos

    

    processed_tokens = []

    data_pipe = nlp.pipe(data)

    for doc in data_pipe:

        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]

        processed_tokens.append(set(filtered_tokens))

    return processed_tokens



# Get NLP Tokens

questions['nlp_tokens'] = nlp_preprocessing(questions['questions_full_text'])
questions['nlp_tokens'].head()
all_q_tokens = list(questions['nlp_tokens'])
type(all_q_tokens)
all_unique_q_tokens = set([token for token_list in all_q_tokens for token in token_list])

flat_q_tokens = [token for token_list in all_q_tokens for token in token_list]
len(all_q_tokens)
all_q_tokens[:5]
len(all_unique_q_tokens)
list(all_unique_q_tokens)[:5]
len(flat_q_tokens)
flat_q_tokens[:5]
from collections import Counter,OrderedDict
tokens_sum = Counter(flat_q_tokens)

sorted_ord_tokens = sorted(tokens_sum.items(), key=lambda kv: kv[1],reverse=True)
sorted_ord_tokens
import spacy

nlp = spacy.load('en')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')



# Spacy Tokenfilter for part-of-speech tagging

token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']



def nlp_preprocessing(data):

    """ Use NLP to transform the text corpus to cleaned sentences and word tokens



    """    

    def token_filter(token):

        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list



        """    

        return not token.is_stop and token.is_alpha and token.pos_ in token_pos

    

    processed_tokens = []

    data_pipe = nlp.pipe(data)

    for doc in data_pipe:

        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]

        temp_string = ','

        processed_tokens.append(temp_string.join(list(set(filtered_tokens))))

    return processed_tokens



# Get NLP Tokens

questions['nlp_tokens'] = nlp_preprocessing(questions['questions_full_text'])
questions['nlp_tokens'].head()
q_exp = questions[['questions_id','nlp_tokens']]
q_exp.to_csv("q_exp_id_and_nlp_tokens.csv", index=False)
q_exp.head().to_csv("q_exp_test_5.csv", index=False)