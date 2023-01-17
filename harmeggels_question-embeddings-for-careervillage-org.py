import pandas as pd

import numpy as np

from bs4 import BeautifulSoup



from IPython.display import HTML



import matplotlib.pyplot as plt



from nltk.corpus import stopwords

import gensim



from gensim.utils import simple_preprocess

from gensim.models import FastText



from sklearn.neighbors import NearestNeighbors

from sklearn.feature_extraction.text import TfidfVectorizer



from functools import partial

import random



from ipywidgets import interact



pd.set_option('display.max_colwidth', -1)
!ls '../input'
def get_text(text):

    try:

        soup = BeautifulSoup(text, 'lxml')

        return soup.get_text()

    except Exception as e:

        print(text)

        raise e
questions = pd.read_csv('../input/questions.csv', parse_dates=['questions_date_added'])

questions['questions_body'] = questions['questions_body'].apply(get_text)



display(questions.head(5))
stopword = stopwords.words('english')



questions['text'] = questions['questions_title'] + questions['questions_body']

questions['text_list'] = questions['text'].apply(simple_preprocess)

questions['text'] = questions['text_list'].apply(lambda x: ' '.join(x))



display(questions.head(3))
emb_size = 100

model_question = FastText(questions['text_list'], size=emb_size, window = 6, sg=1, workers=4)

model_question.train(questions['text_list'], total_examples=len(questions.index), epochs=50)
vect_question = TfidfVectorizer(min_df=model_question.vocabulary.min_count)

tfidf_question = vect_question.fit_transform(questions['text'])
def get_sentence_embedding(m, tfidf, vectorizer, emb_size=100):

    wordvecs = np.zeros((emb_size, tfidf.shape[-1]))

    for i, name in enumerate(vectorizer.get_feature_names()):

        wordvecs[:, i] = m.wv[name]



    emb = tfidf @ wordvecs.T

    emb = emb / (tfidf.sum(axis=1) + 1e-10)

    

    return emb
sen_emb = get_sentence_embedding(model_question, tfidf_question, vect_question, 100)
@interact

def get_similar_question(x=200):

    nn = NearestNeighbors(n_neighbors=6, metric='cosine')

    nn.fit(sen_emb)

    dist, idxs = nn.kneighbors(sen_emb[x])



    sim_questions = questions.loc[idxs[0], ['questions_id', 'questions_author_id', 'questions_title', 'questions_body']]

    sim_questions['Score'] = dist[0]



    #display(HTML('Similar questions (actual question on top):'))

    #display(sim_questions)

    return sim_questions
profs = pd.read_csv('../input/professionals.csv')

tag_users = pd.read_csv('../input/tag_users.csv')

tags = pd.read_csv('../input/tags.csv')

tags['tags_tag_name'] = tags['tags_tag_name'].fillna(' ').apply(get_text)



tag_users = tag_users.merge(tags, left_on='tag_users_tag_id', right_on='tags_tag_id')

tag_users['tags_tag_name'] += ' '

tag_users = tag_users.groupby('tag_users_user_id')['tags_tag_name'].sum().to_frame()



group_memberships = pd.read_csv('../input/group_memberships.csv')

groups = pd.read_csv('../input/groups.csv')



group_memberships = group_memberships.merge(groups, left_on='group_memberships_group_id', right_on='groups_id')

group_memberships['groups_group_type'] += ' '

group_memberships = group_memberships.groupby('group_memberships_user_id')['groups_group_type'].sum().to_frame()



profs = profs.merge(tag_users, left_on='professionals_id', right_on='tag_users_user_id')

profs = profs.merge(group_memberships, left_on='professionals_id', right_on='group_memberships_user_id')

profs['info'] = profs['tags_tag_name'] + ' ' + profs['groups_group_type']

profs = profs.drop(['professionals_location', 'professionals_date_joined'], axis=1)



profs['info_list'] = profs['info'].apply(simple_preprocess)

profs['info'] = profs['info_list'].apply(lambda x: ' '.join(x))



display(profs.sample(5))
#model_profs = FastText(profs['info_list'], size=emb_size, window=8, sg=1, workers=4)

#model_profs.train(profs['info_list'], total_examples=len(profs.index), epochs=50)



#vect_profs = TfidfVectorizer(min_df=model_profs.vocabulary.min_count)

tfidf_profs = vect_question.transform(profs['info'])
prof_emb = get_sentence_embedding(model_question, tfidf_profs, vect_question, 100)
@interact

def get_closest_professional(x=200):

    nn = NearestNeighbors(n_neighbors=6, metric='cosine')

    nn.fit(prof_emb)

    dist, idxs = nn.kneighbors(sen_emb[x])



    question = questions.loc[x, ['questions_id', 'questions_author_id', 'questions_title', 'questions_body']].to_frame()

    display(question)

    

    closest_profs = profs.loc[idxs[0], ['professionals_id', 'professionals_industry', 'professionals_headline', 'info']]

    closest_profs['Score'] = dist[0]



    return closest_profs
answers = pd.read_csv('../input/answers.csv', parse_dates=['answers_date_added'])

answer_score = pd.read_csv('../input/answer_scores.csv')



answers = answers.dropna(subset=['answers_body'])

answers['answers_body'] = answers['answers_body'].apply(get_text)



answers = answers.merge(answer_score, left_on='answers_id', right_on='id')

answers = answers.loc[answers['score'] > 0]

answers = answers.merge(profs.reset_index(), left_on='answers_author_id', right_on='professionals_id')

answers = answers.merge(questions.reset_index(), left_on='answers_question_id', right_on='questions_id')

answers = answers.loc[:, ['answers_id', 'professionals_id', 'score', 'info', 'index_x', 'index_y', 'questions_body', 'text', 'text_list']]



display(answers.head(5))
sen_emb[519].shape


    