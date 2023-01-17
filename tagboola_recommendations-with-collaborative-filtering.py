import pandas as pd

import numpy as np

from scipy.sparse import coo_matrix
# install implicit package

# !pip install implicit

import implicit
!export OPENBLAS_NUM_THREADS=1
FACTORS = 25



# load data

df_professionals = pd.read_csv(f'../input/professionals.csv')

df_questions = pd.read_csv(f'../input/questions.csv')

df_answers = pd.read_csv(f'../input/answers.csv')
# helpers for converting id's to indices and vice-versa

p_id_to_idx = { p_id: i for i, p_id in zip(df_professionals.index, df_professionals.professionals_id) }

p_idx_to_id = { i: p_id for p_id, i in p_id_to_idx.items() }

q_id_to_idx = { q_id: i for i, q_id in zip(df_questions.index, df_questions.questions_id) }

q_idx_to_id = { i: q_id for q_id, i in q_id_to_idx.items() }
# helper methods

def get_prof(prof_id):

    return df_professionals[df_professionals.professionals_id == prof_id][['professionals_industry', 'professionals_headline']] 



def get_q_and_a(prof_id):

    pd_q_and_a = pd.merge(left=df_questions, right=df_answers.rename(columns={'answers_question_id': 'questions_id'}), how='left')

    return pd_q_and_a[pd_q_and_a.answers_author_id == prof_id][['questions_id','questions_title', 'questions_body', 'answers_body']]



def recommend(prof_id, model, item_users):

    q_rec_idxs = [q_idx for q_idx, _ in model.recommend(p_id_to_idx[prof_id], item_users.tocsr().T)]

    return df_questions[df_questions.index.isin(q_rec_idxs)][['questions_id','questions_title', 'questions_body']]



def train(item_users, user_factors=None, item_factors=None, factors=FACTORS):

    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=0.01, dtype=np.float64, iterations=10)

    if user_factors is not None:

        model.user_factors = user_factors

    if item_factors is not None:

        model.item_factors = item_factors

    confidence = 20

    model.fit(confidence * item_users)

    return model



def explain(item_users,p_id, q_id, ):

    _, contributions, _ = model.explain(p_id_to_idx[p_id], item_users.tocsr().T, 

              q_id_to_idx[q_id])

    q_ids = [q_idx_to_id[q_id] for q_id, _ in contributions]

    return df_questions[df_questions.questions_id.isin(q_ids)]



def similar(model, q_id):

    q_idxs_and_scores = model.similar_items(q_id_to_idx[q_id])

    q_idxs = [q_idx for q_idx, _ in q_idxs_and_scores]

    return df_questions[df_questions.index.isin(q_idxs)]
# save index of of professional and question if an answer exists

p_and_q_idxs = set()

for p_id, q_id in zip(df_answers.answers_author_id, df_answers.answers_question_id):

    if p_id in p_id_to_idx and q_id in q_id_to_idx:

        p_and_q_idxs.add((p_id_to_idx[p_id], q_id_to_idx[q_id]))

        

        

p_idxs, q_idxs = [], []

for p_idx, q_idx in p_and_q_idxs:

    p_idxs.append(p_idx)

    q_idxs.append(q_idx)



#create sparse matrix

P = df_professionals.shape[0]

Q = df_questions.shape[0]

item_users = coo_matrix((np.ones(len(p_idxs)), (q_idxs, p_idxs)), shape=(Q, P))
# create model and fit using the matrix, and a confidence level

model = train(item_users)
sample_dev_prof_id = '1ec14aee9311480681dfa81b0f193de8'

get_prof(sample_dev_prof_id)
get_q_and_a(sample_dev_prof_id)
recommend(sample_dev_prof_id, model, item_users).values
sample_architect_prof_id = 'b45e7851aded479a92282ea3c66300ab'

get_prof(sample_architect_prof_id)
get_q_and_a(sample_architect_prof_id)
recommend(sample_architect_prof_id, model, item_users).values
# initialize the user and item factor vectors

user_factors = np.random.rand(df_professionals.shape[0], FACTORS).astype(np.float32) * 0.1

item_factors = np.random.rand(df_questions.shape[0], FACTORS).astype(np.float32) * 0.1



# modify the user factor vector for professionals in the architecture industry 

for i, industry in zip(df_professionals.index, df_professionals.professionals_industry.values):

    if industry == industry and 'architecture' in industry.lower():

        # make the value of the first index much larger than the others

        user_factors[i, 0] = 5.0



# modify the item factor vector for questions that contain the word architecture        

for i, title, body in zip(df_questions.index, df_questions.questions_title, df_questions.questions_body):

    title, body = title.lower(), body.lower()

    if 'architecture' in title or 'architecture' in body:

        # make the value of the first index much larger than the others

        item_factors[i, 0] = 5.0
model = train(item_users, user_factors=user_factors, item_factors=item_factors)
recommend(sample_architect_prof_id, model, item_users).values
sample_bio_prof_id = 'b2def296aff34e5da6b405fae8969e73'

get_prof(sample_bio_prof_id)
get_q_and_a(sample_bio_prof_id)
recommend(sample_bio_prof_id, model, item_users).values
# save index of of professional and question if an answer exists

p_and_q_idxs = set()

for p_id, q_id in zip(df_answers.answers_author_id, df_answers.answers_question_id):

    if p_id in p_id_to_idx and q_id in q_id_to_idx:

        p_and_q_idxs.add((p_id_to_idx[p_id], q_id_to_idx[q_id]))

        

        

p_idxs, q_idxs = [], []

for p_idx, q_idx in p_and_q_idxs:

    p_idxs.append(p_idx)

    q_idxs.append(q_idx)



# get professionals that have not answered a question

df_professionals_cold = df_professionals[~df_professionals.professionals_id.isin(np.unique(df_answers.answers_author_id.values))]

        

# get the indices of professionals in the biotechnology and biomedical engineering industry        

bio_p_idxs, bio_q_idxs = [], []

for i, industry in zip(df_professionals_cold.index, df_professionals_cold.professionals_industry.values):

    if industry == industry and industry.lower() in ['biotechnology', 'biomedical engineering']:

        bio_p_idxs.append(i)



# get the indices of questions with the terms biotechnology or biomedical engineering in them

for i, title, body in zip(df_questions.index, df_questions.questions_title, df_questions.questions_body):

    title, body = title.lower(), body.lower()

    if 'biomedical' in title or 'biomedical' in body or 'biotechnology' in title or 'biotechnology' in body:

        bio_q_idxs.append(i)



# save the pairs of indices to two arrays        

bp_idxs, bq_idxs = [], []        

for i in bio_q_idxs:

    for j in bio_p_idxs:

            bq_idxs.append(i)

            bp_idxs.append(j)

    



# #create sparse matrix

P = df_professionals.shape[0]

Q = df_questions.shape[0]



# assign difference values for questions that professionals answered 

# and questions/professionals that contain biotechnology/biomedical engineering

answer_values = np.full((len(p_idxs), 1), 1.0)

bio_values = np.full((len(bp_idxs), 1), 0.001)



values = np.concatenate((answer_values, bio_values))

updated_item_users = coo_matrix((values[:, 0], (q_idxs + bq_idxs, p_idxs + bp_idxs)), shape=(Q, P), dtype=np.float64)
model = train(updated_item_users)
recommend(sample_bio_prof_id, model, item_users).values