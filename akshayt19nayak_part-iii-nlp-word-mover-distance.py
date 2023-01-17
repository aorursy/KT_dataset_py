import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

from nltk import word_tokenize

from gensim.models import KeyedVectors

from nltk.corpus import stopwords

from gensim.models import Word2Vec

from gensim.similarities import WmdSimilarity

from time import time

from tqdm import tqdm

stop_words = stopwords.words('english')

warnings.filterwarnings('ignore')

root_path = '../input/data-science-for-good-careervillage/'

print('The csv files provided are:\n')

print(os.listdir(root_path))

model = KeyedVectors.load_word2vec_format('../input/word2vec-google/GoogleNews-vectors-negative300.bin', binary=True)
df_emails = pd.read_csv(root_path + 'emails.csv')

df_questions = pd.read_csv(root_path + 'questions.csv')

df_professionals = pd.read_csv(root_path + 'professionals.csv')

df_comments = pd.read_csv(root_path + 'comments.csv')

df_tag_users = pd.read_csv(root_path + 'tag_users.csv')

df_group_memberships = pd.read_csv(root_path + 'group_memberships.csv')

df_tags = pd.read_csv(root_path + 'tags.csv')

df_answer_scores = pd.read_csv(root_path + 'answer_scores.csv')

df_students = pd.read_csv(root_path + 'students.csv')

df_groups = pd.read_csv(root_path + 'groups.csv')

df_tag_questions = pd.read_csv(root_path + 'tag_questions.csv')

df_question_scores = pd.read_csv(root_path + 'question_scores.csv')

df_matches = pd.read_csv(root_path + 'matches.csv')

df_answers = pd.read_csv(root_path + 'answers.csv')

df_school_memberships = pd.read_csv(root_path + 'school_memberships.csv')
def preprocess(doc):

    doc = doc.lower()  # Lower the text.

    doc = word_tokenize(doc)  # Split into words.

    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.

    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.

    return(doc)
clean_corpus = [] 

documents = df_questions['questions_title'].tolist()  #wmd_corpus, with no pre-processing (so we can see the original documents).

for text in tqdm(df_questions['questions_title']):

    text = preprocess(text)   

    clean_corpus.append(text)        
start = time()

#We won't be training our own Word2Vec model and we'll instead use the pretrained vectors

num_best = 10

instance = WmdSimilarity(clean_corpus, model, num_best=num_best)  

print('Cell took %.2f seconds to run.' % (time() - start))
#To see the profile of the volunteers and the questions that they have answered

df_questions['questions_date_added'] = pd.to_datetime(df_questions['questions_date_added'])

df_answers['answers_date_added'] = pd.to_datetime(df_answers['answers_date_added'])

df_answers_professionals = pd.merge(df_answers, df_professionals, left_on='answers_author_id', right_on='professionals_id', how='outer')

df_questions_answers_professionals = pd.merge(df_questions, df_answers_professionals, left_on='questions_id', right_on='answers_question_id')

df_qap_time_taken = df_questions_answers_professionals.groupby(['professionals_id','questions_id']).agg({'questions_date_added':min, 'answers_date_added':min})

df_qap_time_taken['less_than_2_days'] = df_qap_time_taken['answers_date_added'] - df_qap_time_taken['questions_date_added'] < '2 days'

df_qap_time_taken = df_qap_time_taken.reset_index().groupby('professionals_id', as_index=False).agg({'less_than_2_days':np.mean})

last_date = df_questions['questions_date_added'].max() #date of the last question asked on the platform

df_ap_grouped = df_answers_professionals.groupby('professionals_id').agg({'answers_date_added':max}).apply(lambda x:

                                                                                          (last_date-x).dt.days)

df_ap_grouped.rename(columns={'answers_date_added':'days_since_answered'}, inplace=True)

active_professionals = df_ap_grouped[df_ap_grouped['days_since_answered']<365].index
sent = 'Should I declare a minor during undergrad if I want to be a lawyer?'

topk = 5

query = preprocess(sent)

sims = instance[query]  #A query is simply a "look-up" in the similarity class.

#Print the query and the retrieved documents, together with their similarities.

print('Question:')

print(sent)

#We won't consider the first index since that is the question itself

for i in range(1,topk+1): 

    print('\nsim = %.4f' % sims[i][1])

    print(documents[sims[i][0]])
idx = [tup[0] for tup in sims][:5]

author_id = df_answers[df_answers['answers_question_id'].isin(df_questions.iloc[idx]['questions_id'])]['answers_author_id']

active_author_id = author_id[author_id.isin(active_professionals)]

df_recommended_pros = df_qap_time_taken[df_qap_time_taken['professionals_id'].isin(active_author_id)].sort_values('less_than_2_days', ascending=False)

print('The recommended professionals ranked by the proportion of questions answered within 48 hours:', df_recommended_pros['professionals_id'].tolist())

print('The profile of the professionals:')

df_professionals[df_professionals['professionals_id'].isin(df_recommended_pros['professionals_id'])]
sent = 'My current plan is to go to a one year film college to get a certificate in screenwriting. Many people have mentioned that you really don\'t need a film degree to get into film, so a certificate is fine. Is this true?'

topk = 5

query = preprocess(sent)

sims = instance[query]  #A query is simply a "look-up" in the similarity class.

#Print the query and the retrieved documents, together with their similarities.

print('Question:')

print(sent)

for i in range(1,topk+1):

    print('\nsim = %.4f' % sims[i][1])

    print(documents[sims[i][0]])
idx = [tup[0] for tup in sims][:5]

author_id = df_answers[df_answers['answers_question_id'].isin(df_questions.iloc[idx]['questions_id'])]['answers_author_id']

active_author_id = author_id[author_id.isin(active_professionals)]

df_recommended_pros = df_qap_time_taken[df_qap_time_taken['professionals_id'].isin(active_author_id)].sort_values('less_than_2_days', ascending=False)

print('The recommended professionals ranked by the proportion of questions answered within 48 hours:', df_recommended_pros['professionals_id'].tolist())

print('The profile of the professionals:')

df_professionals[df_professionals['professionals_id'].isin(df_recommended_pros['professionals_id'])]
sent = 'If I want to be a lawyer, should I declare a minor during undergrad?' 

query = preprocess(sent)

sims = instance[query]  #A query is simply a "look-up" in the similarity class.

#Print the query and the retrieved documents, together with their similarities.

print('Question:')

print(sent)

for i in range(0,topk+1): 

    print('\nsim = %.4f' % sims[i][1])

    print(documents[sims[i][0]])
"""

Let's concatenate the professionals' industry (which is finite and exhaustive) and the professionals' headline. Let's call it industry_headline

active_pros = pros who have answered questions in the last 1 year

new_pros = pros who have registered in the last 1 year



For each question:

1. Compute similarity with a question asked in the past - on the basis on tags using Tag RecSys (if tags are available) and on the basis of questions title using WMD Similarity



2. Go through the pipeline for making recommendations (https://www.kaggle.com/akshayt19nayak/part-ii-tag-recsys-cosine-levenshtein-dist#Recommender-System) and make a list of active_pros that have answered questions having a similarity score above a certain threshold with the question under consideration. Let's call it reco_active_pros. 

(As far as the similarity score is concerned we can make a weighted score of (0.5 * tag similarity + 0.5 * wmd similarity) and then decide on a threshold)



3. count_active = 0, count_new = 0 

For each pro in reco_active_pros:

    - if count_active <= k:

        - if (pro has already been recommended 2 times in the last 3 days):

            - continue (i.e move onto the next pro)

        - else:

            - recommend 

            count_active = count_active + 1

    Let's call this final list as final_reco_active_pros. The number k is user defined i.e how many active pros should we actually recommend?

    

    - if count_new <= k:

        calculate WMD Similarity of a pro with each new_pros industry_headline or/and tags by using Tag RecSys. Select the ones above a certain similarity threshold. Let's call them sim_new_pros

        - For each new_pro in sim_new_pros:

            - if count_new <= k

                - if (new_pro has already been recommended 2 times in the last 3 days):

                    - continue (i.e move onto the next new_pro)

                - else:

                    - recommend 

                    count_new = count_new + 1

    Let's call this final list as final_reco_new_pros

    

Thus for every question we will make a recommendation of final_reco_active_pros + final_reco_new_pros

"""