import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import seaborn as sns

import warnings

import copy

import nltk

import string

from nltk.corpus import stopwords

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import MultiLabelBinarizer

from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')

root_path = '../input/'

print('The csv files provided are:\n')

print(os.listdir(root_path))
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
df_tag_users_merged = pd.merge(df_tag_users, df_tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')

#To see the tags that are linked with every question

thresh = 0.0025

df_tag_questions_merged = pd.merge(df_tag_questions, df_tags, left_on='tag_questions_tag_id', right_on='tags_tag_id', how='inner')

user_tags = list((df_tag_users_merged['tags_tag_name'].value_counts()/df_tag_users_merged['tag_users_user_id'].nunique() 

                     > thresh).index[(df_tag_users_merged['tags_tag_name'].value_counts()/df_tag_users_merged['tag_users_user_id'].nunique() > thresh)])

question_tags = list((df_tag_questions_merged['tags_tag_name'].value_counts()/df_tag_questions_merged['tag_questions_question_id'].nunique() 

                     > thresh).index[(df_tag_questions_merged['tags_tag_name'].value_counts()/df_tag_questions_merged['tag_questions_question_id'].nunique() > thresh)])

relevant_tags = set(user_tags).union(set(question_tags))
print('The number of relevant tags:', len(relevant_tags))

print('Coverage of tagged questions:', df_tag_questions_merged[df_tag_questions_merged['tags_tag_name'].isin(

relevant_tags)]['tag_questions_question_id'].nunique()/df_tag_questions_merged['tag_questions_question_id'].nunique())

print('Coverage of tagged users:', df_tag_users_merged[df_tag_users_merged['tags_tag_name'].isin(

relevant_tags)]['tag_users_user_id'].nunique()/df_tag_users_merged['tag_users_user_id'].nunique())
def return_if_found(df, column, string):

    return(df[df[column].str.contains(string)][column].unique())

array_computer = return_if_found(df_tag_users_merged, 'tags_tag_name', 'computer')

computer_ed = np.array([nltk.edit_distance('computer',word) for word in array_computer])

df_computer_ed = pd.DataFrame({'string':array_computer, 'edit_distance':computer_ed})

df_computer_ed[df_computer_ed['edit_distance']<=2]
#Sorting tags based on follower count

dict_users_tags = df_tag_users_merged['tags_tag_name'].value_counts().to_dict()

list_users_tags = sorted(dict_users_tags, key=dict_users_tags.get)[::-1]

dict_rank_tags = {}

for rank, tags in enumerate(list_users_tags):

    dict_rank_tags[tags] = rank

def find_similar(relevant, space):

    dict_similar_tags = {}

    for tag_i in tqdm(relevant):

        dict_similar_tags[tag_i] = []

        for tag_j in space:

            if tag_i != tag_j:

                if (len(tag_i) >5 and nltk.edit_distance(tag_i, tag_j) <= 2): #Condition 1

                    dict_similar_tags[tag_i].append(tag_j)

                elif (len(tag_i) <=5 and (tag_j == '#'+tag_i or tag_j == tag_i+'s' or tag_j == tag_i+'-')): #Condition 2

                    dict_similar_tags[tag_i].append(tag_j)

    return(dict_similar_tags)

dict_similar_tags = find_similar(relevant_tags, list_users_tags)
print('Rank of tag internship:', dict_rank_tags['internship'])

print('Tags similar to internship:', dict_similar_tags['internship'])

print('Rank of tag internship:', dict_rank_tags['internships'])

print('Tags similar to internship:', dict_similar_tags['internships'])
#Deepcopy as the value is a mutable data structure

dict_similar_tags_copy = copy.deepcopy(dict_similar_tags)

for tag, similar in dict_similar_tags.items():

    for subtag in similar:

        if dict_rank_tags[subtag] < dict_rank_tags[tag]:

            dict_similar_tags_copy[tag].remove(subtag) 

        else:

            pass

for tag, similar in dict_similar_tags_copy.items():

    extensions = []

    for subtag in similar:

        try:

            extensions.extend(dict_similar_tags_copy[subtag])

        except:

            pass

    dict_similar_tags_copy[tag].extend(extensions)

    dict_similar_tags_copy[tag] = list(set(dict_similar_tags_copy[tag]))

#This is what we'll replace

similar_tags = list(set([item for sublist in list(dict_similar_tags_copy.values()) for item in sublist]))

list_keys = list(dict_similar_tags.keys())

for tag in list_keys:

    if tag in similar_tags:

        del dict_similar_tags_copy[tag]

#This is what we'll use to replace

main_tags = list(dict_similar_tags_copy.keys())
print('Tags similar to internships:', dict_similar_tags_copy['internships'])

print('Tags similar to internship:\n')

try:

    print(dict_similar_tags_copy['internship'])

#internship is not a key on account of being a low ranked tag

except KeyError as e:

    print('Key Error:', e)
print('The number of relevant tags:', len(main_tags))

print('The number of tags that the relevant tags cover:', len(similar_tags))

df_tag_users_merged = pd.merge(df_tag_users, df_tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')

df_tag_questions_merged = pd.merge(df_tag_questions, df_tags, left_on='tag_questions_tag_id', right_on='tags_tag_id', how='inner')

print('Coverage of tagged questions:', df_tag_questions_merged[df_tag_questions_merged['tags_tag_name'].isin(

    set(main_tags).union(set(similar_tags)))]['tag_questions_question_id'].nunique()/df_tag_questions_merged['tag_questions_question_id'].nunique())

print('Coverage of tagged users:', df_tag_users_merged[df_tag_users_merged['tags_tag_name'].isin(

    set(main_tags).union(set(similar_tags)))]['tag_users_user_id'].nunique()/df_tag_users_merged['tag_users_user_id'].nunique())
#To create a dictionary of tags that'll have the tag to be replaced as the key and the tag to replace it by as the value

dict_replace_tag = {}

for tag, similar in dict_similar_tags_copy.items():

    for subtag in similar:

        dict_replace_tag[subtag] = tag

for tag in main_tags:

    dict_replace_tag[tag] = tag

#Only looking at the questions that have atleast one tag out of the union of main tags and similar tags

df_all_tag_questions = df_tag_questions_merged[df_tag_questions_merged['tags_tag_name'].isin(

    set(main_tags).union(similar_tags))].groupby('tag_questions_question_id', as_index=False).agg({'tags_tag_name':list})

def replace(list_tags):

    replaced_list = []

    for tag in list_tags:

        try:

            replaced_list.append(dict_replace_tag[tag])

        except:

            pass

    return(list(set(replaced_list)))

df_all_tag_questions['replaced_tag_name'] = df_all_tag_questions['tags_tag_name'].apply(replace)
#Create MultiLabelBinarizer object

one_hot = MultiLabelBinarizer()

#One-hot encode data

design_matrix = one_hot.fit_transform(df_all_tag_questions['replaced_tag_name'])
#Building cosine similarity matrix 

cos_sim = 1-squareform(pdist(design_matrix, metric='cosine')) #pdist computes cosine distance, so we subtract that from 1 to compute similarity

del design_matrix #To free up the RAM
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
qid = 1

idx = np.argsort(cos_sim[qid,:])[-6:-1]

print('Question Title and Body:\n')

#Sample question

print(list(df_questions[df_questions['questions_id']==df_all_tag_questions['tag_questions_question_id'].iloc[qid]]['questions_title']))

print(list(df_questions[df_questions['questions_id']==df_all_tag_questions['tag_questions_question_id'].iloc[qid]]['questions_body']))
#Printing out the question body as it gives more insight into what the student actually wants to ask

print('Similar questions ranked by cosine similarity:\n')

for rank, index in enumerate(idx[::-1]):

    print(rank, '-', list(df_questions[df_questions['questions_id']==df_all_tag_questions.iloc[index]['tag_questions_question_id']]['questions_body']))
author_id = df_answers[df_answers['answers_question_id'].isin(df_all_tag_questions.iloc[idx[::-1]]['tag_questions_question_id'])]['answers_author_id']

active_author_id = author_id[author_id.isin(active_professionals)]

df_recommended_pros = df_qap_time_taken[df_qap_time_taken['professionals_id'].isin(active_author_id)].sort_values('less_than_2_days', ascending=False)

print('The recommended professionals ranked by the proportion of questions answered within 48 hours:', df_recommended_pros['professionals_id'].tolist())

print('The profile of the professionals:')

df_professionals[df_professionals['professionals_id'].isin(df_recommended_pros['professionals_id'])]
qid = 512

idx = np.argsort(cos_sim[qid,:])[-6:-1]

print('Question Title and Body:\n')

#Sample question. Printing out the question body as it gives more insight into what the student actually wants to ask

print(list(df_questions[df_questions['questions_id']==df_all_tag_questions['tag_questions_question_id'].iloc[qid]]['questions_title']))

print(list(df_questions[df_questions['questions_id']==df_all_tag_questions['tag_questions_question_id'].iloc[qid]]['questions_body']))
#Printing out the question body as it gives more insight into what the student actually wants to ask

print('Similar questions ranked by cosine similarity:\n')

for rank, index in enumerate(idx[::-1]):

    print(rank, '-', list(df_questions[df_questions['questions_id']==df_all_tag_questions.iloc[index]['tag_questions_question_id']]['questions_body']))
author_id = df_answers[df_answers['answers_question_id'].isin(df_all_tag_questions.iloc[idx[::-1]]['tag_questions_question_id'])]['answers_author_id']

active_author_id = author_id[author_id.isin(active_professionals)]

df_recommended_pros = df_qap_time_taken[df_qap_time_taken['professionals_id'].isin(active_author_id)].sort_values('less_than_2_days', ascending=False)

print('The recommended professionals ranked by the proportion of questions answered within 48 hours:', df_recommended_pros['professionals_id'].tolist())

print('The profile of the professionals:')

df_professionals[df_professionals['professionals_id'].isin(df_recommended_pros['professionals_id'])]