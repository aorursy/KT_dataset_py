import os

import re

import math

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.dates as mdates

import networkx as nx



from wordcloud import WordCloud

from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



import spacy

nlp = spacy.load('en')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')

#nlp.remove_pipe('tagger')



import gensim

import pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
input_dir = '../input/data-science-for-good-careervillage'

print(os.listdir(input_dir))



professionals = pd.read_csv(os.path.join(input_dir, 'professionals.csv'))

groups = pd.read_csv(os.path.join(input_dir, 'groups.csv'))

comments = pd.read_csv(os.path.join(input_dir, 'comments.csv'))

school_memberships = pd.read_csv(os.path.join(input_dir, 'school_memberships.csv'))

tags = pd.read_csv(os.path.join(input_dir, 'tags.csv'))

emails = pd.read_csv(os.path.join(input_dir, 'emails.csv'))

group_memberships = pd.read_csv(os.path.join(input_dir, 'group_memberships.csv'))

answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))

students = pd.read_csv(os.path.join(input_dir, 'students.csv'))

matches = pd.read_csv(os.path.join(input_dir, 'matches.csv'))

questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))

tag_users = pd.read_csv(os.path.join(input_dir, 'tag_users.csv'))

tag_questions = pd.read_csv(os.path.join(input_dir, 'tag_questions.csv'))

answer_scores = pd.read_csv(os.path.join(input_dir, 'answer_scores.csv'))

question_scores = pd.read_csv(os.path.join(input_dir, 'question_scores.csv'))
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', -1)



seed = 13

random.seed(seed)

np.random.seed(seed)
token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']

actual_date = datetime(2019, 2 ,1)
def nlp_preprocessing(data):    

    def token_filter(token):   

        return not token.is_stop and token.is_alpha and token.pos_ in token_pos

    data = [re.compile(r'<[^>]+>').sub('', x) for x in data] 

    processed_tokens = []

    data_pipe = nlp.pipe(data)

    for doc in data_pipe:

        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]

        processed_tokens.append(filtered_tokens)

    return processed_tokens
# Transform datetime datatypes

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'], infer_datetime_format=True)

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'], infer_datetime_format=True)

professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'], infer_datetime_format=True)

students['students_date_joined'] = pd.to_datetime(students['students_date_joined'], infer_datetime_format=True)

emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'], infer_datetime_format=True)

comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'], infer_datetime_format=True)



# Merge Question Title and Body

questions['questions_full_text'] = questions['questions_title'] +'\r\n\r\n'+ questions['questions_body']

# Count of answers

temp = answers.groupby('answers_question_id').size()

questions['questions_answers_count'] = pd.merge(questions, pd.DataFrame(temp.rename('count')), left_on='questions_id', right_index=True, how='left')['count'].fillna(0).astype(int)

# First answer for questions

temp = answers[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').min()

questions['questions_first_answers'] = pd.merge(questions, pd.DataFrame(temp), left_on='questions_id', right_index=True, how='left')['answers_date_added']

# Last answer for questions

temp = answers[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').max()

questions['questions_last_answers'] = pd.merge(questions, pd.DataFrame(temp), left_on='questions_id', right_index=True, how='left')['answers_date_added']

# Hearts Score

temp = pd.merge(questions, question_scores, left_on='questions_id', right_on='id', how='left')

questions['questions_hearts'] = temp['score'].fillna(0).astype(int)

# Questions Tags list

temp = pd.merge(questions, tag_questions, left_on='questions_id', right_on='tag_questions_question_id', how='inner')

temp = pd.merge(temp, tags, left_on='tag_questions_tag_id', right_on='tags_tag_id', how='inner')

temp = temp.groupby('questions_id')['tags_tag_name'].apply(list).rename('questions_tags')

questions['questions_tags'] = pd.merge(questions, temp.to_frame(), left_on='questions_id', right_index=True, how='left')['questions_tags']

# Get NLP Tokens

questions['nlp_tokens'] = nlp_preprocessing(questions['questions_full_text'])



# Days required to answer the question

temp = pd.merge(questions, answers, left_on='questions_id', right_on='answers_question_id')

answers['time_delta_answer'] = (temp['answers_date_added'] - temp['questions_date_added'])

# Ranking for answers time

answers['answers_time_rank'] = answers.groupby('answers_question_id')['time_delta_answer'].rank(method='min').astype(int)

# Hearts Score

temp = pd.merge(answers, answer_scores, left_on='answers_id', right_on='id', how='left')

answers['answers_hearts'] = temp['score'].fillna(0).astype(int)



# Time since joining

professionals['professionals_time_delta_joined'] = actual_date - professionals['professionals_date_joined']

# Number of answers

temp = answers.groupby('answers_author_id').size()

professionals['professionals_answers_count'] = pd.merge(professionals, pd.DataFrame(temp.rename('count')), left_on='professionals_id', right_index=True, how='left')['count'].fillna(0).astype(int)

# Number of comments

temp = comments.groupby('comments_author_id').size()

professionals['professionals_comments_count'] = pd.merge(professionals, pd.DataFrame(temp.rename('count')), left_on='professionals_id', right_index=True, how='left')['count'].fillna(0).astype(int)

# Last activity (Answer)

temp = answers.groupby('answers_author_id')['answers_date_added'].max()

professionals['date_last_answer'] = pd.merge(professionals, pd.DataFrame(temp.rename('last_answer')), left_on='professionals_id', right_index=True, how='left')['last_answer']

# First activity (Answer)

temp = answers.groupby('answers_author_id')['answers_date_added'].min()

professionals['date_first_answer'] = pd.merge(professionals, pd.DataFrame(temp.rename('first_answer')), left_on='professionals_id', right_index=True, how='left')['first_answer']

# Last activity (Comment)

temp = comments.groupby('comments_author_id')['comments_date_added'].max()

professionals['date_last_comment'] = pd.merge(professionals, pd.DataFrame(temp.rename('last_comment')), left_on='professionals_id', right_index=True, how='left')['last_comment']

# First activity (Comment)

temp = comments.groupby('comments_author_id')['comments_date_added'].min()

professionals['date_first_comment'] = pd.merge(professionals, pd.DataFrame(temp.rename('first_comment')), left_on='professionals_id', right_index=True, how='left')['first_comment']

# Last activity (Total)

professionals['date_last_activity'] = professionals[['date_last_answer', 'date_last_comment']].max(axis=1)

# First activity (Total)

professionals['date_first_activity'] = professionals[['date_first_answer', 'date_first_comment']].min(axis=1)

# Total Hearts score

temp = answers.groupby('answers_author_id')['answers_hearts'].sum()

professionals['professional_answers_hearts'] = pd.merge(professionals, pd.DataFrame(temp.rename('answers_hearts')), left_on='professionals_id', right_index=True, how='left')['answers_hearts'].fillna(0).astype(int)

# Professionals Tags to List

temp = pd.merge(professionals, tag_users, left_on='professionals_id', right_on='tag_users_user_id', how='inner')

temp = pd.merge(temp, tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')

temp = temp.groupby('professionals_id')['tags_tag_name'].apply(list).rename('professionals_tags')

professionals['professionals_tags'] = pd.merge(professionals, temp.to_frame(), left_on='professionals_id', right_index=True, how='left')['professionals_tags']



# Time since joining

students['students_time_delta_joined'] = actual_date - students['students_date_joined']

# Number of answers

temp = questions.groupby('questions_author_id').size()

students['students_questions_count'] = pd.merge(students, pd.DataFrame(temp.rename('count')), left_on='students_id', right_index=True, how='left')['count'].fillna(0).astype(int)

# Number of comments

temp = comments.groupby('comments_author_id').size()

students['students_comments_count'] = pd.merge(students, pd.DataFrame(temp.rename('count')), left_on='students_id', right_index=True, how='left')['count'].fillna(0).astype(int)

# Last activity (Question)

temp = questions.groupby('questions_author_id')['questions_date_added'].max()

students['date_last_question'] = pd.merge(students, pd.DataFrame(temp.rename('last_question')), left_on='students_id', right_index=True, how='left')['last_question']

# First activity (Question)

temp = questions.groupby('questions_author_id')['questions_date_added'].min()

students['date_first_question'] = pd.merge(students, pd.DataFrame(temp.rename('first_question')), left_on='students_id', right_index=True, how='left')['first_question']

# Last activity (Comment)

temp = comments.groupby('comments_author_id')['comments_date_added'].max()

students['date_last_comment'] = pd.merge(students, pd.DataFrame(temp.rename('last_comment')), left_on='students_id', right_index=True, how='left')['last_comment']

# First activity (Comment)

temp = comments.groupby('comments_author_id')['comments_date_added'].min()

students['date_first_comment'] = pd.merge(students, pd.DataFrame(temp.rename('first_comment')), left_on='students_id', right_index=True, how='left')['first_comment']

# Last activity (Total)

students['date_last_activity'] = students[['date_last_question', 'date_last_comment']].max(axis=1)

# First activity (Total)

students['date_first_activity'] = students[['date_first_question', 'date_first_comment']].min(axis=1)

# Total Hearts score

temp = questions.groupby('questions_author_id')['questions_hearts'].sum()

students['students_questions_hearts'] = pd.merge(students, pd.DataFrame(temp.rename('questions_hearts')), left_on='students_id', right_index=True, how='left')['questions_hearts'].fillna(0).astype(int)

# Students Tags to List

temp = pd.merge(students, tag_users, left_on='students_id', right_on='tag_users_user_id', how='inner')

temp = pd.merge(temp, tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')

temp = temp.groupby('students_id')['tags_tag_name'].apply(list).rename('students_tags')

students['students_tags'] = pd.merge(students, temp.to_frame(), left_on='students_id', right_index=True, how='left')['students_tags']



emails_response = pd.merge(emails, matches, left_on='emails_id', right_on='matches_email_id', how='inner')

emails_response = pd.merge(emails_response, questions, left_on='matches_question_id', right_on='questions_id', how='inner')

emails_response = pd.merge(emails_response, answers, left_on=['emails_recipient_id', 'matches_question_id'], right_on=['answers_author_id', 'answers_question_id'], how='left')

emails_response = emails_response.drop(['matches_email_id', 'matches_question_id', 'answers_id', 'answers_author_id', 'answers_body', 'answers_question_id'], axis=1)

emails_response = emails_response.drop(['questions_author_id', 'questions_title', 'questions_body', 'questions_full_text'], axis=1)

emails_response['time_delta_email_answer'] = (emails_response['answers_date_added'] - emails_response['emails_date_sent'])

emails_response['time_delta_question_email'] = (emails_response['emails_date_sent'] - emails_response['questions_date_added'])
# Gensim Dictionary Filter

extremes_no_below = 20

extremes_no_above = 0.6

extremes_keep_n = 8000



# LDA

num_topics = 21

passes = 15

chunksize = 1000

alpha = 1/50
def get_model_results(ldamodel, corpus, dictionary): 

    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)

    transformed = ldamodel.get_document_topics(corpus)

    df = pd.DataFrame.from_records([{v:k for v, k in row} for row in transformed])

    return vis, df
def get_model_wordcloud(ldamodel):

    plot_cols = 3

    plot_rows = math.ceil(num_topics / 3)

    axisNum = 0

    plt.figure(figsize=(5*plot_cols, 3*plot_rows))

    for topicID in range(ldamodel.state.get_lambda().shape[0]):

                #gather most relevant terms for the given topic

        topics_terms = ldamodel.state.get_lambda()

        tmpDict = {}

        for i in range(1, len(topics_terms[0])):

            tmpDict[ldamodel.id2word[i]]=topics_terms[topicID,i]

        wordcloud = WordCloud( margin=0,max_words=20 ).generate_from_frequencies(tmpDict)

        axisNum += 1

        ax = plt.subplot(plot_rows, plot_cols, axisNum)



        plt.imshow(wordcloud, interpolation='bilinear')

        title = topicID

        plt.title(title)

        plt.axis("off")

        plt.margins(x=0, y=0)

    plt.show()
def topic_query(data, query):

    result = data

    result['sort'] = 0

    for topic in query:

        result = result[result[topic] >= query[topic]]

        result['sort'] += result[topic]

    result = result.sort_values(['sort'], ascending=False)

    result = result.drop('sort', axis=1)

    result = result.head(5)

    return result
def get_text_topics(text, top=20):    

    def token_topic(token):

        return topic_words.get(token, -1)    

    colors = ['\033[46m', '\033[45m', '\033[44m', '\033[43m', '\033[42m', '\033[41m', '\033[47m']    

    nlp_tokens = nlp_preprocessing([text])

    bow_text = [lda_dic.doc2bow(doc) for doc in nlp_tokens]

    bow_text = lda_tfidf[bow_text]

    topic_text = lda_model.get_document_topics(bow_text)

    topic_text = pd.DataFrame.from_records([{v:k for v, k in row} for row in topic_text])    

    print('Question:')

    topic_words = []

    topic_labeled = 0

    for topic in topic_text.columns.values:

        topic_terms = lda_model.get_topic_terms(topic, top)

        topic_words = topic_words+[[topic_labeled, lda_dic[pair[0]], pair[1]] for pair in topic_terms]

        topic_labeled += 1

    topic_words = pd.DataFrame(topic_words, columns=['topic', 'word', 'value']).pivot(index='word', columns='topic', values='value').idxmax(axis=1)

    nlp_doc = nlp(text)

    text_highlight = ''.join([x.string if token_topic(x.lemma_.lower()) <0  else colors[token_topic(x.lemma_.lower()) % len(colors)] + x.string + '\033[0m' for x in nlp_doc])

    print(text_highlight) 

    

    print('\nTopics:')

    topic_labeled = 0

    for topic in topic_text:

        print(colors[topic_labeled % len(colors)]+'Topic '+str(topic)+':', '{0:.2%}'.format(topic_text[topic].values[0])+'\033[0m')

        topic_labeled += 1

    plt_data = topic_text

    plt_data.columns = ['Topic '+str(c) for c in plt_data.columns]

    plt_data['Others'] = 1-plt_data.sum(axis=1)

    plt_data = plt_data.T

    plt_data.plot(kind='pie', y=0, autopct='%.2f')

    plt.xlabel('')

    plt.ylabel('')

    plt.title('Topics Probabilities')

    plt.show()
lda_tokens = questions['nlp_tokens']

lda_dic = gensim.corpora.Dictionary(lda_tokens)

lda_dic.filter_extremes(no_below=extremes_no_below, no_above=extremes_no_above, keep_n=extremes_keep_n)

lda_corpus = [lda_dic.doc2bow(doc) for doc in lda_tokens]



lda_tfidf = gensim.models.TfidfModel(lda_corpus)

lda_corpus = lda_tfidf[lda_corpus]

lda_model = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=num_topics, 

                                            id2word = lda_dic, passes=passes,

                                            chunksize=chunksize,update_every=0,

                                            alpha=alpha, random_state=seed)

lda_vis, lda_result = get_model_results(lda_model, lda_corpus, lda_dic)

lda_questions = questions[['questions_id', 'questions_title', 'questions_body']]

lda_questions = pd.concat([lda_questions, lda_result.add_prefix('Topic_')], axis=1)
get_model_wordcloud(lda_model)
lda_questions.head(5).dropna(axis=1, how='all').T

query = {'Topic_3':0.4, 'Topic_18':0.4}

topic_query(lda_questions, query).dropna(axis=1, how='all').head(2).T
