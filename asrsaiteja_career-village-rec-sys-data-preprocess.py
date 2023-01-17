import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



data_paths = {}



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # print(filename)

        data_paths[filename] = os.path.join(dirname, filename)
df_answer_scores = pd.read_csv(data_paths['answer_scores.csv'])



df_answers = pd.read_csv(data_paths['answers.csv'],parse_dates = ['answers_date_added'])



df_professionals = pd.read_csv(data_paths['professionals.csv'],parse_dates=['professionals_date_joined'])



df_question_scores = pd.read_csv(data_paths['question_scores.csv'])



df_questions = pd.read_csv(data_paths['questions.csv'],parse_dates=['questions_date_added'])



df_students = pd.read_csv(data_paths['students.csv'],parse_dates=['students_date_joined'])



df_tag_questions = pd.read_csv(data_paths['tag_questions.csv'])



df_tag_users = pd.read_csv(data_paths['tag_users.csv'])



df_tags = pd.read_csv(data_paths['tags.csv'])
def create_unique_numeric_id(df, col_name):

    df[col_name] = np.arange(len(df))

    return df.reset_index(drop=1)
df_professionals = create_unique_numeric_id(df_professionals, 'prof_uid')

df_questions = create_unique_numeric_id(df_questions, 'ques_uid')

df_answers = create_unique_numeric_id(df_answers, 'ans_uid')

df_students = create_unique_numeric_id(df_students, 'stud_uid')
# just dropna from tags 

df_tags = df_tags.dropna()

df_tags.loc[:,'tags_tag_name'] = df_tags.loc[:,'tags_tag_name'].apply(lambda x: x.encode('ascii','ignore').decode())

df_tags.loc[:,'tags_tag_name'] = df_tags.loc[:,'tags_tag_name'].str.replace('#', '')
# Questions Tags

# merge tag_questions with tags name & group all tags for each question into single rows

df_tags_question = df_tag_questions.merge(df_tags, how='inner',

    left_on='tag_questions_tag_id', right_on='tags_tag_id')



df_tags_question = df_tags_question.groupby(

    ['tag_questions_question_id'])['tags_tag_name'].apply(','.join).reset_index()



df_tags_question = df_tags_question.rename(columns={'tags_tag_name': 'questions_tag_name'})
# Professionals Tags

# merge tag_users with tags name, group all tags for each user into single rows & rename the tag column name 

df_tags_pro = df_tag_users.merge(

    df_tags, how='inner',left_on='tag_users_tag_id', right_on='tags_tag_id')



df_tags_pro = df_tags_pro.groupby(['tag_users_user_id'])['tags_tag_name'].apply(

    ','.join).reset_index()



df_tags_pro = df_tags_pro.rename(columns={'tags_tag_name': 'professionals_tag_name'})
# merge professionals and questions tags create above with main datasets 

df_questions = df_questions.merge(df_tags_question, how='left',

    left_on='questions_id', right_on='tag_questions_question_id')



df_professionals = df_professionals.merge(df_tags_pro, how='left',

    left_on='professionals_id', right_on='tag_users_user_id')
# merge questions with scores 

df_questions = df_questions.merge(df_question_scores, how='left',

                                  left_on='questions_id', right_on='id')



# merge questions with students 

df_questions = df_questions.merge(df_students, how='left',

    left_on='questions_author_id', right_on='students_id')
# merge answers with questions,  merge professionals and questions score with that 

df_question_answer = df_answers.merge(df_questions, how='inner',

                                      left_on='answers_question_id', right_on='questions_id')

df_merge = df_question_answer.merge(df_professionals, how='inner',

                                    left_on='answers_author_id', right_on='professionals_id')

df_merge = df_merge.merge(df_question_scores, how='inner',

                          left_on='questions_id', right_on='id')
# Generate some features for calculates weights to use with interaction matrix

# df_merge['num_ans_by_professional'] = df_merge.groupby(['answers_author_id'])['questions_id'].transform('count')



df_merge['num_ans_per_ques'] = df_merge.groupby(['questions_id'])['answers_id'].transform('count')

df_merge['num_tags_professional'] = df_merge['professionals_tag_name'].str.split(",").str.len()

df_merge['num_tags_question'] = df_merge['questions_tag_name'].str.split(",").str.len()
# Merge professionals previous answered questions tags into professionals tags 

# select professionals answered questions tags and stored as a dataframe



professionals_prev_ans_tags = df_merge[['professionals_id', 'questions_tag_name']]



# drop null values 

professionals_prev_ans_tags = professionals_prev_ans_tags.dropna()



# group all of tags of each user into single row 

professionals_prev_ans_tags = professionals_prev_ans_tags.groupby(

    ['professionals_id'])['questions_tag_name'].apply(

        ','.join).reset_index()



# drop duplicates tags from each professionals rows

professionals_prev_ans_tags['questions_tag_name'] = (

    professionals_prev_ans_tags['questions_tag_name'].str.split(',').apply(set).str.join(','))



# finally merge the dataframe with professionals dataframe 

df_professionals = df_professionals.merge(professionals_prev_ans_tags, how='left', on='professionals_id')



# join professionals tags and their answered tags 

# we replace nan values with ""

df_professionals['professional_all_tags'] = (

    df_professionals[['professionals_tag_name', 'questions_tag_name']].apply(

        lambda x: ','.join(x.dropna()),

        axis=1))
df_questions['score'] = df_questions['score'].fillna(0).astype(int)



# fill nan with 'No Tag' if any 

df_questions['questions_tag_name'] = df_questions['questions_tag_name'].fillna('No Tag')

df_professionals['professional_all_tags'] = df_professionals['professional_all_tags'].fillna('No Tag')



# replace "" with "No Tag", because previously we replace nan with ""

df_questions['questions_tag_name'] = df_questions['questions_tag_name'].replace('', 'No Tag')

df_professionals['professional_all_tags'] = df_professionals['professional_all_tags'].replace('', 'No Tag')



# remove duplicates tags from each professionals and each questions 

df_questions['questions_tag_name'] = df_questions['questions_tag_name'].str.split(',').apply(set).str.join(',')

df_professionals['professional_all_tags'] = df_professionals[

    'professional_all_tags'].str.split(',').apply(set).str.join(',')
# remove some null values from df_merge

df_merge['num_ans_per_ques']  = df_merge['num_ans_per_ques'].fillna(0)

df_merge['num_tags_professional'] = df_merge['num_tags_professional'].fillna(0)

df_merge['num_tags_question'] = df_merge['num_tags_question'].fillna(0)



# calculate our weight value 

df_merge['total_weights'] = 1 / (df_merge['num_ans_per_ques'])
df_questions.shape, df_professionals.shape, df_merge.shape
df_professionals[['prof_uid', 'professional_all_tags']].to_csv('professionals_clean.csv', index = None)

df_questions[['ques_uid','questions_tag_name','questions_title','questions_body']].to_csv('questions_clean.csv', 

                                                                                          index = None)

df_merge[['prof_uid', 'ques_uid', 'total_weights']].to_csv('prof_ques_interactions.csv', index = None)