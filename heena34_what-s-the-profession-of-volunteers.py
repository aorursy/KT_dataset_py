import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)

pd.set_option('max_colwidth', 1000)
path = '../input/'

answers = pd.read_csv(path+'answers.csv')

comments = pd.read_csv(path+'comments.csv')

emails = pd.read_csv(path+'emails.csv')

group_memberships = pd.read_csv(path+'group_memberships.csv')

groups = pd.read_csv(path+'groups.csv')

matches = pd.read_csv(path+'matches.csv')

professionals = pd.read_csv(path+'professionals.csv')

questions = pd.read_csv(path+'questions.csv')

school_memberships = pd.read_csv(path+'school_memberships.csv')

students = pd.read_csv(path+'students.csv')

tag_users = pd.read_csv(path+'tag_users.csv')

tag_questions = pd.read_csv(path+'tag_questions.csv')

questions.head()
question_answer = questions.merge(answers, left_on='questions_id', right_on='answers_question_id')

question_answer_prof = question_answer.merge(professionals, left_on='answers_author_id', right_on='professionals_id')

question_answer_prof.head()
print('Total number of questions: ', len(questions['questions_id']))

print('Total number of volunteers on the platform who answer the questions: ', len(answers['answers_author_id'].unique()))
plt.figure(figsize=(30,15))

prof_count = pd.DataFrame({'Profession': question_answer_prof.groupby('professionals_id')['professionals_industry'].unique().str[0].values})

prof_count['Profession'].value_counts()[:20].plot(kind='barh')

plt.title('Profession of Volunteers', size=25)

plt.tick_params(labelsize=25)