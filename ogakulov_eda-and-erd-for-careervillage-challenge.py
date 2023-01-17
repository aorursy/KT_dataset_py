# The usual suspects for data processing and visualization

import pandas as pd 

import seaborn as sns

import matplotlib as plt

import datetime



%matplotlib inline





# Load the questions data and process dates properly

questions = pd.read_csv('../input/questions.csv', parse_dates=['questions_date_added'])

min_q_date = min(questions['questions_date_added'])

max_q_date = max(questions['questions_date_added'])

print('There were {:,} questions asked between {} and {}'.format(questions.shape[0], min_q_date.strftime('%Y-%m-%d'), max_q_date.strftime('%Y-%m-%d')))



# Plot count of questions accross years

sns.set_style("white")

sns.countplot(x=questions['questions_date_added'].dt.year, data=questions, facecolor='darkorange').set_title('Volume of Questions per Year')

sns.despine();
answers = pd.read_csv('../input/answers.csv', parse_dates=['answers_date_added'])

min_a_date = min(answers['answers_date_added'])

max_a_date = max(answers['answers_date_added'])

print('There were {:,} answers provided between {} and {}'.format(answers.shape[0], min_a_date.strftime('%Y-%m-%d'), max_a_date.strftime('%Y-%m-%d')))



# Plot count of questions accross years

sns.set_style("white")

sns.countplot(x=answers['answers_date_added'].dt.year, data=answers, facecolor='darkorange').set_title('Volume of Answers per Year')

sns.despine();
q_a = questions.merge(right=answers, how='inner', left_on='questions_id', right_on='answers_question_id')

print('There are {:,} questions that got answered, which is {:.0f}% of all questions.'.format(q_a['questions_id'].nunique(), 100*q_a['questions_id'].nunique()/questions.shape[0]))