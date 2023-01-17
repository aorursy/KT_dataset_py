import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
input_dir = "../input"



emails = pd.read_csv(os.path.join(input_dir, 'emails.csv'))

matches = pd.read_csv(os.path.join(input_dir, 'matches.csv'))



questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))

answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))



emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'])

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'])

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'])
emails_matches = emails.merge(

    matches, left_on='emails_id', right_on='matches_email_id')[

    ['emails_date_sent', 'emails_frequency_level', 'emails_recipient_id', 

      'matches_question_id']]
questions_professionals = questions.merge(

    emails_matches, left_on='questions_id', right_on='matches_question_id')[

    ['questions_id','questions_author_id', 'questions_date_added', 'emails_recipient_id']]
questions_professionals_answers = questions_professionals.merge(

    answers[['answers_question_id', 'answers_author_id']], 

    left_on='questions_id', right_on='answers_question_id', how='left')
negative_examples = questions_professionals_answers[

    questions_professionals_answers['emails_recipient_id'] != 

    questions_professionals_answers['answers_author_id']]



negative_examples = negative_examples[

    ['questions_id', 'questions_author_id', 'questions_date_added', 'emails_recipient_id']]

negative_examples = negative_examples.rename(columns={'emails_recipient_id': 'answer_user_id'})

negative_examples['matched'] = 0
positive_examples = questions.merge(

    answers, left_on='questions_id', right_on='answers_question_id', how='inner')[

    ['questions_id', 'questions_author_id', 'questions_date_added', 'answers_author_id']]

positive_examples = positive_examples.rename(columns={'answers_author_id': 'answer_user_id'})

positive_examples['matched'] = 1
combined_examples = pd.concat([negative_examples, positive_examples], axis=0)



combined_examples = combined_examples.merge(emails_matches,

                                            left_on=['answer_user_id', 'questions_id'],

                                            right_on=['emails_recipient_id', 'matches_question_id'],

                                            how='left')[['questions_date_added', 'questions_author_id', 

                                                         'questions_id', 'answer_user_id', 'emails_date_sent', 'matched']]
combined_examples = combined_examples.sort_values(

    by=['questions_date_added', 'questions_author_id', 'questions_id', 'answer_user_id', 'emails_date_sent'])

combined_examples.to_parquet('positive_negative_examples.parquet.gzip', compression='gzip')