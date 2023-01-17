'''

This is the fourth homework of NLP course.



The aim of the homework is building a faq answer system based on sentence similariy calculation



A "faq and their answers from a bank" provided to the application.

'''
import sqlite3  # The questions and answers are in database

import pandas as pd

import nltk

import numpy as np

from scipy.spatial.distance import cosine
conn = sqlite3.connect('../input/faq.db')

conn = conn.cursor()
conn.execute("SELECT Question from faq_answers")  # Print all questions to see for testing the application



print(conn.fetchall())
question = "Do I need to type # after keying in my number"  #  The original question is: 'Do I need to enter ‘#’ after keying in my Card number
conn.execute("SELECT Question from faq_answers where Question=?", (question,))  # Find the question if the exact version exists in the database



data = conn.fetchone()
q_token = nltk.word_tokenize(question.lower())
cos_sim = []

check1 = pd.DataFrame(columns=['vocabulary', 'tf1', 'tf2', 'tf_norm1', 'tf_norm2', 'idf', 'tf*idf1', 'tf*idf2'])

q_df = pd.DataFrame(columns=['tokens'])

q_df['tokens'] = pd.Series(q_token)

row_df = pd.DataFrame(columns=['tokens'])
if data is None:  # If the question's exact version is not in db

    for row in conn.execute("SELECT Question from faq_answers"):

        str = ''.join(row)

        vocabulary = question + " " + str

        # check1['tf1'] = pd.Series()

        row_token = nltk.word_tokenize(str.lower())

        row_df['tokens'] = pd.Series(row_token)

        # check1['tf2'] = pd.Series(q_token2)

        vocab_token = nltk.word_tokenize(vocabulary.lower())

        vocab_token = list(dict.fromkeys(vocab_token))

        check1['vocabulary'] = pd.Series(vocab_token)

        count = 0

        for r in vocab_token:

            hold = q_df.loc[q_df['tokens'] == r]

            hold2 = row_df.loc[row_df['tokens'] == r]

            check1['tf1'][count] = len(hold)  # Find tf

            check1['tf2'][count] = len(hold2)

            hold3 = len(hold) + len(hold2)

            if hold3 == 1:

                check1['idf'][count] = 1 + np.log(2 / 1) # find idf

            else:

                check1['idf'][count] = 1 + np.log(2 / 2)

            count += 1

        check1['tf_norm1'] = check1['tf1'] / check1['tf1'].max() # find tf_norm

        check1['tf_norm2'] = check1['tf2'] / check1['tf2'].max()

        check1['tf*idf1'] = check1['tf_norm1'] * check1['idf']

        check1['tf*idf2'] = check1['tf_norm2'] * check1['idf']  # find tf-idf

        cosine_sim = 1 - cosine(check1['tf*idf1'], check1['tf*idf2'])

        cos_sim.append(cosine_sim)                              # Apply cosine similarity as similarity func.

    cosine_sim_index = cos_sim.index(max(cos_sim)) + 1

    print(max(cos_sim))                                         # Print max of cosine similariy, most similiar one

    conn.execute("Select Question from faq_answers where ROWID = ?", (cosine_sim_index,))

    print("Initial question: ('" + question + "')")

    print("Question: ", conn.fetchone())

    conn.execute("Select Answer from faq_answers where ROWID = ?", (cosine_sim_index,))

    print("Answer: ", conn.fetchone())

else:

    conn.execute("SELECT Question from faq_answers where Question=?", (question,)) # Directly print the question

    print("Initial question: ('" + question + "')")

    print("Question: ", conn.fetchone())

    conn.execute("SELECT Answer from faq_answers where Question=?", (question,))

    print("Answer: ", conn.fetchone())