import pandas as pd
!ls ../input
questions = pd.read_csv('../input/questions.csv')

answers = pd.read_csv('../input/answers.csv')

professionals = pd.read_csv('../input/professionals.csv')
questions.head()
answers.head()
professionals.head()
question_answers = questions.merge(right=answers, how='inner', left_on='questions_id', right_on='answers_question_id')
question_answers.head()
qa_professionals = question_answers.merge(right=professionals, left_on='answers_author_id', right_on='professionals_id')

qa_professionals.head()