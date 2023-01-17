path = '../input/riiid-test-answer-prediction/'

import pandas as pd

question = pd.read_csv(path + 'questions.csv')
question['part'].unique()
question.groupby('part')['correct_answer'].unique()
# 1 question for each bundle

question.query('part == 1').groupby('bundle_id').count().head()
# 1 question for each bundle

question.query('part == 2').groupby('bundle_id').count().head()
# many questions for each bundle

question.query('part == 3').groupby('bundle_id').count().head()
# many questions for each bundle

question.query('part == 4').groupby('bundle_id').count().head()
# 1 question for each bundle

question.query('part == 5').groupby('bundle_id').count().head()
# many questions for each bundle

question.query('part == 6').groupby('bundle_id').count().head()
# many questions for each bundle

question.query('part == 7').groupby('bundle_id').count().head()