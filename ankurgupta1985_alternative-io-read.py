# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!wc -l /kaggle/input/tensorflow2-question-answering/sample_submission.csv
def select_random_long_answer(long_answer_candidates, seed=None):

    assert isinstance(long_answer_candidates, list)

    for long_answer in long_answer_candidates:

        assert isinstance(long_answer, dict)

        assert 'start_token' in long_answer

        assert 'end_token' in long_answer

    if seed is not None:

        np.random.seed(seed)

    index = np.random.randint(len(long_answer_candidates))

    return long_answer_candidates[index]



def select_random_short_answer(long_answer, seed=None):

    assert isinstance(long_answer, dict)

    assert 'start_token' in long_answer

    assert 'end_token' in long_answer

    if seed is not None:

        np.random.seed(seed)

    start_token = np.random.randint(low=long_answer['start_token'], high=long_answer['end_token'])

    end_token = np.random.randint(low=start_token, high=min(start_token + 10, long_answer['end_token'] + 1))

    return {'start_token': start_token, 'end_token': end_token}



def get_prediction_string(answer):

    assert isinstance(answer, dict)

    assert 'start_token' in answer

    assert 'end_token' in answer

    return '{}:{}'.format(answer['start_token'], answer['end_token'])



def get_answer_text(answer, document_text_tokens):

    assert isinstance(answer, dict)

    assert 'start_token' in answer

    assert 'end_token' in answer

    answer_tokens = document_text_tokens[answer['start_token']:answer['end_token'] + 1]

    answer_text = ' '.join(answer_tokens)

    return answer_text



def predict_on_chunk_dataframe(df, seed=None):

    assert isinstance(df, pd.DataFrame)

    if seed is not None:

        np.random.seed(seed)

    

    df['document_text_tokens'] = df['document_text'].apply(lambda s: s.split())

    df['long_answer'] = df['long_answer_candidates'].apply(lambda v: select_random_long_answer(v))

    df['short_answer'] = df['long_answer'].apply(lambda d: select_random_short_answer(d))

    df['long_answer_text'] = df.apply(lambda row: get_answer_text(row['long_answer'], row['document_text_tokens']), axis=1)

    df['short_answer_text'] = df.apply(lambda row: get_answer_text(row['short_answer'], row['document_text_tokens']), axis=1)

    df['long_answer_prediction_string'] = df['long_answer'].apply(get_prediction_string)

    df['short_answer_prediction_string'] = df['short_answer'].apply(get_prediction_string)



    # Order the columns

    assert len(set(df.columns)) == len(df.columns)

    ordered_columns = ['question_text', 'long_answer_text', 'short_answer_text', 'document_text']

    rest_columns = list(set(df.columns).difference(set(ordered_columns)))

    df = df[ordered_columns + rest_columns]

    return df



def generate_submission(df, seed=None):

    assert isinstance(df, pd.DataFrame)

    if seed is not None:

        np.random.seed(seed)

    

    df = predict_on_chunk_dataframe(df)

    long_predictions = (df[['example_id', 'long_answer_prediction_string']].copy()

                        .rename({'long_answer_prediction_string': 'PredictionString'}, axis=1))

    long_predictions['example_id'] = long_predictions['example_id'].apply(lambda s: '{}_long'.format(s))

    short_predictions = (df[['example_id', 'short_answer_prediction_string']].copy()

                         .rename({'short_answer_prediction_string': 'PredictionString'}, axis=1))

    short_predictions['example_id'] = short_predictions['example_id'].apply(lambda s: '{}_short'.format(s))

    

    submission_df = (pd.concat([long_predictions, short_predictions], axis=0, ignore_index=True)

                     .sort_values(by=['example_id'], ascending=True)

                     .reset_index(drop=True))

    return submission_df
import json

np.random.seed(42)

submission_chunks = []

with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', 'r') as f:

    for i, line in enumerate(f):

        parsed_line = json.loads(line)

        chunk_test_df = pd.DataFrame.from_records([parsed_line], index=[0])

        submission_chunks = submission_chunks + [generate_submission(chunk_test_df)]  # redefine over append
submission_df = (pd.concat(submission_chunks)

                 .sort_values(by=['example_id'], ascending=True)

                 .reset_index(drop=True))

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
n_examples = i + 1

print('Number of examples = {}'.format(n_examples))

with open('n-examples.csv', 'w') as f:

    f.writelines(str(n_examples))
!ls -lh submission.csv
!wc -l submission.csv
!head -10 submission.csv