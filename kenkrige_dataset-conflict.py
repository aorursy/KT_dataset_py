import pandas as pd
sample = pd.read_csv('/kaggle/input/tensorflow2-question-answering/sample_submission.csv').set_index('example_id')

sample.sort_index().head()
test = pd.read_json('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', orient = 'records', lines = True).set_index('example_id')

test.index = test.index.astype(str)

test.sort_index().head()
test2 = pd.read_json('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl',

                    dtype = {'example_id': 'Object'},

                    lines = True).set_index('example_id')

test2.sort_index().head()