import pandas as pd



long_answers = pd.DataFrame(columns = ['example_id' , 'PredictionString'])

short_answers = pd.DataFrame(columns = ['example_id' , 'PredictionString'])
import json



with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', 'rt') as f:

    for i, l in enumerate(f):

        example = json.loads(l)

        

        example_id = example['example_id']

        

        long_answers.loc[i] = [example_id, '']

        short_answers.loc[i] = [example_id, '']
long_answers['example_id'] = long_answers['example_id'].apply(lambda example_id: str(example_id) + '_long')

short_answers['example_id'] = short_answers['example_id'].apply(lambda example_id: str(example_id) + '_short')



submission = pd.concat([long_answers, short_answers], axis=0)



submission.sort_values(by='example_id').to_csv('submission.csv', index=False)