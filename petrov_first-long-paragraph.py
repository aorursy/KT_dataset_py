import json

from pathlib import Path

from tqdm import tqdm



import pandas as pd



TEST_TOTAL = 346



PATH = Path('/kaggle/input/tensorflow2-question-answering')

!ls -1 {PATH}
def get_joined_tokens(answer: dict) -> str:

    return '%d:%d' % (answer['start_token'], answer['end_token'])



def get_pred(json_data: dict) -> dict:

    ret = {'short': 'YES', 'long': ''}

    candidates = json_data['long_answer_candidates']

    

    paragraphs = []

    tokens = json_data['document_text'].split(' ')

    for cand in candidates:

        start_token = tokens[cand['start_token']]

        if start_token == '<P>' and cand['top_level'] and cand['end_token']-cand['start_token']>35:

            break

    else:

        cand = candidates[0]

        

    ret['long'] = get_joined_tokens(cand)

    

    id_ = str(json_data['example_id'])

    ret = {id_+'_'+k: v for k, v in ret.items()} 

    return ret



preds = dict()



with open(PATH / 'simplified-nq-test.jsonl', 'r') as f:

    for line in tqdm(f, total=TEST_TOTAL):

        json_data = json.loads(line) 

        prediction = get_pred(json_data)

        preds.update(prediction)

            

submission = pd.read_csv(PATH / 'sample_submission.csv')

submission['PredictionString'] = submission['example_id'].map(lambda x: preds[x])

submission.to_csv('submission.csv', index=False)