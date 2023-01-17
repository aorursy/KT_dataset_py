import pandas as pd

pred_path = "../input/nq-sample-csv/first50.csv" #replace this with your own dataset.
df = pd.read_json('../input/tensorflow2-question-answering/simplified-nq-test.jsonl', lines = True, dtype={'example_id':'Object'})

submission = pd.DataFrame(index=pd.concat([df.example_id + '_short', df.example_id + '_long']), columns=['PredictionString']).sort_index()
updates = pd.read_csv(pred_path, na_filter=False).set_index('example_id').sort_index()

submission.loc[updates.index.intersection(submission.index),'PredictionString'] = updates['PredictionString']
submission.head(50)
submission.to_csv('submission.csv')