import pandas as pd
sub = pd.read_csv('../input/stage_1_sample_submission.csv')
sub['PredictionString'] = ''
sub.head()
sub.to_csv('naive_submission.csv',index=False)