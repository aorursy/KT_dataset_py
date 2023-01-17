import pandas as pd
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

csv1 = pd.read_csv('../input/covid19v7/submission.csv')

csv2 = pd.read_csv('../input/covid19features/submission.csv')
submission = (csv1.drop(['id_seqpos'], axis=1) + csv2.drop(['id_seqpos'], axis=1))/2

submission['id_seqpos'] = sub['id_seqpos']

cols = [submission.columns[-1]] + list(submission.columns[: -1])

submission = submission[cols]

submission.head()
submission.to_csv('submission.csv', index=False)