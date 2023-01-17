import pandas as pd
test = pd.read_csv('../input/test.csv', nrows=10)
last = test.Sequence.apply(lambda x: pd.Series(x.split(','))).mode(axis=1).fillna(0)
submission = pd.DataFrame({'Id': test['Id'], 'Last': last[0]})
submission.head(10)
#submission = pd.DataFrame({'Id': test['Id'], 'Last': last[0]})
