import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



submission = pd.read_csv('../input/bertpred3/sub.csv')



print(submission.head())

submission.to_csv('submission.csv', index=False)