import pandas as pd
test = pd.read_csv('/kaggle/input/tryinclass/test.csv')

test[['funded_or_not']] = 1

test[['project_id', 'funded_or_not']].to_csv('submission.csv', index=False)