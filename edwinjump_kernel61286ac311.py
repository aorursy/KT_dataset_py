import pandas as pd
df = pd.read_csv('../input/scl-dummy/Dummy data.csv')

df.head(5)
df['new_number'] = df['number'].apply(lambda x: x + 1)

del df['number']

df.head(5)
df.to_csv('../submission.csv', index=False)