import pandas as pd
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()




# dumb - set everyone to rating=8
test_df['Rating'] = 8

# Make sure you write Rating as integer, not float 
test_df['Rating'] = test_df['Rating'].astype(int)
test_df[['Id','Rating']].to_csv('./sample_submission.csv', sep=",", index=False)