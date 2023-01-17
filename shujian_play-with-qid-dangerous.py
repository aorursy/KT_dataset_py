import pandas as pd
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()
df = df_train[['qid', 'question_text']].append(df_test)
df[df.duplicated(['qid'], keep=False)]
for i in range(1, 21):
    df_train['first_' + str(i)] = df_train['qid'].apply(lambda x: x[:i])
    
df_train.head()
df_train.groupby('first_1')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)
df_train.groupby('first_2')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)
df_train.groupby('first_3')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)
df_train.groupby('first_4')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)
df_train.groupby('first_5')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)
df_2122 = df_train[df_train.first_4 == '2122']
for index, row in df_2122.iterrows():
    print(row['target'], ': ',row['question_text'])
df_623a = df_train[df_train.first_4 == '623a']
for index, row in df_623a.iterrows():
    print(row['target'], ': ',row['question_text'])
df_test['first_4'] = df_test['qid'].apply(lambda x: x[:4])
stat = df_train.groupby('first_4')['target'].agg(['mean']).reset_index()
result = pd.merge(df_test, stat, how='left', on=['first_4'])
result.head(n=100)