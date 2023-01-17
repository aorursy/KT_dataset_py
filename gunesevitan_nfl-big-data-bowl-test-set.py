from kaggle.competitions import nflrush

env = nflrush.make_env()



import pandas as pd
df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

df_test = pd.DataFrame(columns=df_train.drop(columns=['Yards']).columns)



print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Training Plays = {}'.format(df_train['PlayId'].nunique()))

print('Number of Training Games = {}'.format(df_train['GameId'].nunique()))

print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
%%time



for (test, sample_pred) in env.iter_test():

    df_test = pd.concat([df_test, test])

    env.predict(sample_pred)

    

print('Number of Test Examples = {}'.format(df_test.shape[0]))

print('Number of Test Plays = {}'.format(df_test['PlayId'].nunique()))

print('Number of Test Games = {}'.format(df_test['GameId'].nunique()))

print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
df_test.head()
# Test set is chronologically sorted

(df_test['PlayId'] == sorted(df_test['PlayId'])).all()
df_test.to_csv('df_test.csv', chunksize=50000, index=False)