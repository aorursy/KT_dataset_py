import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Lasso, Ridge
train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

submission_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
TIME_SET = 50000

REPEAT = 100

def set_time(df):

    df_time = df['time'].values

    df_time_0 = df_time[:TIME_SET]

    for i in range(1,REPEAT):

        df_time_0 = np.hstack([df_time_0, df_time[i*TIME_SET:(i+1)*TIME_SET]])

    df['time'] = df_time_0

    return df
train_df = set_time(train_df)

test_df = set_time(test_df)

X = train_df[['time', 'signal']].values

y = train_df['open_channels'].values

model_l = Lasso()

model_l.fit(X, y)
model_r = Ridge()

model_r.fit(X, y)
X_test = test_df[['time', 'signal']].values

predictions_l = model_l.predict(X_test)

predictions_l = np.clip(predictions_l, 0, 10)

predictions_l = predictions_l.astype(int)



predictions_r = model_r.predict(X_test)

predictions_r = np.clip(predictions_r, 0, 10)

predictions_r = predictions_r.astype(int)

submission_df['open_channels'] =  (np.round(0.5 * predictions_l + 0.50 * predictions_r, 0)).astype(int)
submission_df['time'] = [format(submission_df.time.values[x], '.4f') for x in range(2000000)]

submission_df.to_csv('submission.csv', index=False)