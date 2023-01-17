import numpy as np

import pandas as pd

import glob

from tqdm import tqdm

# from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_absolute_error

from sklearn.svm import SVR

from sklearn.model_selection import KFold
train_file = glob.glob('../input/predict-volcanic-eruptions-ingv-oe/train/*')

test_file = glob.glob('../input/predict-volcanic-eruptions-ingv-oe/test/*')
df = pd.read_csv(train_file[0])

df_mean = pd.DataFrame(df.mean()).T

df_mean['id'] = train_file[0].split('/')[-1].split('.')[0]

train_file.remove(train_file[0])

for file in tqdm(train_file):

    df = pd.read_csv(file)

    df_ = pd.DataFrame(df.mean()).T

    df_['id'] = file.split('/')[-1].split('.')[0]

    df_mean = pd.concat([df_mean,df_])

    del df

df_mean.head(3)
df_train = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train.csv')

df_train.head(2)
df_mean['id'] = df_mean['id'].astype('int64')
df_mean = df_mean.join(df_train.set_index('segment_id'), on='id')

df_mean.head(3)
X_train = df_mean.drop(['id','time_to_eruption'],axis=1)

y_train = df_mean['time_to_eruption']

X_train = X_train.fillna(X_train.mean())

del df_mean
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15)

# clf = LinearRegression().fit(X_train,y_train)

# y_pred = clf.predict(X_test)

# print('MAE:',mean_absolute_error(y_test,y_pred))
df = pd.read_csv(test_file[0])

df_mean_test = pd.DataFrame(df.mean()).T

df_mean_test['id'] = test_file[0].split('/')[-1].split('.')[0]

test_file.remove(test_file[0])

for file in tqdm(test_file):

    df = pd.read_csv(file)

    df_ = pd.DataFrame(df.mean()).T

    df_['id'] = file.split('/')[-1].split('.')[0]

    df_mean_test = pd.concat([df_mean_test,df_])

    del df

df_mean_test.head(3)
X_test = df_mean_test.fillna(df_mean_test.mean())

# X_test = X_test.drop(['id'],axis=1)

del df_mean_test
clfsvrpoly = SVR(kernel='poly').fit(X_train,y_train)

clfsvrrbf = SVR(kernel='rbf').fit(X_train,y_train)



y_pred = clfsvrpoly.predict(X_test.drop(['id'],axis=1))

y_pred += clfsvrrbf.predict(X_test.drop(['id'],axis=1))

X_test['y_pred'] = y_pred / 2
submission = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

X_test['id'] = X_test['id'].astype('int64')

submission = submission.join(X_test[['id','y_pred']].set_index('id'), on='segment_id')

del submission['time_to_eruption']

submission.rename({'y_pred':'time_to_eruption'},axis=1,inplace=True)

submission.to_csv('submission.csv',index=False)
!head submission.csv