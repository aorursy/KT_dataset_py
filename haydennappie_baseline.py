import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] #copying test index for later
df_train.head()
train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']
rf = RandomForestClassifier(n_estimators=50, random_state=123)
rf.fit(train_X, train_y)
df_test = df_test.loc[:, 'V1':'V16']

pred = rf.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)