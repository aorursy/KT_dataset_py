import os
print((os.listdir('../input/')))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')

df_train.head()
train_X = df_train.loc[:, 'X1':'X23']
train_y = df_train.loc[:, 'Y']
rf = RandomForestClassifier(n_estimators=50, random_state=123)
rf.fit(train_X, train_y)
df_test = df_test.loc[:, 'X1':'X23']
pred = rf.predict_proba(df_test)
result = pd.DataFrame(pred[:,1])
result.index.name = 'id'
result.columns = ['predicted_val']
result.to_csv('output.csv', index=True)
