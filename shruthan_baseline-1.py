import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score, f1_score

import numpy as np
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0']
df_train.dtypes
df_train.V6.describe()
# Mean and standard deviation suggest outliers in the data which are not useful 

df = df_train[df_train.V6 < 6000]

df = df[df.V6 > -3000]
(df_train.Class == 1).sum()
(df_train['Class'] == 0).sum()
from sklearn.model_selection import GridSearchCV

params = {"max_depth" : (3,4), "n_estimators" : (100, 150, 200, 250, 300, 350, 400, 450), 'loss' : ('exponential', 'deviance')}
from sklearn.model_selection import train_test_split

train_X = df.loc[:, 'V1':'V16']



train_y = df.loc[:, 'Class']

#train_X.drop('V2', axis = 'columns', inplace = True)



x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.3)



# V2 has many categories for binary classification. However dropping it doesnt help much
train_X.columns
gb = GradientBoostingClassifier(n_estimators = 350,random_state = 123, max_depth = 3, loss = 'exponential')

#gb= GradientBoostingClassifier()

#clf  = GridSearchCV(gb, params, cv = 5, scoring = 'roc_auc')

#clf.fit(train_X, train_y)
#(clf.best_params_)
gb.fit(x_train, y_train)


pred = gb.predict(x_test)
print(f1_score(y_test, pred, average='macro')) 

print(roc_auc_score(y_test, pred))
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0']
df_train = df_train[df_train.V6 < 6000]

df_train = df_train[df_train.V6 > -3000]
gb = GradientBoostingClassifier(n_estimators=350, random_state=123, loss = 'exponential', max_depth = 3)

#df_train.drop("V2", axis = 'columns', inplace = True)

train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']







gb.fit(train_X, train_y)

df_test.head()
df_test.columns
df_test = df_test.loc[:, 'V1':'V16']

#df_test.drop("V2", axis = 'columns', inplace = True)

pred = gb.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output1.csv', index=False)