import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.metrics import roc_auc_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/detecting-anomalies-in-wafer-manufacturing/Train.csv')

df_test = pd.read_csv('/kaggle/input/detecting-anomalies-in-wafer-manufacturing/Train.csv')
df.info()
duplicate=df.drop('Class',axis=1).T.drop_duplicates().T.columns

Class = df['Class']

df_test = df_test[duplicate]

df = df[duplicate]

df['Class'] = Class
df.info()
df.head()
sns.set(style="whitegrid")

ax = sns.boxplot(x="Class", y="feature_1",data=df)
ax = sns.boxplot(x="Class", y="feature_2",data=df)
ax = sns.boxplot(x="Class", y="feature_3",data=df)
pd.DataFrame(df[['feature_1','feature_2','feature_3','Class']].corr()['Class'])

zero = pd.DataFrame((df == 0).astype(int).sum(axis=0))
zero
all_zero = zero[zero[0]>1761].index
df.drop(all_zero,axis=1,inplace=True)
df_test.drop(all_zero,axis=1,inplace=True)
df.info()
X = df.drop('Class',axis=1)

y = df['Class'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from xgboost import XGBClassifier



model = XGBClassifier(silent=True,

                      booster = 'gbtree',

                      scale_pos_weight=5,

                      learning_rate=0.01,  

                      colsample_bytree = 0.7,

                      subsample = 0.5,

                      max_delta_step = 3,

                      reg_lambda = 2,

                     objective='binary:logistic',

                      

                      n_estimators=818, 

                      max_depth=8,

                     )



eval_set = [(X_test, y_test)]

eval_metric = ["logloss"]

%time model.fit(X_train, y_train,early_stopping_rounds=50, eval_metric=eval_metric, eval_set=eval_set)

predictions = model.predict_proba(X_test)[:,-1]
roc_auc_score(y_test, predictions)
import matplotlib.pyplot as plt     

model.feature_importances_

from matplotlib import pyplot

from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(18,10))

plot_importance(model, max_num_features=35, height=0.8, ax=ax)

pyplot.show()
p1 = model.predict_proba(df_test)[:,-1]

submission = pd.DataFrame(p1)

submission = submission.rename(columns={0: "Class"})

submission.index = submission['Class']

submission.drop('Class',axis=1,inplace=True)

#submission.to_csv('submissiom.csv',header=True, index=True)