# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv",header=0, delimiter=',')
df_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv",header=0, delimiter=",")

id_array = df_test['id']

X_test = df_test.drop(['id'],axis = 1)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
label_0_count, label_1_count = df.target.value_counts()



# Split according to label

df0 = df[df['target'] == 0]

df1 = df[df['target'] == 1]



train_0 = df0.sample(label_1_count,random_state = 121)

train = pd.concat([train_0, df1], axis=0)



print(train.target.value_counts())



train.target.value_counts().plot(kind='bar', title='Count (target)');
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,MinMaxScaler



train = shuffle(train)



y_train = train['target']

X_train = train.drop(['id','target'],axis = 1)



scalar = StandardScaler()

norm = scalar.fit(X_train)



X_train = norm.transform(X_train)

X_test = norm.transform(X_test)

model = XGBClassifier()

n_est = [50,100,150]

max_depth = [1,2,5,6]

min_child_weight=[2,5,6,9]

subsample=[0.1,0.2,0.8]

colsample_bytree=[0.1,0.2,0.8]

param= dict(subsample=subsample,colsample_bytree=colsample_bytree,n_estimators=n_est,min_child_weight=min_child_weight,max_depth = max_depth)



grid_search = GridSearchCV(model, param, scoring="roc_auc",n_jobs=-1,verbose=1)

grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
y_test_predicted = grid_search.predict(X_test)

y_test_predicted_prob = grid_search.predict_proba(X_test)
my_submission = pd.DataFrame({'id': id_array, 'target': y_test_predicted_prob[:,1] })

my_submission.to_csv("submission_file.csv", index=False)