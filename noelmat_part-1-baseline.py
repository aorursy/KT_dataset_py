# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, train_test_split
train_data = pd.read_csv('/kaggle/input/learn-together/train.csv',index_col='Id')

test_data = pd.read_csv('/kaggle/input/learn-together/test.csv',index_col='Id')
X_train, X_val, y_train, y_val = train_test_split(train_data.drop('Cover_Type',axis=1),train_data['Cover_Type'],test_size=0.3,random_state=10)
def rmse(x,y): return np.sqrt(((x-y)**2).mean())



def print_scores(m):

    res = [rmse(m.predict(X_train),y_train),rmse(m.predict(X_val),y_val),m.score(X_train,y_train),m.score(X_val,y_val)]

    print(res)

    
# estimators = [20,30,40,50,80,100,140,160,200]

# for est in estimators:

#     rf = RandomForestClassifier(n_jobs=-1, n_estimators=est,random_state=10,min_samples_leaf=5,max_features=.5)

#     rf.fit(X_train,y_train)

#     print('est : {} :: {}'.format(est,print_scores(rf)))
# rf = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=10,min_samples_leaf=5,max_features=.5)

rf = RandomForestClassifier(n_jobs=-1,random_state=10)

rf.fit(X_train,y_train)

print_scores(rf)
def create_submission(model,test_df):

    submit = pd.DataFrame({'Cover_Type':model.predict(test_df)},index=test_df.index)

    submit.head()

    submit.to_csv('submission.csv')
create_submission(rf,test_data)