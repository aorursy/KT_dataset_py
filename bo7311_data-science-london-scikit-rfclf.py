# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename == 'train.csv':

            train=pd.read_csv(os.path.join(dirname, filename), header=None)

        elif filename == 'test.csv':

            test=pd.read_csv(os.path.join(dirname, filename), header=None)

        else:

            trainLabels=pd.read_csv(os.path.join(dirname, filename), header=None)

print(trainLabels)

# Any results you write to the current directory are saved as output.
print(train.shape)

print(test.shape)

print(trainLabels.shape)

#Split Data

X_train, X_test, y_train, y_test =train_test_split(train,trainLabels,train_size=0.7,random_state=42)

#Build RandomForestClassifier Model

RF=RandomForestClassifier(n_estimators=200,criterion="entropy",n_jobs=-1,random_state=22,verbose=1)

#Training

RF.fit(X_train,y_train)

#Score

score=RF.score(X_test,y_test)

print(score)
# predict test

test_predicted = RF.predict(test)

print(test_predicted)



submission = pd.DataFrame(test_predicted)

print(submission.shape)



submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission

submission.to_csv('submission.csv', index=False)
