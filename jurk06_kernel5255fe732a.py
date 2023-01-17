# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv')

test=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')

feat_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv')

train.fillna(train.mean(), inplace=True)

test.fillna(train.mean(), inplace=True)

df=pd.merge(train, feat_train)

feature_col=['city','week_start_date', 'total_cases','weekofyear']
X=df.drop(feature_col, axis=1)
y=df.iloc[:,24]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
np.mean(score)*100

feature_col=['city','week_start_date', 'weekofyear']
test_data=test.drop(feature_col, axis=1).copy()
clf=SVC()
clf.fit(X, y)

pred=clf.predict(test_data)
submission = pd.DataFrame({
    "city":test["city"],
    "year":test['year'],
    "weekofyear":test['weekofyear'],
        "total_cases": pred
})

submission.to_csv('submission_format.csv', index=False)
submission = pd.read_csv('submission_format.csv')
submission.head()
