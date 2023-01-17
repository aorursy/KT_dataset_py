# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# This Python 3 environment comes with many helpful analytics libraries installed

# This Python 3 environment comes with many helpful analytics libraries installed

df = pd.read_csv('../input/xAPI-Edu-Data.csv')

print(df.shape)

df.head()
# Resampling and renaming data

subset_df = df[['gender', 'raisedhands','VisITedResources','Discussion','AnnouncementsView', 'StudentAbsenceDays','Class']]

subset_df['HighPerformance'] = subset_df.Class.map({'H':1, 'L':0, 'M': 0})

subset_df['IsMale'] = subset_df.gender.map({'F':0, 'M': 1})

subset_df['MoreThan7Absences'] = subset_df.StudentAbsenceDays.map({'Above-7':1, 'Under-7': 0})

del subset_df['Class']

del subset_df['StudentAbsenceDays']

del subset_df['gender']



subset_df.tail()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



y = subset_df.HighPerformance.values

del subset_df['HighPerformance']

X = subset_df.values



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1337)
logreg = LogisticRegression(C=1e5)

y_pred = logreg.fit(X_train, y_train).predict(X_test)
logreg.score(X_test,y_test)
confusion_matrix(y_test, y_pred)