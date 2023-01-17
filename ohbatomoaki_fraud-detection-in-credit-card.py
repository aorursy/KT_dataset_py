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
df_train = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv', index_col=0)
from sklearn.ensemble import RandomForestClassifier



X = df_train.drop('Class', axis=1).values

y = df_train['Class'].values

clf = RandomForestClassifier(n_estimators=30, criterion='gini')

clf.fit(X, y)
#from sklearn.model_selection import cross_val_score

#scores = cross_val_score(clf, X, y, cv=3)

#scores.mean()

#0.48360355575348146cv=5n=10

#0.476594686947901cv=3n=10

#0.5388839834724731cv=3n=20

#0.5214498746996362cv=3n=50

#0.560975827239827cv=3n=35

#0.5511381218224478cv=3n=40
predict = clf.predict(df_test)
submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv')

submit['Class'] = predict

submit.to_csv('submission.csv', index=False)