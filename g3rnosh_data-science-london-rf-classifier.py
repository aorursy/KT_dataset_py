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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/data-science-london-scikit-learn/train.csv', header=None)

trainLabel = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv', header=None)

test = pd.read_csv('../input/data-science-london-scikit-learn/test.csv', header=None)
print('train shape:', train.shape)

print('test shape:', test.shape)

print('trainLabel shape:', trainLabel.shape)

train.head()
from sklearn.model_selection import cross_val_score, train_test_split



X, y = train, np.ravel(trainLabel)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(max_features = 40, random_state = 0)

clf.fit(X_train, y_train)



print('Data Science London Dataset')

print('Accuracy of RF Classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



print('Accuracy of RF Classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
testf = np.nan_to_num(test)

submission = pd.DataFrame(clf.predict(testf))

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission.to_csv('submission.csv', index=False)
