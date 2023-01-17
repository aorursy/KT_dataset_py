# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header = None)

train_labels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header = None)

final_test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv',header = None)

from sklearn.linear_model import LogisticRegression

X,Y = train, np.ravel(train_labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

log_reg = LogisticRegression()

log_reg.fit(X_train,Y_train)

preds = log_reg.predict(X_test)

from sklearn import metrics

c = log_reg.score(X_test,Y_test)



submission = pd.DataFrame(log_reg.predict(final_test))

submission.columns = ['Solution']

# submission['ID'] = np.arrange(1, submission[0]+1)

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission.to_csv('submission.csv', index=False)


