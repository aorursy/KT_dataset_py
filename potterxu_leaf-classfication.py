# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler





np.random.seed(42)



train = pd.read_csv('../input/train.csv')

x_train = train.drop(['id', 'species'], axis=1).values

le = LabelEncoder().fit(train['species'])

y_train = le.transform(train['species'])



scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)



params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005]}

log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)

clf.fit(x_train, y_train)



print("best params: " + str(clf.best_params_))

for params, mean_score, scores in clf.grid_scores_:

  print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

  print(scores)



test = pd.read_csv('../input/test.csv')

test_ids = test.pop('id')

x_test = test.values

scaler = StandardScaler().fit(x_test)

x_test = scaler.transform(x_test)



y_test = clf.predict_proba(x_test)



submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)

submission.to_csv('submission.csv')
train.describe()
####有图片数据可以识别

coef=clf.coef_

coef