# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import xgboost as xgb



xgb.__version__
!pip install --upgrade xgboost
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_otto = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")

train_otto.shape
train_otto.head()
train_otto.info()
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from time import time

from matplotlib import pyplot
train_otto_arrays = train_otto.values

# split data into X and y

X = train_otto_arrays[:,0:94]

y = train_otto_arrays[:,94]

# encode string class values as integers

label_encoded_y = LabelEncoder().fit_transform(y)

# evaluate the effect of the number of threads

results = []

num_threads = [1, 2, 3, 4, 5]

for n in num_threads:

    start = time()

    model = XGBClassifier(nthread=n)

    model.fit(X, label_encoded_y)

    elapsed = time() - start

    print(n, elapsed)

    results.append(elapsed)
# plot results

pyplot.plot(num_threads, results)

pyplot.ylabel('Speed (in seconds)')

pyplot.xlabel('Number of Threads')

pyplot.title('XGBoost Training Speed vs Number of Threads')

pyplot.show()
# plot results

pyplot.plot(num_threads, results)

pyplot.ylabel('Speed (seconds)')

pyplot.xlabel('Number of Threads')

pyplot.title('XGBoost Training Speed vs Number of Threads')

pyplot.show()
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import time
# prepare cross validation

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

# Single Thread XGBoost, Parallel Thread CV

start = time.time()

model = XGBClassifier(nthread=1)

results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss',

n_jobs=-1)

elapsed = time.time() - start

print("Single Thread XGBoost, Parallel Thread CV: %f" % (elapsed))

# Parallel Thread XGBoost, Single Thread CV

start = time.time()

model = XGBClassifier(nthread=-1)

results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss',

n_jobs=1)

elapsed = time.time() - start

print("Parallel Thread XGBoost, Single Thread CV: %f" % (elapsed))

# Parallel Thread XGBoost and CV

start = time.time()

model = XGBClassifier(nthread=-1)

results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss',

n_jobs=-1)

elapsed = time.time() - start

print("Parallel Thread XGBoost and CV: %f" % (elapsed))