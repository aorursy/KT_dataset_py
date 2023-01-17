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

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')

X = dataset.iloc[:, 2:4].values

y = dataset.iloc[:, 4].values
import seaborn as sns

sns.scatterplot(x='Age', y='EstimatedSalary', data=dataset, hue='Purchased')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)  

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)   
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

X_valid = sc_X.fit_transform(X_valid)

from catboost import CatBoostClassifier



params = {'loss_function':'Logloss', # objective function

          'eval_metric':'AUC', # metric

          'verbose': 200, # output to stdout info about training process every 200 iterations

          'random_seed': 1

         }

classifier = CatBoostClassifier(**params)

classifier.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)

          eval_set=(X_valid, y_valid), # data to validate on

          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score

          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)

         );
# Predicting the Test Set results

y_pred = classifier.predict(X_test)
y_train.dtype
y_pred.dtype
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.round())

print(cm)
