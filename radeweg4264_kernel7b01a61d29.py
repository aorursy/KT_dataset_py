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
from sklearn.model_selection import train_test_split

data_train=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')

data_test=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')
data_train.head()
df = pd.get_dummies(data_train.drop('type', axis = 1))

X_train, X_test, y_train, y_test = train_test_split(df, data_train['type'], test_size = 0.25, random_state = 0)
from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2, max_time_mins=5, max_eval_time_mins=0.04, population_size=40)

tpot.fit(X_train, y_train)
prediction=tpot.predict(X_test)
from sklearn.metrics import classification_report

print (classification_report(y_test, prediction))