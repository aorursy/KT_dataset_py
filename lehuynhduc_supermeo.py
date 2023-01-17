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
train_df = pd.read_csv("/kaggle/input/sensordata/SData/Training_DataSet.csv")

test_df = pd.read_csv("/kaggle/input/sensordata/SData/Test_DataSet.csv")
# mapping = {"Lying":0,"Walking":1}

# train_df['label'] = train_df['label'].map(mapping)

# test_df['label'] = test_df['label'].map(mapping)

y_train = train_df['label']

X_train = train_df.drop('label',axis=1)

y_test = test_df['label']

X_test = test_df.drop('label',axis=1)

import seaborn as sns

import matplotlib.pyplot as plt



def sigmoid(x):

    return 1 / (1+ np.exp(-x))

for c in X_train.columns:

    plt.hist(X_train[c], bins=20, alpha=0.7,label="train")

    plt.hist(X_test[c], bins=20, alpha=0.7,label="test")

    plt.title(c)

    plt.legend()

    plt.show()
X_train.corr()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=15,

                               max_depth=3,

                               min_samples_leaf=16,

                               n_jobs=-1,

                               random_state=42)
model.fit(X_train_, y_train_)
y_pred = model.predict(X_valid_)
from sklearn.metrics import accuracy_score
accuracy_score(y_train_, model.predict(X_train_))
accuracy_score(y_valid_, model.predict(X_valid_))
accuracy_score(y_test, model.predict(sigmoid(X_test)))