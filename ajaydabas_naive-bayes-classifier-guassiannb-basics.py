# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gender_data = pd.read_csv("../input/voicegender/voice.csv")

gender_data.head()
print(gender_data.shape)
gender_data.isnull().sum()
print(gender_data.info())
gender_data['label'].unique()
gender_data["label"].value_counts()
gender_data.describe()
gender_data.columns
cols = gender_data.columns

features = cols[0:6]

labels = cols[6]

print(features)

print(labels)
X = gender_data.iloc[:,:-1].values

y = gender_data.iloc[:,-1 ].values

X
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 10)

X_train, y_train
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

# model

clf_g = GaussianNB()



# model fitting

clf_g.fit(X_train, y_train)



results = cross_val_score(clf_g, X_train, y_train)

results.mean()
y_pred = clf_g.predict(X_valid)

y_pred
print(confusion_matrix(y_valid, y_pred))