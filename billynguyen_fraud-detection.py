import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
original_data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

original_data.head()
print('number of row and column',original_data.shape)
original_data.describe()
plt.figure(figsize=(15,15))

for i,v in enumerate(original_data.columns):

    plt.subplot(6,6,i+1)

    plt.hist(original_data[v])

    plt.title(v)

plt.tight_layout()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(original_data.drop(['Class','Time'],axis=1),original_data['Class'], test_size=0.3,random_state=1)
print('X_train shape',X_train.shape)

print("X_test shape ", X_test.shape)
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_hat = rf.predict(X_test)

print('accuracy score',accuracy_score(y_hat,y_test))