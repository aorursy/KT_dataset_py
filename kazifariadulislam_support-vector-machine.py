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
df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
y = df.target.values
x = df.drop(['target'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 200)

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train))

X_test = pd.DataFrame(scaler.transform(X_test))
from sklearn import svm
c = svm.SVC(probability=True)

c.fit(X_train, y_train)
probs = c.predict_proba(X_test)

prob = probs[:,1]
predictions = c.predict(X_test)
c = svm.SVC(C = 0.2, probability=True)
from sklearn.model_selection import cross_val_score

score = cross_val_score(c, X_train, y_train)

print(f"Cross Validation Accuracy: {round(score.mean()*100,2)}")
from sklearn.metrics import f1_score

svm_f1 = f1_score(y_test, predictions)

print(f"F1 score: {round(svm_f1*100, 2)}")