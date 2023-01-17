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

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/train-cleaned/train_cleaned.csv')

df.head()
X=df.drop(columns=['Survived'])

y=df[['Survived']]

print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=SVC(kernel='linear',C=1).fit(X_train, y_train)

model.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

clf=SVC(kernel='linear', C=1)

scores=cross_val_score(clf,X,y,cv=4)

scores
from sklearn.model_selection import cross_val_score

clf=SVC(kernel='linear', C=1)

scores=cross_val_score(clf,X,y,cv=6)

scores