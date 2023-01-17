# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/heart.csv")

df.head()

df.info()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

sns.countplot(df['sex'])

plt.xlabel("(male:1, female:0)")
haveheartdiseases=df[df.target==1]
havenotheartdisease = df[df.target==0]
len(haveheartdiseases)
len(havenotheartdisease)
print('people have heart disease:',len(haveheartdiseases))
print('peopele do not have heart disease:', len(havenotheartdisease))
#using machinelearning
X=df.drop('target', axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=23)
linreg=LinearRegression()
linreg.fit(X_train, y_train)
linreg.predict(X_test)
linreg.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.predict(X_test)
logreg.score(X_test, y_test)