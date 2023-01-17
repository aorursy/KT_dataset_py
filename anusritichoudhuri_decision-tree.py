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

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('../input/apndcts/apndcts.csv')

print(df.head())

X=df.drop('class',axis=1)

y=df['class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

dtree=DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions=dtree.predict(X_test)

print('Classification Report:')

print(classification_report(y_test,predictions))

print('\n')

print('Confusion Matrix:')

print(confusion_matrix(y_test,predictions))








