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

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split 
digit=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

digit_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print('Number of rows:',digit.shape[0],'Number of columns:',digit.shape[1])
print('target values')

print(sorted(digit.label.unique()))
print(digit.info())
print("columns which have null values or missing values")

print(digit.columns[digit.isnull().any()])
y=digit.label

X=digit.drop('label',axis=1)
X_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=1)
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
rand=RandomForestClassifier()

rand.fit(X,y)

y_pred=rand.predict(digit_test)

output = pd.DataFrame({'ImageId': range(1,len(digit_test)+1),

                       'Label': y_pred})

output.to_csv('submission_1.csv', index=False)