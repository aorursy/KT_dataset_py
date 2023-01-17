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
train_dataset=pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

test_dataset=pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
train_dataset.head()
test_dataset.head()
train_dataset.info()
test_dataset.info()
train_dataset.isnull().sum()
test_dataset.isnull().sum()
X=train_dataset.drop(labels=['Id','Cover_Type'],axis=1)

y=train_dataset['Cover_Type']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=40)
X_train
# Training the Random Forest Classification model on the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test, y_pred)

print(acc)

predict=classifier.predict(test_dataset.drop(labels=['Id'],axis=1))

Submission=pd.DataFrame(data=predict,columns=['Cover_Type'])

Submission.head()
Submission['Id']=test_dataset['Id']

Submission.set_index('Id',inplace=True)
Submission.head()
Submission.to_csv('Submission.csv')