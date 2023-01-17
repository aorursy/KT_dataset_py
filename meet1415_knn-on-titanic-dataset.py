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
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix, f1_score
train=pd.read_csv(r'/kaggle/input/titanic/train.csv')

test=pd.read_csv(r'/kaggle/input/titanic/test.csv')

gender_submission=pd.read_csv(r'/kaggle/input/titanic/gender_submission.csv')
test.insert(1, "Survived", gender_submission['Survived'], True) 
train.head()
frames = [train,test]

dataset = pd.concat(frames)
dataset.head()
dataset=dataset[['Survived','Pclass','Sex','SibSp','Parch','Embarked']]
dataset.dropna()

dataset=pd.get_dummies(dataset)
y=dataset['Survived']

dataset.drop(['Survived'], axis=1)

y=y.to_frame()

print(type(dataset))

#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=2)

neigh.fit(x_train,y_train)
y_pred=neigh.predict(x_test)
print(y_pred)
#from sklearn.metrics import accuracy_score,confusion_matrix, f1_score

print("accuracy score:",accuracy_score(y_test, y_pred))

print("confision matrix:\n",confusion_matrix(y_test, y_pred))

print("f1 score:",f1_score(y_test, y_pred, average='macro'))