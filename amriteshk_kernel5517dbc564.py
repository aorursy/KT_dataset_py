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
import pandas as pd
raw_data=pd.read_csv('/kaggle/input/titanic/train.csv')

raw_data.head(5)
raw_data.isnull().sum()
#Data Wrangling
# cabin column has huge null value

# Therefore dropping cabin

raw_data.drop('Cabin',axis=1,inplace=True)
raw_data.dropna(inplace=True)
raw_data.isnull().sum()
#Creating Dummy Variables
sex=pd.get_dummies(raw_data['Sex'],drop_first=True)

sex
raw_data=pd.concat([raw_data,sex],axis=1)

raw_data.head(5)
raw_data.drop('Sex',axis=1,inplace=True)
embarked=pd.get_dummies(raw_data['Embarked'],drop_first=True)

embarked
raw_data=pd.concat([raw_data,embarked],axis=1)

raw_data.head(5)
raw_data.drop('Embarked',axis=1,inplace=True)
pclass=pd.get_dummies(raw_data['Pclass'],drop_first=True)

pclass
raw_data=pd.concat([raw_data,pclass],axis=1)

raw_data.head(5)
raw_data.drop('Pclass',axis=1,inplace=True)

raw_data.head(5)
#droping other unnecessary columns

raw_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

raw_data.head(5)
#Train Test Data
X_train=raw_data.drop('Survived',axis=1)

y_train=raw_data.Survived
raw_data_test=pd.read_csv('/kaggle/input/titanic/test.csv')

raw_data_test.head(5)
gen=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gen.head(5)
raw_data_test = raw_data_test.set_index('PassengerId').join(gen.set_index('PassengerId'))

raw_data_test.head()
raw_data_test.isnull().sum()
raw_data_test.dropna(inplace=True)
raw_data_test.isnull().sum()
sex=pd.get_dummies(raw_data_test['Sex'],drop_first=True)

raw_data_test=pd.concat([raw_data_test,sex],axis=1)

raw_data_test.drop('Sex',axis=1,inplace=True)
embarked=pd.get_dummies(raw_data_test['Embarked'],drop_first=True)

embarked
raw_data_test=pd.concat([raw_data_test,embarked],axis=1)

raw_data_test.drop('Embarked',axis=1,inplace=True)

pclass=pd.get_dummies(raw_data_test['Pclass'],drop_first=True)

raw_data_test=pd.concat([raw_data_test,pclass],axis=1)

raw_data_test.drop(['Name','Ticket'],axis=1,inplace=True)

raw_data_test.head(5)
raw_data_test.drop(['Pclass','Cabin'],axis=1,inplace=True)

raw_data_test.head(5)
X_test=raw_data_test.drop('Survived',axis=1)

y_test=raw_data_test.Survived
from sklearn.linear_model import LogisticRegression

LR_model=LogisticRegression()

LR_model.fit(X_train,y_train)

LR_prediction=LR_model.predict(X_test)

from sklearn.metrics import accuracy_score

LR_accuracy=accuracy_score(y_test,LR_prediction)

LR_accuracy