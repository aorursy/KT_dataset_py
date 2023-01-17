import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic/train.csv')
#data.drop(columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"], inplace = True)

data.Sex[data.Sex == 'male'] = 0

data.Sex[data.Sex == 'female'] = 1
vec = data.groupby("Embarked").count()['PassengerId']
data.Embarked[data.Embarked == 'C'] = 1/vec[0]

data.Embarked[data.Embarked == 'Q'] = 1/vec[1]

data.Embarked[data.Embarked == 'S'] = 1/vec[2]
data.head()
data.drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis= 1, inplace = True) 
data = data.fillna(0)
train_o = data.Survived

train_i = data.drop(['Survived'], axis= 1) 
train_i.head()
train_o.head()
model = LogisticRegression()
model.fit(train_i,train_o)
data1 = pd.read_csv('/kaggle/input/titanic/test.csv')
data1.Sex[data1.Sex == 'male'] = 0

data1.Sex[data1.Sex == 'female'] = 1
vec = data1.groupby("Embarked").count()['PassengerId']

data1.Embarked[data1.Embarked == 'C'] = 1/vec[0]

data1.Embarked[data1.Embarked == 'Q'] = 1/vec[1]

data1.Embarked[data1.Embarked == 'S'] = 1/vec[2]
data1.head()
data1 = data1.fillna(0)

p_id=pd.DataFrame(data1['PassengerId'])

data1.drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis= 1, inplace = True) 
test_i = data1
p = model.predict(test_i)
submission = pd.DataFrame({'PassengerId':p_id['PassengerId'],'Survived':p})
submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)