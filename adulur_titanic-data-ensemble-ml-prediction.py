# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

#What our submission file should look like, on the test dataframe
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train_processed = df_train.copy()


#Dropping ticket number and passengerId
df_train_processed.drop(columns = ['PassengerId', 'Name', 'Ticket'], inplace = True)

#One-hot encoding the Cabin "Deck" Letter
df_train_processed['Cabin'] = df_train_processed['Cabin'].str[0] + " Deck"
df_train_processed = pd.concat([df_train_processed, pd.get_dummies(df_train_processed.Cabin)], axis=1)
df_train_processed.drop(columns = 'Cabin', inplace = True)

#One-hot encoding the Port of Origin
df_train_processed = pd.concat([df_train_processed, pd.get_dummies(df_train_processed.Embarked)], axis=1)
df_train_processed.drop(columns = 'Embarked', inplace = True)

#Converting Sex to numbers
df_train_processed.Sex.replace({'male': 0, 'female': 1}, inplace = True)
df_train_processed.head()

#Feature Engineering - I predict that someone alone is more likely to survive, vs. people with families are not (since they go back for them and stay together)
df_train_processed['Alone'] = 0
df_train_processed.loc[(df_train_processed['SibSp'] == 0) & (df_train_processed['Parch'] == 0), 'Alone'] = 1


#Normalizing the Age and Fare columns
df_train_processed['Age'] = df_train_processed['Age'] / (df_train_processed['Age'].max())
df_train_processed['Fare'] = df_train_processed['Fare'] / (df_train_processed['Fare'].max())


df_train_processed.head()
df_test_processed = df_test.copy()


#Dropping ticket number and passengerId
df_test_processed.drop(columns = ['PassengerId', 'Name', 'Ticket'], inplace = True)

#One-hot encoding the Cabin "Deck" Letter
df_test_processed['Cabin'] = df_test_processed['Cabin'].str[0] + " Deck"
df_test_processed = pd.concat([df_test_processed, pd.get_dummies(df_test_processed.Cabin)], axis=1)
df_test_processed.drop(columns = 'Cabin', inplace = True)
df_test_processed['T Deck'] = 0

#One-hot encoding the Port of Origin
df_test_processed = pd.concat([df_test_processed, pd.get_dummies(df_test_processed.Embarked)], axis=1)
df_test_processed.drop(columns = 'Embarked', inplace = True)

#Converting Sex to numbers
df_test_processed.Sex.replace({'male': 0, 'female': 1}, inplace = True)
df_test_processed.head()

#Feature Engineering - I predict that someone alone is more likely to survive, vs. people with families are not (since they go back for them and stay together)
df_test_processed['Alone'] = 0
df_test_processed.loc[(df_test_processed['SibSp'] == 0) & (df_test_processed['Parch'] == 0), 'Alone'] = 1

#Normalizing the Age and Fare columns
df_test_processed['Age'] = df_test_processed['Age'] / (df_test_processed['Age'].max())
df_test_processed['Fare'] = df_test_processed['Fare'] / (df_test_processed['Fare'].max())

df_test_processed.head()
#Training data
x_train = df_train_processed.copy()
x_train = x_train.fillna(x_train.mean())

y_train = df_train_processed[['Survived']]
y_train = np.ravel(y_train)
x_train.drop(columns = 'Survived', inplace = True)

#Testing data
x_test = df_test_processed.copy()
x_test = x_test.fillna(x_test.mean())


from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import statistics

model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, statistics.mode([pred1[i], pred2[i], pred3[i]]))
df_test['Survived'] = final_pred
df_test[['PassengerId', 'Survived']].to_csv('Submission2.csv')
#Let's use some sort of ensemble learning here 
#https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/