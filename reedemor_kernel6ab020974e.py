# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.isnull()


sns.heatmap(train.isnull() , yticklabels = False , cbar = False , cmap = 'viridis' )
sns.set_style('whitegrid')
sns.countplot(x= 'Survived' ,hue = 'Sex' , data =train )

train['Age'].hist(bins = 30, color= 'darkred' , alpha= 0.3)
sns.countplot(x = 'SibSp' , data = train)
train['Fare'].hist(color = 'orange' , bins = 40 , figsize = (8,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind = 'hist' , bins = 30 , color = 'green')
plt.figure(figsize = (12,7))
sns.boxplot(x = 'Pclass' , y = 'Age' , data = train , palette = 'summer' )
for i in range(train.shape[0]):
    #print(i)
    if pd.isnull(train['Age'][i]):
         if train['Pclass'][i] == 1:
                train['Age'][i] = 37
         
         if train['Pclass'][i] == 2:
                train['Age'][i] = 29
        
         if train['Pclass'][i] == 3:
                train['Age'][i] = 24
                
for i in range(test.shape[0]):
    #print(i)
    if pd.isnull(test['Age'][i]):
         if test['Pclass'][i] == 1:
                test['Age'][i] = 37
         
         if test['Pclass'][i] == 2:
                test['Age'][i] = 29
        
         if test['Pclass'][i] == 3:
                test['Age'][i] = 24
       

sns.heatmap(test.isnull())
for i in range(418):
    if pd.isnull(test['Fare'][i]):
        test['Fare'][i] = 35.62



test.info()
train.dropna(inplace = True)

sns.heatmap(test.isnull())
train.info()
test.info()
pd.get_dummies(train['Embarked'] , drop_first =True).head()

sex = pd.get_dummies(train['Sex'] , drop_first= True)
sex_t = pd.get_dummies(test['Sex'] , drop_first = True)
embark = pd.get_dummies(train['Embarked'] , drop_first = True)
embark_t = pd.get_dummies(test['Embarked'] , drop_first = True)
train.drop(['Sex', 'Name' , 'Ticket' , 'Embarked'] , axis= 1 , inplace =True)
test.drop(['Sex', 'Name' , 'Ticket' , 'Embarked'] , axis= 1 , inplace =True)
train.drop('PassengerId' , axis=1 , inplace = True)
test.drop('PassengerId' , axis=1 , inplace = True)
test.head()
train = pd.concat([train , sex ,embark] ,axis =1)
test = pd.concat([test , sex_t ,embark_t] ,axis =1)
test.head()
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(train.drop('Survived' , axis = 1) , train['Survived'] , test_size =0.3 , random_state = 41 )
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train , y_train)
predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test , predictions)
accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , predictions)
accuracy
predictions
from sklearn.metrics import classification_report
print(classification_report(y_test , predictions))
main_predict = model.predict(test)
main_predict
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': main_predict})
output.to_csv('mysubmission.csv' , index = False)
print('Your Submission is successfully saved')
