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
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import  train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import auc,roc_auc_score,accuracy_score
# importing the dataset
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# plotting the missing values
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['total','percennt'])
# missing_data
sns.barplot(x=missing_data.index, y=missing_data['percennt'])
# notice the spread of the age
sns.set_style("whitegrid")
sns.distplot(train['Age'],kde = False)
sns.boxplot(train['Age'])

#  following the 68–95–99.7 rule for finding the best fit for missing values
age = train['Age'].mean() +3*train['Age'].std()
train['Age'].fillna(age,inplace = True)
sns.boxplot(train['Age'])

# dropping the columns,rows
train['Embarked'].dropna(axis = 'rows',inplace = True)
train.drop(columns = ['PassengerId','Cabin','Name','Ticket'],inplace=True)
# splitting into x and y
x = train.drop(columns = 'Survived')
y = train['Survived']

# converting the categorical values
sex = pd.get_dummies(train['Sex'],prefix="sex",drop_first=True)
embarked = pd.get_dummies(train['Embarked'],prefix = 'embarked',drop_first=True)
x.drop(columns =['Sex','Embarked'],inplace = True,axis = 'columns' )

# merging the converted catevalues with the dataset
x = pd.concat([x,sex,embarked],axis='columns')
# scaling the values
x['Fare'] = (x['Fare'] - min(x['Fare']))/(max(x['Fare']) - min(x['Fare']))
x['Age'] = (x['Age'] - min(x['Age']))/(max(x['Age']) - min(x['Age']))
# splitting the dataset in train and test
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 42)

# calling the Randomforest classifier
clf = RandomForestClassifier()
param = { 'criterion' : ['gini','entropy'], 'min_samples_split' : [2,3,4]}
grid = GridSearchCV(clf,param_grid=param,cv = StratifiedKFold())
grid.fit(xtrain,ytrain)

# plotting the AUC curve
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = grid.predict_proba(xtest)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(ytest, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# predicting the accuracy
ypred = grid.predict(xtest)
accuracy_score(ytest,ypred)

# cleaning the test dataset for classification similar to test dataset
test.drop(columns = ['Name','PassengerId','Cabin'],inplace = True)
test.drop(columns = ['Ticket'],inplace = True)
age = test['Age'].mean() + 3 * test['Age'].std()
test['Age'].fillna(age,inplace = True)
test['Fare'].fillna(7.75,inplace = True)
sex = pd.get_dummies(test['Sex'],prefix = 'sex',drop_first=True)
Embarked = pd.get_dummies(test['Embarked'],prefix = 'sex',drop_first=True)
test.drop(columns=['Sex','Embarked'],inplace=True)
test=pd.concat([test,sex,Embarked],axis='columns')

test.describe()

# scaling the data
test['Age']  = (test['Age'] - min(test['Age']))/(max(test['Age']) - min(test['Age']))
test['Fare'] = (test['Fare'] - min(test['Fare']))/(max(test['Fare']) - min(test['Fare']))

# predicting the values
ypred = grid.predict(test)

# saving the csv file for subbmission
test_df = pd.read_csv('../input/titanic/test.csv')
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = ypred
submission_df.to_csv('submission.csv', header=True, index=False)

