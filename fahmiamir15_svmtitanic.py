# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
train.head()

#print("\n\nSummary statistics of training data")
#train.describe()

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sb

train.columns
train = train.drop(train[['Name','Ticket']],axis=1)
sb.pairplot(train.drop('PassengerId',axis=1).dropna(),hue='Survived')
pasID = train['PassengerId'].values
train = train.drop('PassengerId',axis=1)
train.columns
plt.figure(figsize=(15,30))
for col_id , col in enumerate(train.columns):
	if col == 'Survived':
		continue
	plt.subplot(4,2,col_id)
	sb.violinplot(x=col,y='Survived',data=train.dropna())
train['Embarked'][train['Embarked'].isnull()] = 'S'
train['Embarked'].unique()
train['Embarked'][train['Embarked'].isnull()]
train['Embarked'][train['Embarked']=='S'] = 0
train['Embarked'][train['Embarked']=='C'] = 1
train['Embarked'][train['Embarked']=='Q'] = 2
test['Embarked'][test['Embarked']=='S'] = 0
test['Embarked'][test['Embarked']=='C'] = 1
test['Embarked'][test['Embarked']=='Q'] = 2
train['Age'][train['Age'].isnull()]
train['Age'].hist()
train['Age'].mean()
train['Age'].median()
train['Age'][train['Age'].isnull()] = train['Age'].mean()
train['Age'][train['Age'].isnull()]
train.columns
train['Sex'][train['Sex']=='male'] = 1
train['Sex'][train['Sex']=='female'] = 0
test['Sex'][test['Sex']=='male'] = 1
test['Sex'][test['Sex']=='female'] = 0
train_f = train[['Pclass','Sex','SibSp','Parch','Embarked']].values
train_c = train['Survived'].values
test.head()
test_f = test[['Pclass','Sex','SibSp','Parch','Embarked']].values
from sklearn.svm import SVC

svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svc.fit(train_f,train_c)
svc.score(train_f,train_c)
pred =svc.predict(test_f)
len(pred)
id = test['PassengerId'].values
ans = list(zip(id,pred))
df = pd.DataFrame(data = ans, columns=['PassengerId','Survived'])
df.to_csv('prediction.csv', sep=';', encoding='utf-8')
