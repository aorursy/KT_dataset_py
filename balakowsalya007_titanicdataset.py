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
data = pd.read_csv('/kaggle/input/gender_submission.csv')
data
df = pd.read_csv('/kaggle/input/train.csv',header = 0)
df.info()
cols = ['Name','Ticket','Cabin']

df = df.drop(cols,axis=1)
df
df.dropna()
dummies = []

cols = ['Pclass','Sex','Embarked']

for col in cols:

    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies,axis=1)
df = pd.concat((df,titanic_dummies),axis=1)
df
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
df
df['Age'] = df['Age'].interpolate()
df
df.info()
df
X = df.values

y = df['Survived'].values
X = np.delete(X,1,axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,Y_train)

clf.score(X_test,Y_test)
clf.feature_importances_
from sklearn import ensemble

clf = ensemble.GradientBoostingClassifier()

clf.fit (X_train, Y_train)

clf.score (X_test, Y_test)

clf = ensemble.GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train,Y_train)

clf.score(X_test,Y_test)
y_test = clf.predict(X_test)
output = np.column_stack((X_test[:,0],y_test))
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df_results.to_csv('titanic_results.csv',index=False)