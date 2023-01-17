# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.

train_csv = pd.read_csv('../input/train.csv')

test_csv = pd.read_csv('../input/test.csv')



train_csv.head(10)



#dropping not required cols from both training and test data

cols = ['Name','Ticket','Cabin']

train_csv = train_csv.drop(cols,axis=1)

test_csv = test_csv.drop(cols,axis=1)





dummies = []

cols = ['Pclass','Sex','Embarked']

#

for col in cols:

    dummies.append(pd.get_dummies(train_csv[col]))

    

test_dummies=[]    

test_dummies.append(pd.get_dummies(test_csv['Pclass']))

test_dummies.append(pd.get_dummies(test_csv['Sex']))

test_dummies.append(pd.get_dummies(test_csv['Embarked']))







titanic_dummies = pd.concat(dummies,axis=1)

train_csv=pd.concat((train_csv,titanic_dummies),axis=1)



test_titanic_dummies = pd.concat(test_dummies,axis=1)

test_csv = pd.concat((test_csv,test_titanic_dummies),axis=1)

                        





#drop cols

train_csv = train_csv.drop(['Pclass','Sex','Embarked'],axis=1)

test_csv = test_csv.drop(['Pclass','Sex','Embarked'],axis=1)


                        

train_csv['Age'] = train_csv['Age'].interpolate()

test_csv['Age'] = test_csv['Age'].interpolate()

test_csv['Fare'] = test_csv['Fare'].interpolate()   







train_csv['Age'] = (train_csv['Age'] - train_csv['Age'].min())/(train_csv['Age'].max()-train_csv['Age'].min())

train_csv['Fare'] = (train_csv['Fare'] - train_csv['Fare'].min())/(train_csv['Fare'].max()-train_csv['Fare'].min())



test_csv['Age'] = (test_csv['Age'] - test_csv['Age'].min())/(test_csv['Age'].max()-test_csv['Age'].min())

test_csv['Fare'] = (test_csv['Fare'] - test_csv['Fare'].min())/(test_csv['Fare'].max()-test_csv['Fare'].min())



train_csv.info()

test_csv.info()



train_csv.describe()

test_csv.describe()



train_csv.corr('pearson')



X = train_csv.values

y = train_csv['Survived'].values



X_result = test_csv.values



                        

X = np.delete(X,1,axis=1)

#X_result = np.delete(X_result,1,axis=1)



train_csv.corr()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)



#Decision tree classifier

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=5)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)



#Random forest classifier

from sklearn import ensemble

# clf = ensemble.RandomForestClassifier(n_estimators=100)

# clf.fit (X_train, y_train)

# clf.score (X_test, y_test)



# #GradientBoosting classifier

# clf = ensemble.GradientBoostingClassifier()

# clf.fit (X_train, y_train)

# clf.score (X_test, y_test)



#Gradient Boosting classifier 

clf = ensemble.GradientBoostingClassifier(n_estimators=100)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)



# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression()

# clf.fit(X_train,y_train)



#clf.score(X_result,y_result)

y_results = clf.predict(X_result)

output = np.column_stack((X_result[:,0],y_results))

df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])

df_results.head()

df_results.to_csv('submission.csv',index=False)





#df_results.to_csv('titanic_results.csv',index=False)



#X.describe()

#train_csv.describe()

#titanic_dummies.describe()



#train_csv['Age'] = train_csv['Age'].interpolate()



#info of the training and testing data

#train_csv.head(10)

#test_csv.describe()


