# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading in the training dataset and verifying any columns with NaN values

X = pd.read_csv('../input/train.csv')

X.isnull().any()
#Calculating mean value for age and fitting it for null values in age

np.mean(X['Age'])



#dropping 'Cabin', Name, ticket and fare from the model completely, 

# dropping NA's from 'embarked'

X = X[['PassengerId','Survived',

 'Pclass',

 'Sex',

 'Age',

 'SibSp',

 'Parch',

 'Embarked']]



X['Embarked'] = X['Embarked'].fillna(value = 'S')



X['Age'].fillna(value = 29, inplace = True)
#setting y/response column to "Survived"

y = X['Survived']
#Using selected columns for model and verifying shape is good for scikit

X = X[['PassengerId',

 'Pclass',

 'Sex',

 'Age',

 'SibSp',

 'Parch',

 'Embarked']]



print(X.shape)

print(y.shape)
#Verifying data types for the feature columns and modifying them if neccessary

X.dtypes
X['Sex'] = X['Sex'].astype('category')

X['Embarked'] = X['Embarked'].astype('category')
#Since scikit doesn;t like strings, getting panda dummy columns for category values

X = pd.get_dummies(X, columns=['Sex','Embarked'])
#Although we have a test set, Doing a train/test split



from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
#let's try logistic regression model from sklearn



from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()
#fit the training set to the instance

logreg.fit(X_train, y_train)
#predict

y_pred = logreg.predict(X_test)
#Let's evaluate the model using RMSE

from sklearn import metrics



print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(metrics.accuracy_score(y_test, y_pred))
#Now let's massage the data in the test dataset as per our model

X_new = pd.read_csv('../input/test.csv')   

X_new = X_new[['PassengerId',

 'Pclass',

 'Sex',

 'Age',

 'SibSp',

 'Parch',

 'Embarked']]

X_new.isnull().any()
X_newMeanAge = np.mean(X_new['Age'])

X_new['Age'].fillna(value = X_newMeanAge,inplace = True)
X_new.dtypes
X_new['Sex'] = X_new['Sex'].astype('category')

X_new['Embarked'] = X_new['Embarked'].astype('category')
X_new = pd.get_dummies(X_new, columns=['Sex','Embarked'])
y_pred = logreg.predict(X_new)
y_pred = pd.Series(y_pred) #coverting from numpy array to series 
X_new['Survived'] = y_pred

X_new[['PassengerId' , 'Survived']]