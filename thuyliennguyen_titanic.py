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
import matplotlib.pyplot as plt

titanic_df=pd.read_csv("/kaggle/input/titanic/train.csv")
titanic_df.head()
titanic_df.info()
titanic_df.describe()
titanic_df.columns
print(pd.pivot_table(titanic_df, index='Survived', columns='Pclass', values='Ticket',aggfunc='count'))
print(pd.pivot_table(titanic_df, index='Survived', columns='Sex', values='Ticket',aggfunc='count'))
print(pd.pivot_table(titanic_df, index='Survived', columns='Age', values='Ticket',aggfunc='count'))
print(pd.pivot_table(titanic_df, index='Survived', columns='Fare', values='Ticket',aggfunc='count'))
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.head()

X= titanic_df[['Pclass','Sex', 'Age','Fare','SibSp','Parch']]
X.head()

# check for null values
round(100*(titanic_df.isnull().sum()/titanic_df.shape[0]),2)
## impute the missing values with mean of the Age 
age_mean = titanic_df['Age'].mean()
titanic_df['Age'].fillna(age_mean,inplace=True)

test_df['Age'].fillna(age_mean,inplace=True)

#test_df['Fare'].fillna(fare_mean,inplace=True) ## impute missing fare values 
# check for null values
round(100*(titanic_df.isnull().sum()/titanic_df.shape[0]),2)


round(100*(test_df.isnull().sum()/test_df.shape[0]),2)
## impute the missing values with mean of the Age 
fare_mean = titanic_df['Fare'].mean()
test_df['Fare'].fillna(fare_mean,inplace=True) ## impute missing fare values 
for i in X[['Pclass','Age','Fare']].columns:
    plt.hist(X[['Pclass','Age','Fare']][i])
    plt.title(i)
    plt.show()


testX=test_df[['Pclass','Sex', 'Age','Fare','SibSp','Parch']]
testX.head()
y=titanic_df['Survived']
y.head()
pd.pivot_table(titanic_df, index='Survived', values= ['Pclass','Sex', 'Age','Fare'])
X=X.values
testX=testX.values
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['female','male'])
X[:,1] = le_sex.transform(X[:,1]) 
X = SimpleImputer().fit_transform(X)
testX[:,1] = le_sex.transform(testX[:,1]) 
testX = SimpleImputer().fit_transform(testX)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=1, solver='liblinear').fit(X_train,y_train)
LR

yhat = LR.predict(X_test)
yhat

y_test

yhat_prob = LR.predict_proba(X_test)
yhat_prob
#from sklearn import svm
#clf = svm.SVC(kernel='rbf')
#clf.fit(X_train, y_train) 
#ySVChat = clf.predict(X_test)
from sklearn import metrics
metrics.confusion_matrix(y_test, yhat)


# accuracy
print("accuracy", metrics.accuracy_score(y_test, yhat))

# precision
print("precision", metrics.precision_score(y_test, yhat))

# recall/sensitivity
print("recall", metrics.recall_score(y_test, yhat))

# accuracy
#print("accuracy", metrics.accuracy_score(y_test, ySVChat))

# precision
#print("precision", metrics.precision_score(y_test, ySVChat))

# recall/sensitivity
#print("recall", metrics.recall_score(y_test, ySVChat))
test_df['Survived'] = LR.predict(testX)
test_df.head()
titanic_test_final = test_df[['PassengerId','Survived']]
titanic_test_final.head()
titanic_test_final.to_csv("prediction_titanic_svm.csv",index=False)