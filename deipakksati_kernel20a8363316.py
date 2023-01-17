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
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
def female (row):

    if row['Sex'] == 'male':

        a1 = -1

    else:

        a1 = 1        

    return a1
def C (row):

    if row['Embarked'] == 'C':

        a1 = 1

    else:

        a1 = 0        

    return a1

def Q (row):

    if row['Embarked'] == 'Q':

        a1 = 1

    else:

        a1 = 0        

    return a1



def S (row):

    if row['Embarked'] == 'S':

        a1 = 1

    else:

        a1 = 0        

    return a1
def Cab (row):

    if row['Cabin'] == 0:

        a = -1

    else:

        if row['female'] == 1:

            a = 1

        else:

            a = -1

    return a
from sklearn.utils import shuffle

col = ['Survived','Pclass','Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']

out = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female','C','S','Q','Cab']

train['female'] = train.apply (lambda row: female (row),axis=1)

train['C'] = train.apply (lambda row: C (row),axis=1)

train['S'] = train.apply (lambda row: S (row),axis=1)

train['Q'] = train.apply (lambda row: Q (row),axis=1)

mean_age = train['Age'].median()

mean_Fare = train['Fare'].median()

train['Age'] = train['Age'].fillna(mean_age)

train['Fare'] = train['Fare'].fillna(mean_Fare)

train['Cabin'] = train['Cabin'].fillna(0)

train['Cab'] = train.apply (lambda row: Cab (row),axis=1)

train = train[train['Age'] != mean_age] 

train = train[train['Fare'] != mean_Fare] 



test['female'] = test.apply (lambda row: female (row),axis=1)

test['C'] = test.apply (lambda row: C (row),axis=1)

test['S'] = test.apply (lambda row: S (row),axis=1)

test['Q'] = test.apply (lambda row: Q (row),axis=1)

test['Cab'] = test.apply (lambda row: Cab (row),axis=1)

test['Age'] = test['Age'].fillna(mean_age)

test['Fare'] = test['Fare'].fillna(mean_Fare)

test['Cabin'] = test['Cabin'].fillna(0)



test['Q'] = test.apply (lambda row: Q (row),axis=1)

df_train , df_val= train_test_split(train, test_size=0.2, random_state=111)



y_train_final = shuffle(train)

y_train_final_o = y_train_final['Survived']

X_train_final_i = y_train_final[out]



y_train = df_train['Survived']

y_val = df_val['Survived']

X_train = df_train[out]

X_val = df_val[out]

X_test = test[out]



print(X_train_final_i.shape,y_train_final_o.shape, y_train.shape)



from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn import svm

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error




#random foreset 

clf = RandomForestClassifier(n_estimators=50)

    

clf.fit(X_train, y_train)

clf_prediction = clf.predict(X_val)







print('Random Forest Classifier details')

print('Accuracy score:', accuracy_score(clf_prediction, y_val))

print(classification_report(clf_prediction, y_val))

print(confusion_matrix(clf_prediction, y_val))





#final 
#random foreset 

clf_f = RandomForestClassifier(n_estimators=50)

clf_f.fit(X_train_final_i, y_train_final_o)

output = clf_f.predict(X_test)

pre = clf_f.predict(X_val)

print('Accuracy score:',accuracy_score(pre, y_val))

#SVC_model_f = svm.SVC(kernel = 'poly')

#Std = StandardScaler()

#std_fit = Std.fit(X_train_final_i)

#X_train_f = Std.transform(X_train_final_i)





#SVC_model_f.fit(X_train_f, y_train_final_o)



#output = SVC_model_f.predict(X_test)
print(output.shape , X_test.shape)

print(output)
# Create the pandas DataFrame 

df_final = pd.DataFrame(columns = ['PassengerId', 'Survived']) 

df_final['PassengerId'] = test['PassengerId']

df_final['Survived'] = output
df_final.to_csv('final_3.csv')