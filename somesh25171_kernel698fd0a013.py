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
#loading libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline


#loading train data set

train  = pd.read_csv('/kaggle/input/titanic/train.csv')

train.shape
#loading test dataset

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(test.shape)

test.head(3)

test.isnull().sum()
#checking for null values

train.isnull().sum()
#getting information on the dataset

train.info()
train.head(5)
#dealing with null values-----IMPUTATION

train = train.drop(columns = ['Cabin'])#train.drop(['Cabin'],axis = 1)

test = test.drop(columns=['Cabin'])

#Imputation replace age,fare with median value

test['Fare'].fillna(test['Fare'].median(),inplace = True)

test['Age'].fillna(test['Age'].median(),inplace = True)

train['Age'].fillna(train['Age'].median(),inplace = True)

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace = True)
train.head()
test.head(3)
plt.figure(figsize=(10,8))

plt.style.use('seaborn')

print(plt.style.available)

sns.countplot(x = 'Survived', data  = train)
fig = plt.figure(figsize = (15, 30))

ax1 = fig.add_subplot(4, 2, 1)

ax2 = fig.add_subplot(4, 2, 2)

ax3 = fig.add_subplot(4, 2, 3)

ax4 = fig.add_subplot(4, 2, 4)

ax5 = fig.add_subplot(4, 2, 5)

ax6 = fig.add_subplot(4, 2, 6)



ax1 = sns.countplot(x = "Survived",hue = "Pclass" , data  =  train, ax = ax1)

ax1.set_title("Pclass -survival distribution")



ax2 = sns.countplot(x = "Survived",hue = "Sex" , data  =  train, ax = ax2)

ax2.set_title("Sex -survival distribution")





ax3 = sns.countplot(x = "Survived",hue = "SibSp" , data  =  train, ax = ax3)

ax3.set_title("SiblingSpouse -survival distribution")





ax4 = sns.countplot(x = "Survived",hue = "Parch" , data  =  train, ax = ax4)

ax4.set_title("Parch -suvival distribution")



ax5 = sns.distplot(train[train['Survived'] == 0]['Age'].dropna(), color = 'red', label = '0', ax = ax5)

ax5 = sns.distplot(train[train.Survived == 1]['Age'].dropna(), color = 'blue', label = '1', ax = ax5)

ax5.set_title('Age')

ax5.legend(title = 'Age', loc = 'best')





ax6 = sns.distplot(train[train.Survived == 0]['Fare'], color = 'red', label = '0', ax = ax6)

ax6 = sns.distplot(train[train.Survived == 1]['Fare'], color = 'blue', label = '1', ax = ax6)

ax6.set_title('Fare')

ax6.legend(title = 'Fare', loc = 'upper right')
train_pair_plot =  train[['Age','Pclass','Fare']]

sns.pairplot(train_pair_plot)
Male = pd.get_dummies(train['Sex'])

Male = Male.drop(['female'], axis=1)

Male.head(5)

pclass = pd.get_dummies(train['Pclass'])

pclass = pclass.drop([3], axis =1)

pclass = pclass.rename(columns = {1:"pclass_1st",2:"pclass_2nd"})

pclass.head(5)
embarked = pd.get_dummies(train['Embarked'])

embarked = embarked.drop(['S'],axis =1)

embarked = embarked.rename(columns = {"C":"embarked_C","Q":"embarked_Q"})

embarked.head(3)
train.info()
train = train.drop(columns = ['Pclass','Sex','Embarked','Name','Ticket','PassengerId'],axis =1)

train = pd.concat([train,Male,pclass,embarked],axis =1)

train.head(5)
Male = pd.get_dummies(test['Sex'])

Male = Male.drop(['female'], axis=1)

Male.head(5)





pclass = pd.get_dummies(test['Pclass'])

pclass = pclass.drop([3], axis =1)

pclass = pclass.rename(columns = {1:"pclass_1st",2:"pclass_2nd"})



embarked = pd.get_dummies(test['Embarked'])

embarked = embarked.drop(['S'],axis =1)

embarked = embarked.rename(columns = {"C":"embarked_C","Q":"embarked_Q"})

embarked.head(3)





test = test.drop(columns = ['Pclass','Sex','Embarked','Name','Ticket'],axis =1)

test = pd.concat([test,Male,pclass,embarked],axis =1)

test.head(5)
train.shape,test.shape
X = train.iloc[:,1:10]

y = train.iloc[:,0]

X.shape,y.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)







X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.svm import SVC

from sklearn import metrics

svc = SVC(gamma = 'auto')

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

print('Accuracy Score with default hyperparameter:')

print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='linear',gamma='auto')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with linear kernel:')

print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='rbf',gamma = 'auto')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with RBF kernel:')

print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='poly',gamma = 'auto')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with polynomial kernel:')

print(metrics.accuracy_score(y_test,y_pred))
from sklearn.model_selection import cross_val_score

svc=SVC(kernel='rbf',gamma = 'auto')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') 

print(scores)

print('average of scores',scores.mean())
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cm
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

#adding second hidden layer

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

#adding output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compile ann



classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
print('ANN acc:', classifier.predict(X_test))
test.head(2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

z = test.iloc[:,1:10]

scaler.fit(z)

z = scaler.transform(z)

z.shape
result = pd.DataFrame()

result['PassengerId'] = test['PassengerId']

result_1= classifier.predict(z)

result_2 = (result_1 > 0.5)

result_2
result['Survived'] = pd.DataFrame(result_2.astype('int'))

result.to_csv('submission.csv', index=False,header = 1)

result.head(10)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = result



# create a link to download the dataframe

create_download_link(df)