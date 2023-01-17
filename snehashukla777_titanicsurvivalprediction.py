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

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score



sns.set_style('whitegrid')
data = pd.read_csv('/kaggle/input/titanic/train.csv')

data
data.shape # get shape
sns.countplot(x = 'Survived',data = data) # countplot to check how many survived (1) and not survived (0).
sns.countplot(x = 'Survived',hue = 'Sex',data = data) # check how many male/female are survived (1) and not survived (0). 

sns.countplot(x = 'Survived',hue = 'Pclass', data = data) # check pasangers are from which class.
data['Age'].hist(bins = 10) # histogram of Age (agewise frequency of pasanger in titanic)
sns.countplot(x = 'SibSp',data = data) # get countplot of 'SibSp'
data.info() # get data info
data.isnull() # check null values in data (False = not null, True = null)
data.isnull().sum()  # get sum of null values in each column.
sns.heatmap(data.isnull()) # heatmap where 'Age' and 'Cabin' has more null values.
data['Age'] = data['Age'].fillna(data['Age'].mean()) # null values in 'Age' is replaced by mean.

data = data.drop(['Cabin'],axis = 1) # drop 'Cabin' which is having more null values
data.shape # get shape (1 column dropped)
data.dropna(inplace = True) # remaining null values removed
data.isnull().sum() # check for the null values
sns.heatmap(data.isnull()) # heatmap for null values 
data.drop(['PassengerId','Name','Ticket'],axis = 1,inplace = True)  # drop unwanted column from data
pd.options.display.float_format = '{:,.2f}'.format

data.corr() # get correlation
sns.heatmap(data.corr(),annot = True,fmt = '.2f') # visualize correlation
# get dummies for 'Sex', 'Embarked','Pclass'

sex = pd.get_dummies(data['Sex'],drop_first = True) 

embarked = pd.get_dummies(data['Embarked'],drop_first = True)

pclass = pd.get_dummies(data['Pclass'],drop_first = True)
data = pd.concat([data,sex,pclass,embarked],axis = 1) # add it into data

data.drop(['Pclass','Sex','Embarked'],axis = 1, inplace = True) # remove previous one
x = data.drop(['Survived'],axis = 1) # get independent variable

y = data['Survived'] # get dependent (target) variable
# perform train-test-split with test_size of 0.2 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
# data scaling

sc = StandardScaler()



x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
model = LogisticRegression(max_iter = 150,n_jobs = 1)  # model building

model.fit(x_train,y_train) # model training

y_pred = model.predict(x_test) # get prediction
# performance evaluation 



print(confusion_matrix(y_test,y_pred))

print("Accuracy:",round(accuracy_score(y_test,y_pred)*100,2),'%')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv') # load test data

test_data
test_data.shape # get shape
test_data.isnull().sum() # check null values
# handling null values



test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())  # replace null values

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean()) # replace null values

test_data.drop(['Cabin'],axis = 1, inplace = True) # drop 'Cabin' having more null value
test_data.isnull().sum() # check null values
# get dummies 



tsex = pd.get_dummies(test_data['Sex'],drop_first = True)

tembarked = pd.get_dummies(test_data['Embarked'],drop_first = True)

tpclass = pd.get_dummies(test_data['Pclass'],drop_first = True)
t_data = pd.concat([test_data,tsex,tpclass,tembarked],axis = 1) # add it into data

t_data = t_data.drop(['Pclass','PassengerId','Name','Sex','Ticket','Embarked'],axis = 1) # drop unwanted coulumn

t_data = sc.fit_transform(t_data) # data scaling
result = model.predict(t_data) # make prediction on test data

result
result.shape # check result shape
# add it to csv file



id =  test_data['PassengerId']

d = {'PassengerId':id,'Survived':result}

df = pd.DataFrame(d)

df.to_csv('TitanicSubmission.csv',index = False)