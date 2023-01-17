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
import pandas as pd 

import numpy as np 



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier




data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
data.head()
test_data.head()
data.describe().sum()
data.describe(include=[np.object])
data.isnull().sum()
# removing the column cabin  



data = data.drop("Cabin" , axis=1)
data.isnull().sum()
# filling the missing age values  

data['Age'] = data['Age'].fillna(0)
# filling the missing embarked sections 



temp=data.Embarked.describe()[2]

data['Embarked'] = data['Embarked'].fillna(temp)
data.isnull().sum()
data[['Pclass' , 'Survived']].groupby(by='Pclass').mean()
data.Pclass.unique()
data[['Sex' , 'Survived']].groupby(by='Sex').mean()
data[['Embarked' , 'Survived']].groupby(by='Embarked').mean()
data.head()


data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data['Title'],data['Sex'])

data['Title']=data['Title'].replace(['Miss','Mrs'],'Ladies')
data.head()
data['Age_range'] = pd.cut(data.Age,5 )
data[['Age_range' , 'Survived']].groupby(by = 'Age_range').sum()
# classifying the data set on age ranges 



data.loc[data['Age']<16.0,'Age']=0

data.loc[(data['Age']>=16.0) & (data['Age']<32.0),'Age']=1

data.loc[(data['Age']>=32.0) & (data['Age']<48.0),'Age']=2

data.loc[(data['Age']>=48.0) & (data['Age']<64.0),'Age']=3

data.loc[(data['Age']>=64.0) & (data['Age']<80.0),'Age']=4

data.loc[data['Age']>=80.0,'Age']=5
data.Age = data.Age.astype(int)
data.head()
# creating a new column relation 



data['Relation']= data.SibSp + data.Parch
data[['Relation' , 'Survived']].groupby(by = 'Relation').mean()
# people having no relationships  



data['No_one'] = 0 

data.loc[data['Relation']>=1 , 'No_one' ]=1
data.head()
data[['No_one','Survived']].groupby(by='No_one').mean()
data = data.drop(['Title' , 'Name' , 'Age_range' , 'Relation'] , axis =1)
data.head()
data['Fare_cut'] = pd.cut(data.Fare , 5)



data[['Fare_cut' , 'Survived']].groupby(by = 'Fare_cut').mean()
data.loc[data['Fare']<102.0,'Fare']=0

data.loc[(data['Fare']>=102.0) & (data['Fare']<204.0),'Fare']=1

data.loc[(data['Fare']>=204.0) & (data['Fare']<307.0),'Fare']=2

data.loc[(data['Fare']>=307.0) & (data['Fare']<409.0),'Fare']=3

data.loc[(data['Fare']>=409.0) & (data['Fare']<512.0),'Fare']=4

data.loc[data['Fare']>=512.0,'Fare']=5

data.Fare=data.Fare.astype('int')
data.head()
data['Embarked'].unique()
data.Embarked=data.Embarked.replace({'S':0,'C':1,'Q':2})
data.Sex=data.Sex.replace({'male':0,'female':1}).astype('int')
data.head()
# replacing the values of embarked with 0 1 2



data.Embarked = data.Embarked.replace({'S':0 , 'C':1 , 'Q':2})
data.head()
data = data.drop('Ticket' , axis =1 )
data.head()
test_data.head()
test_data['Age']=test_data['Age'].fillna(0)

test_data['Embarked']=test_data['Embarked'].fillna(temp)
test_data['age_range']=pd.cut(test_data.Age,5)

test_data.loc[test_data['Age']<16.0,'Age']=0

test_data.loc[(test_data['Age']>=16.0) & (test_data['Age']<32.0),'Age']=1

test_data.loc[(test_data['Age']>=32.0) & (test_data['Age']<48.0),'Age']=2

test_data.loc[(test_data['Age']>=48.0) & (test_data['Age']<64.0),'Age']=3

test_data.loc[(test_data['Age']>=64.0) & (test_data['Age']<80.0),'Age']=4

test_data.loc[test_data['Age']>=80.0,'Age']=5
test_data['Relation']=test_data.SibSp+test_data.Parch

test_data['No_one']=0

test_data.loc[test_data['Relation']>=1,'No_one']=1
test_data=test_data.drop(['age_range','Relation' , 'Name'],axis=1)

test_data=test_data.drop(['Ticket'],axis=1)
test_data.head()
test_data['fare_cut']=pd.cut(test_data.Fare,5)

test_data.loc[test_data['Fare']<102.0,'Fare']=0

test_data.loc[(test_data['Fare']>=102.0) & (test_data['Fare']<204.0),'Fare']=1

test_data.loc[(test_data['Fare']>=204.0) & (test_data['Fare']<307.0),'Fare']=2

test_data.loc[(test_data['Fare']>=307.0) & (test_data['Fare']<409.0),'Fare']=3

test_data.loc[(test_data['Fare']>=409.0) & (test_data['Fare']<512.0),'Fare']=4

test_data.loc[test_data['Fare']>=512.0,'Fare']=5
test_data.Fare=test_data.Fare.fillna(0)

test_data.Fare=test_data.Fare.astype('int')
#test_data.Embarked=test_data.Embarked.replace({'S':0,'C':1,'Q':2})



test_data=test_data.drop(['SibSp','Parch','fare_cut'],axis=1)
test_data.head()
Id = test_data['PassengerId']
test_data=test_data.drop('PassengerId',axis=1)
test_data.head()
test_data.Sex=test_data.Sex.replace({'male':0,'female':1}).astype('int')
test_data.Age=test_data.Age.astype(int)
test_data = test_data.drop(['Cabin' , 'Fare'] , axis=1)
test_data.head()
y = data.Survived
data = data.drop('Survived' , axis=1)
data = data.drop(['PassengerId' , 'SibSp' , 'Parch' , 'Fare' , 'Fare_cut'] , axis=1)
data.head()
# Logistic Regression



logistic = LogisticRegression().fit(data , y)

logistic_score = logistic.score(data , y)

logistic_score
# SVC

vector = SVC().fit(data , y) 

vector_score = vector.score(data, y)

vector_score
# decision tree  



tree = DecisionTreeClassifier().fit(data , y)

tree_score = tree.score(data , y  )

tree_score
# random forest  



random=RandomForestClassifier(n_estimators=100).fit(data,y)

random_score=random.score(data,y)

random_score
# KNN 



knn=KNeighborsClassifier(n_neighbors=3).fit(data,y)

knn_score=knn.score(data,y)

knn_score
# final opting for Random forest classifier 



predict_y = random.predict(test_data)
prediction = pd.DataFrame({'PassengetId':Id , 'Survived':predict_y})
prediction
my_submission = pd.DataFrame(prediction)

my_submission.to_csv('submission.csv', index = False)
my_submission