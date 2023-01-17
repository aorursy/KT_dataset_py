import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
%matplotlib inline
training = pd.read_csv('../input/titanic/train.csv')

testing = pd.read_csv('../input/titanic/test.csv')
training.info()
testing.info()
training.head()
training.describe()
training['Sex'].value_counts()
sns.catplot('Sex',data=training , kind='count')
sns.catplot('Pclass',data=training,hue='Sex' , kind='count')
training['Age'].hist(bins=20)
training['Age'].plot.kde()
training["Survivor"] = training.Survived.map({0: "No", 1: "Yes"})



 

sns.catplot('Survivor',data=training, kind = 'count')
training.isnull().sum()
mean = training['Age'].mean()

std = training['Age'].std()



low  = mean - std

high = mean + std

Age_missing = training['Age'].isnull().sum()



Age_rand = np.random.randint(low=low , high=high , size =Age_missing)



training['Age'][np.isnan(training['Age'])]=Age_rand

training['Age'] = training['Age'].astype(int)


training['Age'].isnull().sum()
training['Embarked'].value_counts()
training['Embarked'] = training['Embarked'].fillna('S')
training['Embarked'].isnull().sum()
#training['Fare']=training['Fare'].astype(int)

testing.isnull().sum()
mean_test = testing['Age'].mean()

std_test = testing['Age'].std()



low_test  = mean_test - std_test

high_test = mean + std

Age_missing_test = testing['Age'].isnull().sum()



Age_rand = np.random.randint(low=low_test , high=high_test , size =Age_missing_test)



testing['Age'][np.isnan(testing['Age'])]=Age_rand

testing['Age'] = testing['Age'].astype(int)
testing['Fare'].plot.kde()


testing['Fare']=testing['Fare'].fillna(testing['Fare'].median())
testing['Age'].isnull().sum()
training = pd.get_dummies(training , columns = ['Pclass' , 'Sex' , 'Embarked' ] , drop_first = True)
training.head()
X = training.iloc[:, [3 ,4,5,7,10,11,12,13,14]].values

y = training.iloc[:, 1].values
testing = pd.get_dummies(testing , columns = ['Pclass' , 'Sex' , 'Embarked' ] , drop_first = True)
testing.head()
X_test = testing.iloc[:, [2,3 ,4,6,8,9,10,11,12]].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X, y)
survived_test = classifier.predict(X_test)
from sklearn.svm import SVC

classifier = SVC(random_state = 0)
classifier.fit(X, y)
survived_test = classifier.predict(X_test)
len(survived_test)
df = pd.DataFrame({'PassengerId':testing['PassengerId'],'survived':survived_test })
df.rename({'PassengerId':'PassengerId','suvivied':'survived_test'},inplace='True')
df.shape
df.to_csv('Titanic_prediction.csv',index=False)