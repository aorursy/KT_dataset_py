#Libraries

import numpy as np

import pandas as pd
df_training = pd.read_csv('../input/train.csv')



df_training.shape
df_training.dtypes
# Lets remove passenger id out of the training set and store it in another variable

training_passengerId = df_training.PassengerId



df_training.drop(columns=['PassengerId'],inplace=True)



#dropping Name and Ticket and fare as well out of the data

df_training.drop(columns=['Name','Ticket','Fare'],inplace=True)

df_training.head()
#Lets annalyze the values of remaining data



print('Survived value counts: ')

print(df_training.Survived.value_counts())



print('Count by class: ')

print(df_training.Pclass.value_counts())



print('count by sex: ')

print(df_training.Sex.value_counts())



print('Cabin or without cabin count')

print('Without cabin', df_training.Cabin.isnull().sum())

print('With cabin', df_training.shape[0] - df_training.Cabin.isnull().sum())



print('Count by Journey Embarking point:')

print(df_training.Embarked.value_counts())
#creating category types

df_training.Survived=df_training.Survived.astype('category')

df_training.Pclass=df_training.Pclass.astype('category')

df_training.Sex=df_training.Sex.astype('category')

df_training.Embarked = df_training.Embarked.astype('category')



# lets do feature engineering using cabin. if a passenger has cabin and if a passenger doesnot have a cabin.

df_training['cabinAllocated'] = df_training.Cabin.apply(lambda x: 0 if type(x)==float else 1)

df_training['cabinAllocated'] = df_training['cabinAllocated'].astype('category')
df_training.dtypes
# Lets drop Cabin first

df_training.drop(columns=['Cabin'],inplace=True)
print("Min Age : {}, Max age : {}".format(df_training.Age.min(),df_training.Age.max()))
df_training.Age.isnull().sum()
random_list = np.random.randint(df_training.Age.mean() - df_training.Age.std(), 

                                         df_training.Age.mean() + df_training.Age.std(), 

                                         size=df_training.Age.isnull().sum())

df_training['Age'][np.isnan(df_training['Age'])] = random_list

df_training['Age'] = df_training['Age'].astype(int)
# Lets divide age in 5 bins



df_training['AgeGroup'] = pd.cut(df_training.Age,5,labels=[1,2,3,4,5])

#As we have categorized age into AgeGroup, lets remove Age

df_training.drop(columns=['Age'],inplace=True)
#Adding 1 to indicate the person in that row

df_training['family'] = df_training.Parch+df_training.SibSp+1
df_training.drop(columns=['SibSp','Parch'],inplace=True)

df_training.head()
df_training['Sex'].value_counts()
df_training['category_sex'] = df_training['Sex'].apply(lambda x: 1 if x=='male'  else 0)
df_training.drop(columns=['Sex'],inplace=True)
df_training.Embarked.value_counts()
df_training.Embarked = df_training.Embarked.fillna('S')

df_training.Embarked = df_training.Embarked.map({'S':1,'C':2,'Q':3}).astype('int')
df_training.Embarked.value_counts()
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(df_training.iloc[:,1:],df_training.iloc[:,0],test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier



randomForest = RandomForestClassifier(n_estimators=100)



randomForest.fit(train_x,train_y)
y_hat = randomForest.predict(test_x)
from sklearn.metrics import accuracy_score

accuracy_score(test_y,y_hat)
randomForest.fit(df_training.iloc[:,1:],df_training.iloc[:,0])
df_testing = pd.read_csv('../input/test.csv')



# Lets remove passenger id out of the training set and store it in another variable

testing_passengerId = df_testing.PassengerId



df_testing.drop(columns=['PassengerId'],inplace=True)



#dropping Name and Ticket and fare as well out of the data

df_testing.drop(columns=['Name','Ticket','Fare'],inplace=True)

df_testing.head()



#creating category types

df_testing.Pclass=df_testing.Pclass.astype('category')

df_testing.Sex=df_testing.Sex.astype('category')

df_testing.Embarked = df_testing.Embarked.astype('category')



# lets do feature engineering using cabin. if a passenger has cabin and if a passenger doesnot have a cabin.

df_testing['cabinAllocated'] = df_testing.Cabin.apply(lambda x: 0 if type(x)==float else 1)

df_testing['cabinAllocated'] = df_testing['cabinAllocated'].astype('category')



# Lets drop Cabin first

df_testing.drop(columns=['Cabin'],inplace=True)



random_list_test = np.random.randint(df_testing.Age.mean() - df_testing.Age.std(), 

                                         df_testing.Age.mean() + df_testing.Age.std(), 

                                         size=df_testing.Age.isnull().sum())

df_testing['Age'][np.isnan(df_testing['Age'])] = random_list_test

df_testing['Age'] = df_testing['Age'].astype(int)



# Lets divide age in 5 bins



df_testing['AgeGroup'] = pd.cut(df_testing.Age,5,labels=[1,2,3,4,5])





#As we have categorized age into AgeGroup, lets remove Age

df_testing.drop(columns=['Age'],inplace=True)



#Adding 1 to indicate the person in that row

df_testing['family'] = df_testing.Parch+df_testing.SibSp+1



df_testing.drop(columns=['SibSp','Parch'],inplace=True)



df_testing['category_sex'] = df_testing['Sex'].apply(lambda x: 1 if x=='male'  else 0)

df_testing.drop(columns=['Sex'],inplace=True)



df_testing.Embarked = df_testing.Embarked.fillna('S')

df_testing.Embarked = df_testing.Embarked.map({'S':1,'C':2,'Q':3}).astype('int')

submission_data = pd.DataFrame({'PassengerId':testing_passengerId, 'Survived':randomForest.predict(df_testing)})



submission_data.to_csv("Submission_Data.csv",index=False)