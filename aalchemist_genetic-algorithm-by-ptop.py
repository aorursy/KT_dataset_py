%matplotlib inline
import numpy as np
import pandas as pd
import re as re

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
ids = test['PassengerId']
full_data = [train, test]

print (train.info())
print (test.info())
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
import random
for dataset in full_data:
    a=random.choice(['S','C','Q'])
    dataset['Embarked'] = dataset['Embarked'].fillna("S")
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].fillna('0')
#train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
for dataset in full_data:
    dataset.loc[ dataset['Cabin'] != '0', 'Cabin'] = '1'
    dataset['Cabin'] = dataset['Cabin'].astype(int)
print (train[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean())
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    #dataset['Fare']=dataset['Fare']/max(dataset['Fare'])
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    #dataset['Age']=dataset['Age']/max(dataset['Age'])

# Feature Selection . 
drop_elements = ['PassengerId', 'Name', 'Ticket'
                 ]
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)


test  = test.drop(drop_elements, axis = 1)

print (train.head(10))

train = train.values
test  = test.values
from keras.utils import np_utils
X = train[0::, 1::]
y = train[0::, 0]
#Y=np_utils.to_categorical(y)
from tpot import TPOTClassifier
model = TPOTClassifier(generations=10,population_size=100,random_state=433, verbosity=2)
model.fit(X,y)
#model.fit(train[0::, 1::], train[0::, 0])
result = model.predict(test)
#result=[np.argmax(pred) for pred in result]
pdtest = pd.DataFrame({'PassengerId': ids,
                            'Survived': result})
pdtest.to_csv('gptest.csv', index=False)