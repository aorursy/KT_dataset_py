import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.model_selection import RandomForest

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print('The size of the training set: ', train_data.shape)
print('The size of the test set is: ' ,test_data.shape)


train_data.head()
test_data.head()
test_data.isnull().sum()
train_data.isnull().sum()
for x in [train_data, test_data]:
    x['Age'] = x['Age'].fillna (x['Age'].median())
    x['Fare'] = x['Fare'].fillna (x['Fare'].mean())
#lest take a look at the data again
#We will take care of the cabin situation later
test_data.isnull().sum()
#lets see what is the percentage of all enbarkes
train_data['Embarked'].value_counts(normalize = True)
#we see that 72 percent of all enkarked were S so we use S
train_data['Embarked'] = train_data['Embarked'].fillna('S')
#Lets take a look at uour data onr more time
train_data.isnull().sum()

lb = preprocessing.LabelBinarizer()
for x in [train_data, test_data]:
    x['Sex'] = lb.fit_transform(x['Sex'])
input_Embarked = {'S':0, 'Q':1, 'C':2}
train_data['Embarked'] = train_data['Embarked'].map(input_Embarked)
test_data['Embarked'] = test_data['Embarked'].map(input_Embarked)
train_data.tail()
train_data.drop(columns = ['Name', 'Ticket','Cabin'])
test_data.drop(columns = ['Name', 'Ticket','Cabin'])
train_data.dtypes.value_counts()
test_data.dtypes.value_counts()


