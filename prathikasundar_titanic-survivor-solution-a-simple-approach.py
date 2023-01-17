import pandas as pd
import numpy as np
#import os
import matplotlib.pyplot as plt
#os.chdir('../input')
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
# The train data has survived column (which is our target attribute), whereas the test data doesn't contain it
# We should remove the survived column from train data later
test_data.head()
# Have a glimpse of both training and test data summary
train_data.describe()
test_data.describe()
# Purely going with the gender for prediction first
# With the both dataset we could see that age has missing columns
# We can either drop out them or fill the missing values with mean value
import seaborn as sns
sns.countplot(data=train_data,x='Sex',hue='Survived')
train_data['Age'].fillna((train_data['Age'].mean()), inplace=True)

train_data.describe(include='all')
# Survivor based on Pclass
sns.countplot(data=train_data,x='Pclass',hue='Survived')
# Fill the missing age values with mean value
test_data['Age'].fillna((test_data['Age'].mean()), inplace=True)

test_data.describe(include='all')
train_data['Family'] =  train_data["Parch"] + train_data["SibSp"]


test_data['Family'] =  test_data["Parch"] + test_data["SibSp"]


# drop Parch & SibSp
train_data = train_data.drop(['SibSp','Parch'], axis=1)
test_data = test_data.drop(['SibSp','Parch'], axis=1)
train_data
# Convert the string (gender) to numeric value
gender = {'male':0, 'female':1}
train_data['Sex'] = train_data['Sex'].apply(lambda x: gender.get(x))
test_data['Sex'] = test_data['Sex'].apply(lambda x: gender.get(x))
#columns_to_drop = ['Name','SibSp','Parch','Ticket','Fare','Cabin']
# The below columns can be dropped, just trying to predict the survival based on Age,Sex and Pclass for now
# columns_to_drop = ['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
columns_to_drop = ['Name','Ticket','Fare','Cabin','Embarked']
X_train = train_data.drop(columns_to_drop+['Survived'],axis=1)
Y_train = train_data['Survived']
X_test = test_data.drop(columns_to_drop, axis=1)
X_test.head()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=10,min_samples_split=2,n_estimators=100 , random_state=1 )
rf_model = forest.fit(X_train,Y_train)

my_prediction = rf_model.predict(X_test)

my_solution = pd.DataFrame(my_prediction, X_test.PassengerId, columns = ["Survived"])

my_solution.to_csv("titanic_own_soln_family_size.csv", index_label = ["PassengerId"])
X_test.shape
X_train.shape
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=10,min_samples_split=2,n_estimators=100 , random_state=1 )
rf_model = forest.fit(X_train,Y_train)
print(rf_model.feature_importances_)
rf_model.score(X_train, Y_train)