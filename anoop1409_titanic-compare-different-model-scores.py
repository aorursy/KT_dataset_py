#Import all the necessary python packages to perform data exploration and prediction

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier #Simple model for initial classification

from sklearn.model_selection import train_test_split # For splitting train and test data
#Read both training dataset and test dataset

train_data = pd.read_csv(r'../input/train.csv') 

test_dataset = pd.read_csv(r'../input/test.csv')
#Get information about the training dataset

train_data.info()
# Remove the Survived column from train data and put into a target variable test_data. 

# Because this is the feature / target that we need to predict

test_data= train_data.pop('Survived')
# We do not need passenger name and Passenger Id for prediction as they do not impact the result at all.

train_data.pop('Name')

train_data.pop('PassengerId')
# We can even ignore the Ticket feature as a ticket can not help in predicting the survival.

train_data.pop('Ticket')
# Get an overview of the training dataset now.

train_data.head()
# Lets get more information about the Pclass feature

train_data['Pclass'].value_counts()
# Now we know that the Pclass has only 3 values i.e 3, 1 and 2. 

# So, it is a categorical feature and hence we should convert to dummy variable.
train_data = pd.get_dummies(train_data, columns = ['Pclass'])
# Here is how the new dataset looks like

train_data
train_data['Sex'].value_counts()
# So, there are 577 males and 314 females. Again we need to create dummy variable for this.

# This is required because Sex is also a categorical feature and has only two values - male and female
train_data = pd.get_dummies(train_data, columns = ['Sex'])
# Lets see the new train dataset

train_data
# Since age is a continuous variable and not categorical, we should not create dummy

# Lets check for any missing data in age

train_data['Age'].count()
# So, we have only 714 rows with age data. Whereas we should be having 891 rows.

# We need to fill the empty Age data with some value.

# This value should be derived logically. The most common way is to find the mean age 

# And fill the empty values with the mean age.



train_data['Age'][train_data['Age'].isnull()] = train_data['Age'].mean()
# Let's see whether its a continuous or categorical variable

train_data['SibSp'].value_counts()
# As you can see, it just has 7 values, it is a categorical variable.

# So, we should create dummy variable for this

# I.e create a separate feature for each value
train_data = pd.get_dummies(train_data, columns = ['SibSp'])
# Lets see the new Training data

train_data
train_data['Parch'].value_counts()
# This is again a categorical variable and hence we need to create dummy variables
train_data = pd.get_dummies(train_data, columns=['Parch'])
train_data['Fare'].count()
# Since there is no missing data in Fare feature, we need not do much of feature engineering here
train_data['Cabin'].value_counts()
# Looks like Cabin is not a categorical variable and has alphanumeric values.

# So, we should be good to ignore the data for Cabin

train_data.pop('Cabin')
train_data['Embarked'].value_counts()
# So, Embarked is a categorical variable and we need to create dummy variables out of it

# Before that, Embarked has only 889 values. We need to fill the two missing rows

# Here we can use the value that appears the max number of times to fill the blank two values

# So, the value to be used will be S. Because S appears 644 times. So, it should be a fair guess 

# to use S as the missing value
# FIll the empty two rows with value S

train_data['Embarked'][train_data['Embarked'].isnull()] = 'S'
train_data = pd.get_dummies(train_data, columns=['Embarked'])
train_data
# If you check the test data, there is an additional PARCH_9 value 

# which is not available in the training data. So, adding a new feature Parch_9 with value = 0

train_data['Parch_9'] = 0
train_data.info()


#Lets split the training and test data for predicting the score

X_train,X_test,y_train,y_test = train_test_split(train_data, test_data, random_state =100)
# Initialize the KNN classifier

knn = KNeighborsClassifier(n_neighbors=1)
# Train the model

knn = knn.fit(X_train, y_train)
#Calculate the score on training data

knn.score(X_train, y_train)
#Calculate the score on test data

knn.score(X_test, y_test)
from sklearn import tree
my_tree = tree.DecisionTreeClassifier(random_state=1)
my_tree = my_tree.fit(X_train, y_train)
my_tree.score(X_train, y_train)
my_tree.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier
my_forest = RandomForestClassifier(max_depth = 10, min_samples_split=2,

                                   random_state=10, n_estimators=80)
my_forest = my_forest.fit(X_train, y_train)
my_forest.score(X_train, y_train)
my_forest.score(X_test, y_test)
#Random forest gave the best prediction, so we should use this model to submit our prediction to Kaggle
test_dataset.head()
# Need to pop out all unwanted features

PassengerId = test_dataset.pop('PassengerId')

test_dataset.pop('Name')

test_dataset.pop('Ticket')

test_dataset.pop('Cabin')
test_dataset.info()
# Fill empty rows just like we did for training dataset

test_dataset['Age'][test_dataset['Age'].isnull()] = test_dataset['Age'].mean()
test_dataset['Fare'][test_dataset['Fare'].isnull()] = test_dataset['Fare'].mean()
test_dataset.info()
#Create dummy variables just like we did for training dataset

test_dataset = pd.get_dummies(test_dataset, columns = ['Pclass'])
test_dataset = pd.get_dummies(test_dataset, columns = ['Sex'])
test_dataset = pd.get_dummies(test_dataset, columns = ['SibSp'])
test_dataset = pd.get_dummies(test_dataset, columns=['Parch'])
test_dataset = pd.get_dummies(test_dataset, columns=['Embarked'])
test_dataset.head()
test_dataset.shape
train_data.shape
# Predict the result on test data

prediction = my_forest.predict(test_dataset)
# Save in CSV and submit to Kaggle.