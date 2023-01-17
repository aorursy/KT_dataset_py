import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#train.info()

train.dtypes
train.head()
test.head()
# cleaning data 

train = train.drop(['Embarked', 'Name', 'PassengerId','Ticket', 'Cabin'], axis=1) 

train.head()
# convert from float to int

#train['Fare'] = train['Fare'].astype(int)

#test['Fare'] = test['Fare'].astype(int)



#train['Age'] = train['Age'].astype(int)

#test['Age'] = test['Age'].astype(int)
# check missing values 

#train.isnull().sum()



# Age - Missing Values

sum(pd.isnull(train['Age']))
missing_val = (train.isnull().sum())

print(missing_val[missing_val > 0])
# Filter rows with missing values

#train = train.dropna(axis=1)
#train.head(5)
train.groupby(['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare' ]).Survived.value_counts().head()
# Sex vs Survived

print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values

      (by='Survived', ascending=False)) 
# Convert the male/female to 0/1

train.Sex = train.Sex.map({"male": 0, "female":1})

train.Sex.head()
# prediction target

y = train.Survived



# Choosing features 

features = ['Age', 'Sex', 'Fare', 'Parch', 'SibSp', 'Pclass']

X = train[features]



#X.describe()
# check missing values 

X.isnull().sum()



# to fill NaN values with the mean

X['Age'] = X['Age'].fillna(np.mean(X['Age']))

X['Fare'] = X['Fare'].fillna(np.mean(X['Fare']))

#X.head()
#y.head()
from sklearn.tree import DecisionTreeRegressor

# Define model

titanic_model = DecisionTreeRegressor()

# Fit model

titanic_model.fit(X, y) 
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 

                                                    random_state = 100)

from sklearn.tree import DecisionTreeClassifier



clsf = DecisionTreeClassifier()

clsf.fit(X,y)
# Training accuracy

train_acc = np.mean(y_train == clsf.predict(X_train))*100

print('Training accuracy: ', train_acc)

# Validation accuracy

test_acc = np.mean(y_test == clsf.predict(X_test))*100

print('Validation accuracy: ', test_acc)
# Using max_depth to control the size of the tree to prevent overfitting



clsf = DecisionTreeClassifier(max_depth=3)

clsf.fit(X,y)
# Training accuracy

train_acc = np.mean(y_train == clsf.predict(X_train))*100

print('Depth=3, Training accuracy: ', train_acc)



# Validation accuracy

test_acc = np.mean(y_test==clsf.predict(X_test))*100

print('Depth=3, Validation accuracy: ', test_acc)

    
clsf = DecisionTreeClassifier(min_samples_leaf=5)

clsf.fit(X,y)
# Training accuracy

train_acc = np.mean(y_train == clsf.predict(X_train))*100

print('min_samples_leaf=5, Training accuracy: ', train_acc)



# Validation accuracy

test_acc = np.mean(y_test==clsf.predict(X_test))*100

print('min_samples_leaf=5, Validation accuracy: ', test_acc)

    
test[features].head()

# Convert the male/female to 0/1

test.Sex = test.Sex.map({"male": 0, "female":1})

test.Sex.head()
# Choosing features 

features = ['Age', 'Sex', 'Fare', 'Parch', 'SibSp', 'Pclass']

Xt = test[features]
#test.head()

# Check for Nan entries

print(test.isnull().sum())
# to fill NaN values with the mean

Xt['Age'] = Xt['Age'].fillna(np.mean(Xt['Age']))

Xt['Fare'] = Xt['Fare'].fillna(np.mean(Xt['Fare']))
#Make predictions 

predictions = clsf.predict(Xt)

predictions
#Create a  DataFrame 

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file 



filename = 'Titanic Predictions.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
#g = sns.FacetGrid(train, col='Survived')

#g.map(plt.hist, 'Age', bins=30)
#a2_dims = (10, 3)

#fig, ax = plt.subplots(figsize=a2_dims)

#sns.barplot(train['Sex'], train['Survived'], ax = ax)
#a2_dims = (10, 3)

#fig, ax = plt.subplots(figsize=a2_dims)

#sns.barplot(train['Age'], train['Survived'], ax = ax)