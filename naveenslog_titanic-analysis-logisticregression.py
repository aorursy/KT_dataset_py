# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Checking for missing values
sns.heatmap(train.isnull(), yticklabels=False,cbar=False)

train.isna().sum()
train["Age"].isna().sum()/len(train) # 19 % of age is missing we can deal with the missing data
train["Cabin"].isna().sum()/len(train) # 77 % of the data is missing so we cant consider Cabin for analysis 
train["Embarked"].isna().sum()/len(train) # Only two missing values so we can assign the most frequest value to this
# Plotting histogram to check how we can deal with the missing value
# Handling Age missing value

plot = train["Age"].hist(bins = 15, color = 'blue', alpha = 0.8)
plot.set(xlabel = "Age", ylabel= 'Count')  # The age is right skewed so its better to use median to fill na
train["Age"].fillna(train["Age"].median(), inplace = True)


# Handling Embarked missing value
plot = train["Embarked"].hist(bins = 15, color = 'blue', alpha = 0.8)
plot.set(xlabel = "Embarked", ylabel= 'Count')  # Most of the passengers boarded from southhampton
train["Embarked"].fillna("S", inplace = True) # Assigning Southhampton to the missing values

# Applying the smae changes in the test data-set
sns.heatmap(test.isna(), yticklabels=False, cbar=False)
test.isna().sum()

test["Age"].fillna(test["Age"].median(), inplace = True)
test.isna().sum()  # only 1 value is missing in fare column 

print(test[test['Fare'].isna()]) # The passenger with na value was from 3rd class
sns.barplot(x = 'Pclass', y = 'Fare', data = test ) 
np.mean(train[train['Pclass'] == 3]) 
test['Fare'].fillna(13.675550, inplace = True) # Replaced Na with the mean of Pclass 3
train.describe()
grid = sns.FacetGrid(train, col = 'Survived')
grid.map(sns.distplot, "Age") # More younger survived

sns.barplot('Pclass', 'Survived', data = train) # No wonders, beind 1st class was safest

sns.barplot('Embarked', 'Survived', data = train)

summary = pd.pivot_table(train[['Survived','Pclass', 'Embarked']], index = 'Embarked',
                         columns = 'Pclass', aggfunc = 'sum')
summary

sns.barplot('Sex', 'Survived', data = train) # Clearly being female incresed the changes of survival

train.columns
final_train = train[['Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]
X = pd.get_dummies(final_train)
y = train['Survived']
print(X.sample(5))
# Test train split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size = 0.20,
                                                 random_state = 0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_train,y_train)

# Making confusion matrix 
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_pred,y_test)
confusion_mat 

X_test = test[['Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]
X_test = pd.get_dummies(X_test)
final_pred = model.predict(X_test)
test["Survived"] = final_pred 
test.head()
