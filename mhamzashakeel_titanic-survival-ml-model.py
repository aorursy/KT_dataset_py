import pandas as pd # for dataframes
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns # for data visualization

%matplotlib inline
train = pd.read_csv("../input/titanic-train/titanic_train.csv")
train.head()
train.isnull().sum()
# using seaborn
sns.heatmap(train.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)
# countplot for death and survival rate
sns.set_style("whitegrid")
sns.countplot(x = "Survived", data = train, palette = 'rainbow')
# countplot for male and female
sns.set_style('darkgrid')
sns.countplot(x = "Sex", data = train, palette = 'rocket')
# countplot for people died against Pclass
sns.set_style('whitegrid')
sns.countplot(x = "Survived", hue = 'Pclass', data = train)
# getting the count of Age person
train['Age'].hist(bins=40, color='darkred', alpha=0.5)
train['Fare'].hist(bins=20, color='purple', alpha = 0.7, figsize = (10, 5))
plt.figure(figsize = (10, 5))
sns.boxplot(x = 'Pclass', y = 'Age', data = train, palette = 'winter')
# making a method for imputing age
def impute_age(cols):
    # we will pass 2 cols as arguments, col at 0 index will be for age and col at 1 index will be for Pclass
    Age = cols[0]
    Pclass = cols[1]
    
    # getting null values
    if pd.isnull(Age):
        # returning avg. age (37) for 1st Class
        if Pclass == 1:
            return 37
        # for 2nd class age (28)
        elif Pclass == 2:
            return 28
        # for 3rd class age (25)
        else:
            return 25
    else:
        return Age
            
# applying above function
# col[0] = 'Age' & col[1] = 'Plcass'
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
sns.heatmap(train.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)
# cabin
train.drop('Cabin', inplace = True, axis = 1)
sns.heatmap(train.isnull())
train.shape # 1 column is dropped now
train.head()
train.info() # 'object' are strings in python
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
sex # into numerical values
embark # into numerical
# dropping extra cols
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train # categorical values dropped here!
train = pd.concat([train, sex, embark], axis = 1)
train # all values in numercial form! yay!!
from sklearn.model_selection import train_test_split
# we required 'Survived' values on Y-Axis
Y = train['Survived'] # Y == Survived column
# all other features will be on X-axis. So, we are dropping 'Survived' column and storing all others
X = train.drop(['Survived'], axis=1) # X == all cols, excluding Survived column
X
# splitting Testing and Training data with 20-80 margine
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
# training our Logistic Regression Model here
logReg.fit(X_train, Y_train)
# making predictions on Testing data
predictions = logReg.predict(X_test)
from sklearn.metrics import classification_report
# using actual testing data and the predictions our Model just made
print(classification_report(Y_test, predictions)) # getting accuracy
# for making a DataFrame shape must be same
pred = logReg.predict(X)
pred.shape
X.shape # same number of Rows
# making our own DataFrame with 'Submission' as Name
submission = pd.DataFrame({
    'PassengerId' : X['PassengerId'],
    'Survived' : pred
})
submission.head