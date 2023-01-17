# Data Manipulation & Analysis Libraries
import numpy as np
import pandas as pd

# Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# For the Machine Learning Processes
from sklearn import linear_model
from sklearn import metrics
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Looking at all the values, we can see that there are null values in 3 columns. 
# I'm not sure how to account for the missing values in Cabin. 
# Because there are only 2 missing values in the Embarked column, I will ignore it. 
# Age is a column that will be fixed. 

# PassengerID does not contribute to survival rate and can thus be dropped. 
# Intuitively speaking, name does not seem to correlate with survival and will be dropped.

# We also have a few assumptions of who may have had a better chance of survival:
# Females, children under a certain age, and people of higher fare/class.

train.info()
# I want to first fix the age.
# I fill the null values of the age with the mean of the age.

age = np.nanmean(train['Age'])
train['Age'] = train['Age'].fillna(age)
# Next I will try to find correlations between features and survival rate. 
# I will start off with Pclass

train[['Survived', 'Pclass']].groupby('Pclass').count()
# It is indeed clear that upper class people had a better survival rate. 
# We can also plot this and visualize it in the following two ways:
    # The countplot shows that 3rd class had higher deaths than survivals. 
        # This is not true for first class and barely the case for second class. 
    # The factorplot shows that chances of survival decreases with class.
        # I will also add a hue marker to see the difference between males and females. 

plt.subplot(1, 2, 2)
sns.countplot('Pclass', data = train, hue = 'Survived', palette = 'coolwarm')

plt.subplot(1, 2, 2)
sns.factorplot(x = 'Pclass',y = 'Survived', data = train, hue = 'Sex')
# In addition to Pclass, being a female increases your survival rate. 
# Upper class females had the best chance of survival it seems. 

# Numerically, we can see females had a better chance of survival below:
train[['Sex', 'Survived']].groupby('Sex').mean()
# This can also once again be visualized. 
# Survival ratio of female is favorable but very unfavorable for males. 

sns.countplot(x = 'Sex', data = train, hue = 'Survived', palette = 'seismic')
# Now that we've established a correlation for sex and class, let's check age. 
# I am going to plot two distributions of age. 
    # One for those who survived and one for those who did not. 

# Children seemed likelier to survive. 
# A slight uptick at the end shows that some of the oldest passengers survived. 

age = sns.FacetGrid(train, col = 'Survived')
age.map(plt.hist, 'Age')
# Fare prices would make sense with survival rate, but may not be independent of class. 

train[['Fare', 'Pclass']].groupby('Pclass').mean()
# Fares decrease with class which makes sense. 
# To add to it, fare price correlates positively with survival. 
    # This seems to work again to the advantage of females and slower for males. 

sns.lmplot(x = 'Fare', y = 'Survived', data = train, hue = 'Sex', 
           logistic = True, palette = 'Set1')
# It might be interesting to see if where passenger boarded from would've had any affect. 
# This would be the Embarked feature. 

em = sns.FacetGrid(train, col = 'Embarked', height = 2.2, aspect = 1.6)
em.map(sns.pointplot, 'Pclass', 'Survived')
# There does not seem to be too much of a correlation here. 



# We can now move on to drop out a few features as this will speed up the process. 
# We will drop cabin as there are simply too many missing points to consider it. 
# We will also drop Passenger ID as this in itself does not increase survival. 
# We will drop ticket as we have details of fares already included. 
# As discussed initially, name will also be dropped. 
# This is mimicked in the test data as well. 



train = train.drop(['PassengerId', 'Cabin', 'Ticket', 'Name', 'Embarked'], axis = 1)
test = test.drop(['Cabin', 'Ticket', 'Name', 'Embarked'], axis = 1)
# Let's edit some more data before moving on. 
# One of our features, Sex, can be converted into a numerical category. 

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Two of our features represent similar attributes:
    # SibSp: Siblings aboard
    # Parch: Parents/Children aboard 
# We can remove these and simply create one feature column that represents family size.

train['Family Members'] = train['SibSp'] + train['Parch']
test['Family Members'] = test['SibSp'] + test['Parch']

sns.countplot('Family Members', data = train)
# It seems like most people were travelling alone. 
# As such, we can simply categories people as alone or with family. 
    # Let 1 represent persons with family and 0 represent alone. 


# Here is a function that will return 0 or 1 depending on Family Members
def family_or_alone(data):
        if data == 0:
            return 0
        else:
            return 1

# Create a new column and apply it to the Family Members column
train['Family'] = train['Family Members'].apply(family_or_alone)
test['Family'] = test['Family Members'].apply(family_or_alone)

# Drop the Family Members, SibSp, and Parch columns as it is no longer needed. 
train = train.drop(['Family Members', 'SibSp', 'Parch'], axis = 1)
test = test.drop(['Family Members', 'SibSp', 'Parch'], axis = 1)

train.head()
# A quick glance shows that those with family were more likely to survive.

train[['Family', 'Survived']].groupby('Family').mean()
# Now we move on to creating a model and predicting on the test set. 

# I will first separate the training set into features and target. 

X_train = train.drop('Survived', axis = 1)
Y_train = train['Survived']
X_test = test.drop('PassengerId', axis = 1)

X_train.shape, Y_train.shape, X_test.shape
# X_test still has a couple of features that need a quick fix. 

X_test.info()
# I will first fill the missing age values with the mean as was done in the training set.
# Same thing for missing fare. 

test_age = np.nanmean(X_test['Age'])
X_test['Age'] = X_test['Age'].fillna(test_age)

fare = np.nanmean(X_test['Fare'])
X_test['Fare'] = X_test['Fare'].fillna(fare)
# I will first try to use logistic regression to model and predict. 
# It is one of the simpler classification algoithms and is easy to use. 


# I will first create an instance of logistic regression. 
# Fit it on the training set X_train, Y_train
# Then use it to predict using X_test data

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
# Let's print out the coefficients of each feature and compare it to correlation matrix. 

# First a series that contains the coefficients
co = pd.Series(logreg.coef_[0])

# Match it up with it's features. 
co_df = pd.DataFrame({'Features': X_train.columns, 'Coefficients': co})
co_df.sort_values(by = 'Coefficients', ascending = False)
# In both P class is the most inversely correlated with survival rate.  
# In both Sex is the most positively correlated survival rate. 

train.corr()
# We can calculate confidence score for our algorithm. 

logistic_score = logreg.score(X_train, Y_train) * 100
print(logistic_score)
sub = pd.DataFrame({'PassengerID': test['PassengerId'], 'Survived': Y_pred})
Filename = 'Titanic Submission.csv'

sub.to_csv(Filename, index = False)