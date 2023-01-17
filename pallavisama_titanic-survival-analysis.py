# Loading the required libraries/packages



# For data loading and manipulation

import pandas as pd

import numpy as np



# For Visualization/EDA

import seaborn as sns

sns.set(style="white")

import matplotlib.pyplot as plt

%matplotlib inline



# For data science and machine learning techniques

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
# Read the train and test datasets from Kaggle to create two DataFrames using Pandas

train_loc = "../input/train.csv"

test_loc = "../input/test.csv"

train = pd.read_csv(train_loc)

test = pd.read_csv(test_loc)

# Print the first few records of the train and test datasets

print(train.head())

print(test.head())
# Checking the dimensions of the train dataset

train.shape
# Checking the distribution of the numerical columns

train.describe()
# Getting a summary on data types and missingness of each column

train.info()
# Treating missing values (Age, Cabin, Embarked)

# Check the average Age by Gender

train.groupby('Sex')['Age'].mean()
# Impute missing values of age by the respective average of the genders

train.loc[(train.Age.isnull())&(train.Sex=='female'),'Age'] = train["Age"][train["Sex"] == 'female'].mean()

train.loc[(train.Age.isnull())&(train.Sex=='male'),'Age'] = train["Age"][train["Sex"] == 'male'].mean()

train.Age.isnull().any()
# Check the Cabin distribution

#train.groupby('Cabin').size()

len(train.Cabin.unique())
# How much information does it provide?

train.Cabin.isnull().sum()/(train.shape[0])
# Dropping Cabin from the analysis

train.drop(['Cabin'],axis=1,inplace=True)
# Check the Embarked distribution

train.groupby('Embarked').size()
# Since S=Southampton is the most occuring port of embarkation, impute the missing values with that (Mode)

train.loc[(train.Embarked.isnull()),'Embarked'] = "S"

train.info()
# Overall distribution of the survived passengers

train["Survived"].value_counts(normalize = True)
# Unfortunately, 62% of the passengers could not survive. Below is the bar chart for the absolute numbers

sns.countplot('Survived',data=train).set_title('Survival Count')
# Survival Rates by Gender

pd.crosstab(train.Survived, train.Sex, normalize='index')
# Plot the survival count by Gender

sns.countplot('Sex',hue='Survived',data=train).set_title('Survival by Gender')
# Next we will explore the Passenger Class variable

pd.crosstab(train.Survived, train.Pclass, normalize='index')
# Plot the survival count by Passenger Class

sns.countplot('Pclass',hue='Survived',data=train).set_title('Survival by Passenger Class')
# Create a boxplot for the Fare distribution of each class

sns.boxplot("Pclass", "Fare", data=train, palette=["lightblue", "lightpink", "lightgreen"]).set_title('Fare Distribution by Pclass')
# Same story told by the fare distributions of survived vs. non survived passensers below

plt.figure(figsize = (10,5))

sns.kdeplot(train["Fare"][train.Survived == 0], color = "lightpink", shade = True)

sns.kdeplot(train["Fare"][train.Survived == 1], color = "lightgreen", shade = True)

plt.title("Fare distribution of Survivors vs. Non Survivors")

plt.legend(['Survived = 0', 'Survived = 1'])
# Let's look into the Age factor

# Like fare, it is a continuous variable so let's plot a histogram to check the distribution

sns.FacetGrid(train, col='Survived').map(plt.hist, 'Age', bins=20)
# Create a violin plot to check the distribution of Age across the Survivors vs. Non Survivors for each Gender

sns.violinplot("Sex","Age", hue="Survived", data = train, split = True).set_title("Age distribution of Survivors vs. Non Survivors by Gender")
# Next, we explore the Embarked variable

pd.crosstab(train.Survived, train.Embarked, normalize='index')
# Plot the absolute count of Survivors by Port

sns.countplot('Embarked',hue='Survived',data = train).set_title('Survival by Port of Embarkation')
# Create a factor plot to include Pclass and Gender variables

sns.factorplot('Embarked', 'Survived', hue = 'Sex', col = 'Pclass', data = train)
# Check if the ticket variable has any value add

#train.groupby('Ticket').size()

len(train.Ticket.unique())
# Drop the ticket variable

train.drop(['Ticket'],axis=1,inplace=True)
# Next, let's explore the SibSp variable

pd.crosstab(train.Survived, train.SibSp, normalize='index')
# Create a factor plot to include Gender

sns.factorplot('SibSp', 'Survived', hue = 'Sex', data = train)

plt.title('Survival rates of SibSp by Gender')
# Lastly, we take a look at the Parch variable

pd.crosstab(train.Survived, train.Parch, normalize='index')
# Create a factor plot to include Gender

sns.factorplot('Parch', 'Survived', hue = 'Sex', data = train)

plt.title('Survival rates of Parch by Gender')
# From the 12 columns of the training dataset, we have already dropped Cabin and Ticket

# We do not need Name and PassengerID, so let's drop them

train.drop(['Name','PassengerId'],axis=1,inplace=True)

train.info()
# Convert the male and female groups to integer form

train["Gender"] = 0

train.loc[train['Sex']=='male','Gender']=0

train.loc[train['Sex']=='female','Gender']=1



# Convert the Embarked classes to integer form

train["Port"] = 0

train.loc[train['Embarked']=='S','Port']=0

train.loc[train['Embarked']=='C','Port']=1

train.loc[train['Embarked']=='Q','Port']=2



# Create buckets for Age

train["Age_cat"] = 0

train.loc[train['Age']<=12,'Age_cat']=0

train.loc[(train['Age']>12)&(train['Age']<=20),'Age_cat']=1

train.loc[(train['Age']>20)&(train['Age']<=35),'Age_cat']=2

train.loc[(train['Age']>35)&(train['Age']<=50),'Age_cat']=3

train.loc[train['Age']>50,'Age_cat']=4



# Create buckets for Fare

train["Fare_cat"] = 0

train.loc[train['Fare']<=8,'Fare_cat']=0

train.loc[(train['Fare']>8)&(train['Fare']<=15),'Fare_cat']=1

train.loc[(train['Fare']>15)&(train['Fare']<=31),'Fare_cat']=2

train.loc[train['Fare']>31,'Fare_cat']=3



# Create a new variable family size and buckets for the same as travel_company

train["family_size"] = train["SibSp"] + train["Parch"] + 1

train["travel_company"] = 0

train.loc[train['family_size']<=1,'travel_company']=0

train.loc[(train['family_size']>1)&(train['family_size']<=4),'travel_company']=1

train.loc[train['family_size']>4,'travel_company']=2
# Remove the unneccessary vaiables and make sure the new variables got added

#train.describe()

#train.info()

train.drop(['Sex','Age','SibSp','Parch','Fare','Embarked','family_size'],axis=1,inplace=True)

#train.describe()

train.info()
# Check the correlation among the rest of the available variables

sns.heatmap(train.corr()).set_title('Correlation Heat map for candidate variables')
# Separating the response (y) and explanatory (x) variables

#X = train[["Pclass", "Gender", "Port", "Age_cat", "Fare_cat", "travel_company"]].values

# Removing the Fare category increases the model accuracy, so decided to exclude that from the final models

X = train[["Pclass", "Gender", "Port", "Age_cat", "travel_company"]].values

y = train["Survived"].values
# Splitting the dataset into test and training with 80% for training the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=25)
# Building the Logistic Regression model using the training dataset

LogReg = LogisticRegression()

LogReg.fit(X_train, y_train)



# Testing the model with the test dataset (do not confuse with the actual Test dataset)

y_pred = LogReg.predict(X_test)

print('The model accuracy is', metrics.accuracy_score(y_pred, y_test))

print('The R-square value is', metrics.r2_score(y_pred, y_test)) 

#Although R-square doesn't provide a lot of info in binary models
# Building the Decision tree model using the training dataset

DecTree = tree.DecisionTreeClassifier()

DecTree.fit(X_train, y_train)



# Testing the model with the test dataset (do not confuse with the actual Test dataset)

y_pred = DecTree.predict(X_test)

print('The model accuracy is', metrics.accuracy_score(y_pred, y_test))

print('The R-square value is', metrics.r2_score(y_pred, y_test))

#Although R-square doesn't provide a lot of info in binary models
# Building the Random forest model using the training dataset

RandFor = RandomForestClassifier(max_depth = 6, min_samples_split=2, n_estimators = 100, random_state = 1)

RandFor.fit(X_train, y_train)



# Testing the model with the test dataset (do not confuse with the actual Test dataset)

y_pred = RandFor.predict(X_test)

print('The model accuracy is', metrics.accuracy_score(y_pred, y_test))

print('The R-square value is', metrics.r2_score(y_pred, y_test))

#Although R-square doesn't provide a lot of info in binary models
# Also compare the feature importance of the Decision tree and Random forest models

print(DecTree.feature_importances_)

print(RandFor.feature_importances_)
# Pre-process and transform the data same as the training dataset

test.info()
# Impute missing values of age by the respective average of the genders

test.loc[(test.Age.isnull())&(test.Sex=='female'),'Age'] = test["Age"][test["Sex"] == 'female'].mean()

test.loc[(test.Age.isnull())&(test.Sex=='male'),'Age'] = test["Age"][test["Sex"] == 'male'].mean()

test.Age.isnull().any()



# Fare not included in the final model, but in case we want to revert, need to treat missingness

# Impute the missing value of fare by the pclass median

test.loc[(test.Fare.isnull())&(test.Pclass==1),'Fare'] = test["Fare"][test["Pclass"] == 1].median()

test.loc[(test.Fare.isnull())&(test.Pclass==2),'Fare'] = test["Fare"][test["Pclass"] == 2].median()

test.loc[(test.Fare.isnull())&(test.Pclass==3),'Fare'] = test["Fare"][test["Pclass"] == 3].median()

test.Fare.isnull().any()



# Since Cabin will be dropped so, not required to fill the missing values



# Convert the male and female groups to integer form

test["Gender"] = 0

test.loc[test['Sex']=='male','Gender']=0

test.loc[test['Sex']=='female','Gender']=1



# Convert the Embarked classes to integer form

test["Port"] = 0

test.loc[test['Embarked']=='S','Port']=0

test.loc[test['Embarked']=='C','Port']=1

test.loc[test['Embarked']=='Q','Port']=2



# Create buckets for Age

test["Age_cat"] = 0

test.loc[test['Age']<=12,'Age_cat']=0

test.loc[(test['Age']>12)&(test['Age']<=20),'Age_cat']=1

test.loc[(test['Age']>20)&(test['Age']<=35),'Age_cat']=2

test.loc[(test['Age']>35)&(test['Age']<=50),'Age_cat']=3

test.loc[test['Age']>50,'Age_cat']=4



# Create  buckets for family size/travel company

test["family_size"] = test["SibSp"] + test["Parch"] + 1

test["travel_company"] = 0

test.loc[test['family_size']>=1,'travel_company']=0

test.loc[(test['family_size']>1)&(test['family_size']<=4),'travel_company']=1

test.loc[test['family_size']>4,'travel_company']=2



test.describe()
# Extract the features from the test set and predict using the final model

test_features = test[["Pclass", "Gender", "Port", "Age_cat", "travel_company"]].values

test_Survived = RandFor.predict(test_features)



# Create a data frame with two columns: PassengerId & Survived for the final submission

Titanic_Prediction = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],

                                   'Survived': test_Survived})



# Checking for the final dimensions : 418 x 2

print(Titanic_Prediction.shape)



# Export to a csv file

Titanic_Prediction.to_csv("Titanic_Prediction.csv", index=False)