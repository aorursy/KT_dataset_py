# Importing libraries and loading the files

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



# Load files

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



# Checking for NaNs for each column

print(train.isna().sum())

print(test.isna().sum())
# To test the importance of Age, we can calculate the correlation matrix a see how it correlates with the other variables

print(train.corr()) 



# It seems pretty uncorrelated with the survived column. There is a weak correlation both with Pclass and SibSp.

# Age could be dropped and Pclass and SibSp used instead in thqe model (especially the first one).

# Fare is als correlated with the survived column. Embarked should be filled with the most frequent value of the column

train['Embarked'] = train['Embarked'].fillna('S')

train['Family_size'] = train['SibSp']+train['Parch']+1 #summing up the two columns to obtain the family size per each passenger

train['Fare_class'] = pd.qcut(train.Fare, 5, labels=False) #Fare divided into 5 classes



# We look also at the test set to check for eventual NaN and to fill what is deemed important

test['Fare'] = test['Fare'].fillna(test.Fare.mean())

test['Family_size'] = test['SibSp']+test['Parch']+1 #summing up the two columns to obtain the family size per each passenger

test['Fare_class'] = pd.qcut(test.Fare, 5, labels=False) #Fare divided into 5 classes
# Calculating the percentage of survived passengers depending on the class, sex, siblings/spouse, and parents/children

# Pclass vs Survived

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# sex vs Survived

print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# SibSp vs Survived

print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# Parch vs Survived

print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# Family_size vs Survived

print(train[["Family_size", "Survived"]].groupby(['Family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# Fare_class vs Survived

print(train[["Fare_class", "Survived"]].groupby(['Fare_class'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Data visualization: number of people per class, sex, and family size, plus how many survived per feature

f, axes = plt.subplots(4, 3, figsize=(15, 15))

sns.catplot('Pclass',data=train,kind='count',ax=axes[0,0]) #Number of people per ticket class

sns.catplot('Sex',data=train,hue='Pclass',kind='count',ax=axes[0,1]) #Number of people per sex, further divided per ticket class

sns.catplot('Family_size',data=train,hue='Pclass',kind='count',ax=axes[0,2]) #Number of people per family size, further divided per ticket class

sns.catplot('Family_size',data=train,hue='Sex',kind='count',ax=axes[1,0]) #Number of people per family size, further divided per sex

sns.catplot('Fare_class',data=train,hue='Pclass',kind='count',ax=axes[1,1]) #Number of people per fare class, further divided per ticket class

sns.catplot('Fare_class',data=train,hue='Sex',kind='count',ax=axes[1,2]) #Number of people per fare class, further divided per sex

sns.catplot(x='Pclass',y='Survived',data=train,kind='bar',ci=None,ax=axes[2,0]) #Number of survived people per ticket class

sns.catplot(x='Sex',y='Survived',data=train,hue='Pclass',kind='bar',ci=None,ax=axes[2,1]) #Number of survived people per sex, further divided per ticket class

sns.catplot(x='Family_size',y='Survived',data=train,hue='Pclass',kind='bar',ci=None,ax=axes[2,2]) #Number of survived people per family size, further divided per ticket class

sns.catplot(x='Family_size',y='Survived',data=train,hue='Sex',kind='bar',ci=None,ax=axes[3,0]) #Number of survived people per family size, further divided per sex

sns.catplot(x='Fare_class',y='Survived',data=train,hue='Pclass',kind='bar',ci=None,ax=axes[3,1]) #Number of survived people per fare class, further divided per ticket class

sns.catplot(x='Fare_class',y='Survived',data=train,hue='Sex',kind='bar',ci=None,ax=axes[3,2]) #Number of survived people per fare class, further divided per sex

for abc in range(2,14,1):

    plt.close(abc)
# Converting sex and embarked to dummy variables

sex_type = {"male":0,"female":1}

train["Sex"] = train["Sex"].map(sex_type)

test["Sex"] = test["Sex"].map(sex_type)

embarked_loc = {"S":0,"C":1,"Q":2}

train["Embarked"] = train["Embarked"].map(embarked_loc)

test["Embarked"] = test["Embarked"].map(embarked_loc)
# Creating input train and test structures

X_train = train.drop(["Survived","Fare","Cabin","Age","PassengerId","Name","Ticket"], axis=1)

Y_train = train["Survived"]

X_test = test.drop(["PassengerId","Name","Age","Ticket","Fare","Cabin"],axis=1)
# Random forest algorigthm with 100 estimations

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
# Can age really improve the random forest classification?

f, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.distplot(train["Age"],hist=True,bins=20,ax=axes[0]) #approximately normal distribution, it can be used to fill the NaNs

axes[0].set_title('Age distribution train set')

ax2 = sns.distplot(test["Age"],hist=True,bins=20,ax=axes[1], color="r") #approximately normal distribution, it can be used to fill the NaNs

axes[1].set_title('Age distribution test set')
# Filling the Age NaNs by sampling from the two normal distributions

meanAge_train, stdAge_train = train["Age"].mean(), train["Age"].std() #mean and standard deviation for the column Age, train set

nullAge_train = train["Age"].isnull() #get the indeces where NaNs are present in the Age column

sampAge_norm_dist_train = np.random.normal(meanAge_train, stdAge_train, nullAge_train.sum())

for ii in range(0,len(sampAge_norm_dist_train)): #checking for eventual age values below zero

    if sampAge_norm_dist_train[ii] < 0:

        sampAge_norm_dist_train[ii] = np.random.normal(meanAge_train, stdAge_train, 1)

train["Age"].loc[nullAge_train] = sampAge_norm_dist_train #fill in the NaNs with the samples values

        



meanAge_test, stdAge_test = test["Age"].mean(), test["Age"].std() #same procedure as for the train set

nullAge_test = test["Age"].isnull()

sampAge_norm_dist_test = np.random.normal(meanAge_test, stdAge_test, nullAge_test.sum())

for jj in range(0,len(sampAge_norm_dist_test)):

    if sampAge_norm_dist_test[jj] < 0:

        sampAge_norm_dist_test[jj] = np.random.normal(meanAge_test, stdAge_test, 1)

test["Age"].loc[nullAge_test] = sampAge_norm_dist_test



# Now both train and test set should have no NaNs in the Age column

print(train.isna().sum()) 

print(test.isna().sum())
# Plotting Age distribution after NaNs filling

f, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.distplot(train["Age"],hist=True,bins=20,ax=axes[0]) #approximately normal distribution, it can be used to fill the NaNs

axes[0].set_title('Age distribution train set after NaNs filling')

ax2 = sns.distplot(test["Age"],hist=True,bins=20,ax=axes[1], color="r") #approximately normal distribution, it can be used to fill the NaNs

axes[1].set_title('Age distribution test set after NaNs filling')
# Binning Age into discrete intervals

age_threshold = [0,18,35,50,65,train["Age"].max()] #should make sense to divide into minors, young adults, adults, old adults, elderly

age_label = [1,2,3,4,5] #new label for the age bins

train["Age_discrete"] = pd.cut(train["Age"], age_threshold, labels=age_label, include_lowest = True) #new column that will be used in the classification

test["Age_discrete"] = pd.cut(test["Age"], age_threshold, labels=age_label, include_lowest = True) #same procedure for the test dataset
# Creating input train and test structures with Age

X_train_w_age = train.drop(["Survived","Fare","Cabin","Age","PassengerId","Name","Ticket"], axis=1)

X_test_w_age = test.drop(["PassengerId","Name","Age","Ticket","Fare","Cabin"],axis=1)



# Random forest algorigthm with 100 estimations, including Age as variable

random_forest_w_age = RandomForestClassifier(n_estimators=100)

random_forest_w_age.fit(X_train_w_age, Y_train)



Y_prediction_w_age = random_forest_w_age.predict(X_test_w_age)



random_forest_w_age.score(X_train_w_age, Y_train)
# Calculating the importance of each variable in both models

features= X_train.columns

importances = random_forest.feature_importances_

indices = np.argsort(importances)

plt.figure(1)

plt.title('Feature Importances without Age')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')
# Same procedure as before, but including the random forest with the Age

features_w_age = X_train_w_age.columns

importances_w_age = random_forest_w_age.feature_importances_

indices_w_age = np.argsort(importances_w_age)

plt.figure(1)

plt.title('Feature Importances with Age')

plt.barh(range(len(indices_w_age)), importances_w_age[indices_w_age], color='b', align='center')

plt.yticks(range(len(indices_w_age)), [features_w_age[i] for i in indices_w_age])

plt.xlabel('Relative Importance')
# Save result to CSV

submission_titanic = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':Y_prediction_w_age})

submission_titanic.to_csv('Titanic_submission_1.csv',index=False)



submission_titanic_without_age = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':Y_prediction})

submission_titanic_without_age.to_csv('Titanic_submission_2.csv',index=False)