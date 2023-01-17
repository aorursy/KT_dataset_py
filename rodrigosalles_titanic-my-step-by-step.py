import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
train.head() # Checking the data
print("Train:",train.shape, "\n \n",train.describe(),"\n \n")

print("Test",test.shape, "\n \n", test.describe())



# As we can see, the training set has 891 lines with passenger information. 

# Some columns have a smaller numberm, which indicates missing data. 

# The training set does not have the surviving colunm

# This way the data needs to be cleaned before starting the classifier project
sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

plt.show()

# It can be observed that more women survived
fig = plt.figure(figsize=(18,6))



plt.subplot2grid((2,3),(0,0))

train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.9)

plt.title("total survivors(%)")



plt.subplot2grid((2,3),(0,1))

plt.scatter(train.Survived,train.Age, alpha=0.1)

plt.title("Age vs Survived(years)")



plt.subplot2grid((2,3),(0,2))

train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.9)

plt.title("survivors by class of tickets")



plt.subplot2grid((2,3),(1,0), colspan=2)

for x in [1,2,3]:

    train.Age[train.Pclass == x].plot(kind = "kde")

plt.title("Tickets Class vs Age")

plt.legend(("1st","2nd","3rd"))



plt.show()

# As you can see, some information is more important. Others not so much.
# Training set

plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(),cbar=False, yticklabels=False, cmap='viridis')

plt.show()

# The values in yellow represent missing data. You may notice that the "Age", 

# "Cabin", and "Embarked" columns are incomplete.
# Training set

plt.figure(figsize=(10,6))

sns.heatmap(test.isnull(),cbar=False, yticklabels=False, cmap='viridis')

plt.show()

# The values in yellow represent missing data. You may notice that the "Age", 

# "Cabin", and "Fare" columns are incomplete.
# Age: complete with averages of ages per class

# Average age per class



plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass', y='Age', data=train)

plt.show()
# getting the averages

m1=round(np.mean(train["Age"][train["Pclass"]==1]))

m2=round(np.mean(train["Age"][train["Pclass"]==2]))

m3=round(np.mean(train["Age"][train["Pclass"]==3]))

print(" Average age of 1st class: ",m1, " \n Average age of 2nd class: ",m2, "\n Average age of 3rd class: ",m3 )
# replacing the missing values by the averages per class

def age_update(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return m1

        elif Pclass == 2:

            return m2

        else: return m3

    else: return Age
# applying the age_update function

train['Age']=train[['Age','Pclass']].apply(age_update, axis=1)
# Cabim: delete column

train.drop('Cabin', axis=1, inplace=True)
# Embarked: only a few values are missing, we'll eliminate the lines

train.dropna(inplace=True)
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(),cbar=False, yticklabels=False, cmap='viridis')

plt.show()



# As we can see, there are no missing data now
# Age: complete with averages of ages per class

# Average age per class



plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass', y='Age', data=test)

plt.show()
# getting the averages

m4=round(np.mean(test["Age"][test["Pclass"]==1]))

m5=round(np.mean(test["Age"][test["Pclass"]==2]))

m6=round(np.mean(test["Age"][test["Pclass"]==3]))

print(" Average age of 1st class: ",m4, " \n Average age of 2nd class: ",m5, "\n Average age of 3rd class: ",m6 )
# replacing the missing values by the averages per class

def age_update(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return m4

        elif Pclass == 2:

            return m5

        else: return m6

    else: return Age
# applying the age_update function

test['Age']=test[['Age','Pclass']].apply(age_update, axis=1)
# Cabim: delete column

test.drop('Cabin', axis=1, inplace=True)
# getting the  fare averages per class

m7=round(np.mean(test["Fare"][test["Pclass"]==1]))

m8=round(np.mean(test["Fare"][test["Pclass"]==2]))

m9=round(np.mean(test["Fare"][test["Pclass"]==3]))

print(" Average Fare of 1st class: ",m7, " \n Average Fare of 2nd class: ",m8, "\n Average Fare of 3rd class: ",m9 )
def fare_update(cols):

    Fare = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Fare):

        if Pclass == 1:

            return m7

        elif Pclass == 2:

            return m8

        else: return m9

    else: return Fare
# applying the age_update function

test['Fare']=test[['Fare','Pclass']].apply(fare_update, axis=1)
plt.figure(figsize=(10,6))

sns.heatmap(test.isnull(),cbar=False, yticklabels=False, cmap='viridis')

plt.show()

# As we can see, there are no missing data now
# The data is clean, but you can observe the presence of strings, 

# which will not be processed by the algorithm. We need to code them.

train.info()
# The data is clean, but you can observe the presence of strings, 

# which will not be processed by the algorithm. We need to code them.

test.info()
# The function will create a new column encoding the sex column

train['Male'] = pd.get_dummies(train['Sex'], drop_first=True)

test['Male'] = pd.get_dummies(test['Sex'], drop_first=True)
# Coding the column Embarked

embarked_train = pd.get_dummies(train['Embarked'])

train = pd.concat([train, embarked_train], axis=1)

embarked_test = pd.get_dummies(test['Embarked'])

test = pd.concat([test, embarked_test], axis=1)
train.head()
test.shape

# only the Survived column is missing. Ok.
# Discarding some irrelevant columns

train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)

test.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
train.head()
test.head()
# Let's train the classifier with all the columns



columns = ['Pclass', 'Age','SibSp','Parch','Fare', 'Male','C', 'Q','S']

lr = LogisticRegression()

lr.fit(train[columns], train["Survived"])
# To make a prior assessment, we will divide our training set into two parts: 

# 80% for training and 20% for testing.

all_X = train[columns]

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.20,random_state=0)
# Firts test

lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)



print(accuracy)
# To get a better idea of classifier performance, we can use the the "K-Folder" cross validation method. 

# K-Folder is used to train and test our model on different splits of our data, and then average the accuracy scores.

lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

scores.sort()

accuracy = scores.mean()



print(scores)

print(accuracy)
# Now let's prepare our classifier to make predictions on test set data and submit to competition

lr = LogisticRegression()

lr.fit(all_X,all_y)

test_predictions = lr.predict(test[columns])
# We must create a submissiom file with exactly 2 columns: PassengerId (sorted in any order)

# Survived (contains the binary predictions: 1 for survived, 0 for deceased)



# The csv file will be in the same directory as the python file.  All you need to do is 

# upload the file the file on the competition page

test_ids = test["PassengerId"]

submission_df = {"PassengerId": test_ids,

                 "Survived": test_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv("submission.csv",index=False)