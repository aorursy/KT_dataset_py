# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

%matplotlib  inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train =pd.read_csv('../input/train.csv')
test =pd.read_csv('../input/test.csv')
train.sample(50)
print("The types of data our dataset has")
train.dtypes
print('lets see the statistical values of our dataset')
train.describe()
print("let's see the number of non values in our data and their types")
train.info()
missing_values_count = train.isnull().sum()
total_cells = np.product(train.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
print("The percentage of NaN values in descending order")
print((train.isnull().sum().sort_values(ascending=False)/len(train)*100))
print('lets see in this column in our training set')
train.Cabin

train.drop(labels=['Cabin'],axis=1,inplace=True)
test.drop(labels=['Cabin'],axis=1,inplace=True)
print('lets see in the age columns instances')
train.Age
copy = train.copy()
copy.dropna(inplace=True)
sns.distplot(copy.Age)
train.Age.fillna(train.Age.median(),inplace=True)
sns.distplot(train.Age)
test.Age.fillna(test.Age.median(),inplace=True)
print("lets see the values in Embarked column")
train.Embarked
train.Embarked.value_counts()
#seems that S has the more counts and it's less than 1 percent that is missing in this column so we just going to place S in the NaN instances
train.Embarked.fillna("S",inplace=True)
#horaaaay no missing in training set
train.isnull().sum()
print("lets check missing data in the testing dataset to move on to step")
test.isnull().sum()
#only one instance will just put the median
test.Fare.fillna(test.Fare.median(),inplace=True)
train.sample(10)
test.sample(10)
#Most easy column is the Sex column as it just consists of 2 values let' begin with it so that you can have the intuition 
#let's say put the value=1 to represent a male and value 0 to represent a women
train.loc[train['Sex']=='male','Sex'] = 1
train.loc[train['Sex']=='female','Sex'] = 2
# and the same for test data
test.loc[test['Sex']=='male','Sex'] = 1
test.loc[test['Sex']=='female','Sex'] = 2

#let's see what values Embarked has as i forgot :D
train.Embarked.unique()
#aha ok sorry
train.loc[train['Embarked']=='S','Embarked'] = 1
train.loc[train['Embarked']=='C','Embarked'] =2
train.loc[train['Embarked']=='Q','Embarked'] =3
#test data
test.loc[test['Embarked']=='S','Embarked'] =1
test.loc[test['Embarked']=='C','Embarked'] =2
test.loc[test['Embarked']=='Q','Embarked'] =3

#Done let's see what we have done so far
train.sample(10)
test.sample(10)
#just adding the values in Parents and Siblings to get the family size
#and adding 1 counts for the person himself
train['FamSize'] = train['Parch'] + train['SibSp'] + 1
test['FamSize'] = test['Parch'] + test['SibSp'] + 1

#See if famsize is 1 then the person is considered alone
train["IsAlone"] = train.FamSize.apply(lambda x: 1 if x == 1 else 0)
test["IsAlone"] = test.FamSize.apply(lambda x: 1 if x == 1 else 0)
# inspect the correlation between Family and Survived
train[['FamSize', 'Survived']].groupby(['FamSize'], as_index=False).mean()
train.FamSize = train.FamSize.map(lambda x: 0 if x > 4 else x)
train[['FamSize', 'Survived']].groupby(['FamSize'], as_index=False).mean()
#let's see what we have done so far
train.sample(10)
for name in train["Name"]:
    train["Title"] = train["Name"].str.extract("([A-Za-z]+)\.",expand=True)
    
for name in test["Name"]:
    test["Title"] = test["Name"].str.extract("([A-Za-z]+)\.",expand=True)
train.head()
print("Unique values in the Title column")
unique_titles=train.Title.unique()
unique_titles = list(unique_titles)
unique_titles
print("let's see the frequencies of each title in our dataset")
title_list = list(train["Title"])
frequency_titles = []

for i in unique_titles:
    frequency_titles.append(title_list.count(i))
    
print(frequency_titles)
print("integrating both title as a string and its frequency to see which title most occured and which is least")
title_dataframe = pd.DataFrame({
    "Titles" : unique_titles,
    "Frequency" : frequency_titles
})

print(title_dataframe.sort_values(by='Frequency',ascending=False))
#instead of repeating my steps for training and test sets will just put them as a list and iterate through them
#
datasets = [train,test]
for dataset in datasets:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print("unique values after working on them")
test.Title.unique()
print("assigning an easy number to indicate it to each title")
for dataset in datasets:
    dataset.loc[dataset["Title"] == "Miss", "Title"] = 0
    dataset.loc[dataset["Title"] == "Mr", "Title"] = 1
    dataset.loc[dataset["Title"] == "Mrs", "Title"] = 2
    dataset.loc[dataset["Title"] == "Master", "Title"] = 3
    dataset.loc[dataset["Title"] == "Rare", "Title"] = 4

train.Title
test.Title
for dataset in datasets:
    dataset.drop(columns=['Name','Ticket'],axis=1,inplace=True)
train
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#scaler requires arguments to be in a specific format shown below
#convert columns into numpy arrays and reshape them 
for dataset in datasets:
    ages_train = np.array(dataset["Age"]).reshape(-1, 1)
    fares_train = np.array(dataset["Fare"]).reshape(-1, 1)
#we replace the original column with the transformed/scaled values
    dataset["Age"] = scaler.fit_transform(ages_train)
    dataset["Fare"] = scaler.fit_transform(fares_train)

train
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
sns.barplot(x='Sex',y='Survived',data=train)
plt.title("Distribution of Survival based on Gender")
plt.show()

total_survived_females = train[train.Sex == 2]["Survived"].sum()
total_survived_males = train[train.Sex == 1]["Survived"].sum()

print("Total people survived is: " + str((total_survived_females + total_survived_males)))
print("Proportion of Females who survived:") 
print(total_survived_females/(total_survived_females + total_survived_males))
print("Proportion of Males who survived:")
print(total_survived_males/(total_survived_females + total_survived_males))
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")
sns.barplot(x="Age",y='Survived',data=train)
survived_ages = train[train.Survived == 1]["Age"]
not_survived_ages = train[train.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(survived_ages, kde=False)
plt.axis([0, 1, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1, 2, 2)
sns.distplot(not_survived_ages, kde=False)
plt.axis([0, 1, 0, 100])
plt.title("Didn't Survive")
plt.subplots_adjust(right=1.7)
plt.show()
sns.stripplot(x="Survived", y="Age", data=train, jitter=True)
sns.pairplot(train)
X_train = train.drop(labels=["Survived","PassengerId"], axis=1) #define training features set
y_train = train["Survived"] #define training label set
X_test=test.drop('PassengerId',axis=1)
#we don't have y_test, that is what we're trying to predict with our model
from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
from keras.layers import Dense
from keras.models import Sequential
# Initialising the NN
model = Sequential()
#149-20-20-20-20-20-20-20-20-1
# layers
model.add(Dense(units = 149, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN,
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, y_train, batch_size = 32, epochs = 1000,validation_data=(X_valid,y_valid))
submission_predictions = model.predict(X_test)
#checking when a probality is bigger than 0.5 so if so convert the "True" it should spit out to 1 and reshape 
#submission to be like test data
y_final = (submission_predictions > 0.5).astype(int).reshape(X_test.shape[0])

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_final
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)