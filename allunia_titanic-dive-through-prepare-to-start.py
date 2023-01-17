# data analysis tools
import numpy as np 
import pandas as pd 

# data visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Let's have a look at our input files:
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/gender_submission.csv")
train.head()
# Your turn: Peek at the test data ;-)
# Your turn: Peek at the submission data ;-)
train[train.PassengerId == 2].Name.values
train.drop("Name", axis=1, inplace=True)
train.head()
train.drop("Ticket", axis=1, inplace=True)
# Your turn: Drop the name and ticket feature from the test set.
train.info()
sex_map = {"male": 0, "female": 1}
train.Sex = train.Sex.apply(lambda l: sex_map[l])
train.head()
# Apply the sex_map to the test set.
train.Embarked.unique()
train.isnull().sum()
nans_in_train = train.isnull().sum() / train.shape[0] * 100
nans_in_train = nans_in_train[nans_in_train > 0]

plt.figure(figsize=(10,4))
sns.barplot(x=nans_in_train.index.values, y=nans_in_train.values, palette="Set1")
plt.ylabel("Percentage of nans")
plt.ylim([0, 100])
plt.title("Missing values in train")
# Your task: Compute the relative frequencies of missing values in the test set
# Your job: Drop the cabin feature from the train and test set.
train[train.Embarked.isnull()]
original_train = pd.read_csv("../input/train.csv")
original_train[train.Embarked.isnull()]
original_train[(original_train.Cabin == "B28") | (original_train.Ticket == "113572")]
plt.figure(figsize=(10,4))
sns.countplot(train.Embarked)
train.Embarked.fillna("S", inplace=True)
train.Embarked.isnull().value_counts()
# Replace the embarkation object values with numerical ones. 
# You can use a dictionary as we have done for the sex feature. 
test[test.Fare.isnull()]
plt.figure(figsize=(20,5))
train_fares = sns.distplot(train.Fare, kde=False, label="Train", norm_hist=True)
test_fares = sns.distplot(test.Fare.dropna(), kde=False, label="Test", norm_hist=True)
plt.title("Normed histogram of ticket fares")
train_fares.legend()
test_fares.legend()
example = np.array([1,2,3,1,2,1,3,99,1,4,5,2])
print(np.mean(example))
print(np.std(example))
print(np.median(example))
print(train.Fare.median())
print(test.Fare.median())
# Replace the nan value in the test set with the fare median 
fig, ax = plt.subplots(2, 2, figsize=(15,8))
sns.violinplot(x=train.Pclass, y=train.Fare, ax=ax[0,0])
sns.violinplot(x=train.Sex, y=train.Fare, ax=ax[0,1])
sns.violinplot(x=train.Embarked, y=train.Fare, ax=ax[1,0])
ax[1,1].scatter(train.Age.values, train.Fare.values)
ax[1,1].set_xlabel("Age")
ax[1,1].set_ylabel("Fare")
# plot the normalized fare histograms for train and test conditioned on the 3rd ticket class. 
# Then compute some statistics you like and make a new decision for the nan replacement of Mr Storey.
# Replace the our previous fare value with yours.
# Your task: Plot the ages distributions of both test and train set. 
# In addition use data.Age.describe() to obtain some statistical values. 
# Use the median for your replacements.
# Validate that there are no nans left by using data.Age.isnull().sum()
# train.to_csv("your_prepared_train.csv")
# test.to_csv("your_prepared_test.csv")