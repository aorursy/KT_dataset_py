import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
print(os.listdir("../input"))

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print(train_df.head())
print("-"*70)
print(test_df.head())
print(train_df.columns)
print("-"*70)
print(test_df.columns)
print("-"*70)
print(train_df.info())
print("-"*70)
print(test_df.info())
print("-"*70)
print(train_df.describe())
print("-"*70)
print(test_df.describe())
null_values_in_train = pd.isnull(train_df).sum()
null_values_in_test = pd.isnull(test_df).sum()

plt.subplot(1,2,1)
plt.title("NaN values in training set")
null_values_in_train.plot.bar()
plt.subplot(1,2,2)
plt.title("NaN values in test set")

null_values_in_test.plot.bar()
plt.show()
#First check the Age distribution
plt.subplot(2,1,1)
sns.distplot(train_df.Age.dropna())
plt.subplot(2,1,2)
sns.distplot(test_df.Age.dropna())
plt.show()

#Check the mode & median of distirbution
print("Mode of Age in train_df is {}".format(train_df.Age.dropna().mode()))
print("Median of Age in train_df is {}".format(train_df.Age.dropna().median()))
print("Mode of Age in test_df is {}".format(test_df.Age.dropna().mode()))
print("Median of Age in train_df is {}".format(test_df.Age.dropna().median()))

#fill NaN values with interpolation
train_df["Age"] = train_df.Age.interpolate()
test_df["Age"] = test_df.Age.interpolate()

#Now Check the distribution of Age
plt.subplot(2,1,1)
sns.distplot(train_df.Age, color = "#388E3C")
plt.subplot(2,1,2)
sns.distplot(test_df.Age, color = "#388E3C")
plt.show()
#Let's see how many inputs features we have
print(train_df.columns)
#Pclass -> Feature Engineering

#Training Set

#its a categorical vairble (1, 2, 3)
#change it into categorical variable
train_df.Pclass = train_df.Pclass.astype("category")

#Testing Set
test_df.Pclass = train_df.Pclass.astype("category")
#Name -> Feature Analysis

#Training Set

print(train_df.Name)
#though its text but we can see there are some common prefixes in each name, now we'll extract them.
train_df["Name"] = train_df.Name.str.extract(".((Mrs?)|(Miss)|(Master)).", expand = False)
train_df.info()

#As we can see there are 24 NaN values in 'Name_prefix' column.
#First we'll map these values to int value then we'll fill the NaN values with interpolation.
train_df["Name"] = train_df["Name"].dropna().map({"Mr":0, "Mrs":1, "Miss":2, "Master":3})
print(train_df.Name)
sns.distplot(train_df.Name.dropna())
plt.title("Name prefix distribution")
plt.legend()
plt.show()

train_df.Name.fillna(train_df["Name"].mode()[0], inplace=True)
print(train_df.Name)

#Testing Set
test_df.info()
test_df["Name"] = test_df.Name.str.extract(".((Mrs?)|(Miss)|(Master)).", expand = False)
print(test_df.Name.isnull().sum()) # 7 null values
test_df.Name.fillna(train_df["Name"].mode()[0], inplace=True)
test_df["Name"] = test_df["Name"].dropna().map({"Mr":0, "Mrs":1, "Miss":2, "Master":3})
#Sex -> Feature Analysis

#Training Set

print(train_df.Sex)
#Chnage this into categorical dtype and map it to int values
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
train_df["Sex"] = train_df["Sex"].astype("category")
print(train_df.Sex)

#Testing Set
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].astype("category")
#SibSP & Parch -> Feature Analysis

#Training Set
plt.subplot(2,1,1)
plt.hist(train_df.SibSp)
plt.subplot(2,1,2)
plt.hist(train_df.Parch)
plt.show()

#We can create a new column called "family_member" by combining the these tow columns named "SibSp" & "Parch"
#For that we need to check about NaN values in each column
print(train_df.Parch.isnull().sum()) # Zero NaN values
print(train_df.SibSp.isnull().sum()) #Zero NaN values

train_df["family_member_no"] = train_df["SibSp"] + train_df["Parch"]


#Testing Set
test_df["family_member_no"] = test_df["SibSp"] + test_df["Parch"]
#Ticket -> Feature Analysis

#Training Set

#We need to first extract numeric value of Ticket from string, for that we'll create a function
def numeric_extract(string):
    l = string.split()
    return l[-1]

#create new column named "ticket" by applying that function to training dataset
train_df["ticket"] = train_df["Ticket"].apply(numeric_extract)


#Testing set
test_df["ticket"] = test_df["Ticket"].apply(numeric_extract)
#Fare -> Feature Analysis


#Training Set
print(train_df.Fare.isnull().sum())
#Its all good, so we'll keep as it is.

#Testing Set
print(train_df.Fare.isnull().sum())



#Embarked -> Feature Analysis

#Training Set
print(train_df.Embarked.isnull().sum()) # 2 NaN values
print(train_df.Embarked)

train_df.Embarked.fillna(train_df.Embarked.mode()[0], inplace=True)
print(train_df.Embarked.isnull().sum()) # zero NaN values

#let's change it into category dtype
train_df["Embarked"] = train_df["Embarked"].astype("category")


#Map values to integer
train_df["Embarked"] = train_df["Embarked"].map({"S":0, "C":1, "Q":2})
print(train_df.Embarked)

#Testing Set
print(test_df.Embarked.isnull().sum()) # Zero NaN values
print(test_df.Embarked)

test_df["Embarked"] = test_df["Embarked"].astype("category")
test_df["Embarked"] = test_df["Embarked"].map({"S":0, "C":1, "Q":2})
print(test_df.Embarked)

#Now lets finalize final Feature set

#Training Set
#As we saw the column named "Cabin" have so much NaN values so we can drop it.
train_df = train_df.drop(["Cabin", "Ticket"], axis=1)
train_df


#Testing Set
test_df =  test_df.drop(["Cabin", "Ticket"], axis=1)
test_df

#Now Split it into Input/Output 
X_train = train_df[["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "family_member_no", "ticket"]]
Y_train = train_df["Survived"]

X_test = test_df
#First we create a copy of training set and testing set so that original copy wont affected.
copy_train_df = train_df.copy()
copy_test_df = test_df.copy()
copy_train_df
#Lets see perterson corelation factor b/w all features
cmap = sns.diverging_palette(220, 10, as_cmap=True)
cor = copy_train_df.corr()
sns.heatmap(cor, xticklabels=copy_train_df.columns, yticklabels=copy_train_df.columns, cmap=cmap)
plt.show()

#Now lets visualize them
#Prepare I/p and O/p

copy_train_df = copy_train_df.apply(pd.to_numeric, errors = "coerce")
copy_test_df = copy_test_df.apply(pd.to_numeric, errors = "coerce")

copy_train_df.info()
copy_test_df.info()

copy_train_df.ticket.fillna(train_df["ticket"].mode()[0], inplace=True)
copy_test_df.Name.fillna(train_df["Name"].mode()[0], inplace=True)
copy_test_df.Fare.fillna(test_df["Fare"].mode()[0], inplace=True)


X_train = copy_train_df.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
Y_train = copy_train_df.iloc[:, 1]
X_test = copy_test_df


print("columns in X_train is {} and columns in X_test is {}".format(X_train.shape[1], X_test.shape[1])) #For confirmation whether we're going write or not.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state = 21, stratify = Y_train)
#KNighbourClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_val)
knn_accuracy = accuracy_score(y_val, knn_pred)
print("Score of KNeighbourClassifier is {}".format(knn_accuracy))
#LogisiticRegression

log_clf = LogisticRegression()
log_clf.fit(x_train, y_train)
logi_pred = log_clf.predict(x_val)
logisitic_accouracy = accuracy_score(y_val, logi_pred)
print("Score of Logisitic Regression is {}".format(logisitic_accouracy))
submission_predictions = log_clf.predict(X_test)

submission = pd.DataFrame({
                "PassengerId": X_test["PassengerId"],
                "Survived": submission_predictions
})

submission.to_csv("titanic.csv", index = False)
print(submission.shape)
print("Submission Completed!")