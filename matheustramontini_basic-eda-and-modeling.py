import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
# Loading data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
display(train.head(5))



print("Train shape: ", train.shape)
display(test.head(5))



print("Test shape: ", test.shape)
# Checking null values

train.isnull().sum()
plt.pie(train["Survived"].value_counts(),explode=[0, 0.02],autopct='%1.1f%%', labels=train["Survived"].value_counts().index)

plt.title("Survival Rate")

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,5))



sns.countplot('Sex',hue='Survived',data=train, ax=ax[0])

ax[0].set_title("Male/Female survival plot")



survivors = train.query("Survived == 1")

survivors["Sex"].value_counts().plot.pie(explode=[0, 0.02],autopct='%1.1f%%', 

        labels=survivors["Sex"].value_counts().index, ax=ax[1])

ax[1].set_title("Male/Female Survival Rate")

plt.show()
# Removing the target feature from the remaining dataset

new_train = train.drop("Survived", axis=1)

join_df = pd.concat([new_train, test])
# Let's have a general view of our feature Age

np.unique(join_df["Age"])
# Replacing the nulls

male_mean = train.query("Sex == 'male' and Survived==1")["Age"].mean()

female_mean = train.query("Sex == 'female' and Survived==1")["Age"].mean()



join_df.loc[(join_df.Age.isnull())&(join_df.Sex=='female'),'Age']=female_mean

join_df.loc[(join_df.Age.isnull())&(join_df.Sex=='male'),'Age']=male_mean



print("Total null: {}".format(join_df.Age.isnull().sum()))



# Rounding the ages

join_df["Age"] = join_df["Age"].map(lambda age: int(age))
survivor_list = train.query("Survived == 1")



print("Survivors mean age is {:.0f}".format(survivor_list["Age"].mean()))

print("Survivors mean age for males is {:.0f}".format(survivor_list.query("Sex == 'male'")["Age"].mean()))

print("Survivors mean age for females is {:.0f}".format(survivor_list.query("Sex == 'female'")["Age"].mean()))

print("Minimal Survivor Age is {:.0f}".format(min(survivor_list["Age"])))

print("Maximum Survivor Age is {:.0f}".format(max(survivor_list["Age"])))



plt.figure(figsize=(25,6))

sns.barplot(train['Age'],train['Survived'], ci=None)

plt.xticks(rotation=90);

plt.show()
# Let's see our classes

np.unique(join_df["Pclass"])
train["Pclass"].value_counts()
class_count_dict = dict(train["Pclass"].value_counts().sort_index())



for k,v in class_count_dict.items():

    print("People from the {} class: {}".format(k, v))
f,ax=plt.subplots(3,2,figsize=(15,15))



train["Pclass"].value_counts().plot.pie(explode=[0, 0.02, 0.02],autopct='%1.1f%%', 

        labels=survivors["Pclass"].value_counts().index, ax=ax[0][0])

ax[0][0].set_title("Class Survival Proportion")





sns.countplot(train["Pclass"], ax=ax[0][1])

ax[0][1].set_title("Count passengers count")



sns.countplot('Pclass',hue='Survived',data=train, ax=ax[1][0])

ax[1][0].set_title("General Survivors per Class")



sns.countplot('Pclass',hue='Sex',data=train, ax=ax[1][1])

ax[1][1].set_title("General Class per Sex")



sns.countplot('Pclass',hue='Sex',data=train.query("Survived == 1"), ax=ax[2][0])

ax[2][0].set_title("Survivors Class per Sex")



sns.barplot(x='Pclass',y='Survived',data=train, ax=ax[2][1])

ax[2][1].set_title("Survivors Rate per Class")
# Check unique embarked places

print("Unique places: ", train.Embarked.unique())



# First of all, as we have only a few null values, lets fill with the place that had more embarks

f,ax=plt.subplots(1,1,figsize=(6,5))



train["Embarked"].value_counts().plot.pie(explode=[0, 0.02, 0.02],autopct='%1.1f%%', 

                                              labels=train["Embarked"].value_counts().index, ax=ax)
print("Null values: ", train.Embarked.isnull().sum())

# Treating missing

train['Embarked'].fillna('S',inplace=True)

join_df['Embarked'].fillna('S',inplace=True)

print("Null values after cleaning: ", train.Embarked.isnull().sum())
f,ax=plt.subplots(3,2,figsize=(15,15))



sns.countplot(train["Embarked"], ax=ax[0][0])

ax[0][0].set_title("Quantity of people that embarked in place")



sns.countplot('Embarked',hue='Survived',data=train, ax=ax[0][1])

ax[0][1].set_title("Survived quantity by place of embark")



sns.countplot('Embarked',hue='Pclass',data=train, ax=ax[1][0])

ax[1][0].set_title("Quantity of class embarked by place")



sns.countplot('Embarked',hue='Sex',data=train, ax=ax[1][1])

ax[1][1].set_title("Sex by place")



sns.countplot('Embarked',hue='Sex',data=train, ax=ax[1][1])

ax[1][1].set_title("Sex by place")



sns.barplot(x='Embarked',y='Survived',data=train, ax=ax[2][0])

ax[2][0].set_title("Embarked vs Survived rate")



train["Embarked"].value_counts().plot.pie(explode=[0, 0.02, 0.02],autopct='%1.1f%%', 

                                              labels=train["Embarked"].value_counts().index, ax=ax[2][1])

ax[2][1].set_title("Embarked place proportion")
# Creating the feature family_size

train["family_size"] = train["SibSp"] + train["Parch"]

train.head(5)



# Replicating to joined data frame

join_df["family_size"] = join_df["SibSp"] + join_df["Parch"]
f,ax=plt.subplots(3,2,figsize=(15,15))



sns.countplot('SibSp',hue='Survived',data=train, ax=ax[0][0])

ax[0][0].set_title("Survived by siblings/spouses quantity")

sns.barplot(x='SibSp',y='Survived',data=train, ax=ax[0][1])

ax[0][1].set_title("Survived by siblings/spouses rate")



sns.countplot('Parch',hue='Survived',data=train, ax=ax[1][0])

ax[1][0].set_title("Survived by parents/children quantity")

sns.barplot(x='Parch',y='Survived',data=train, ax=ax[1][1])

ax[1][1].set_title("Survived by parents/children rate")



sns.countplot('family_size',hue='Survived',data=train, ax=ax[2][0])

ax[2][0].set_title("Survived by family size")

sns.barplot(x='family_size',y='Survived',data=train, ax=ax[2][1])

ax[2][1].set_title("Survived by family size rate")



plt.subplots_adjust(wspace=0.2,hspace=0.5)

# Getting the correlation between features

sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 

# Get current figure

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()

# Converting Sex into numerical values

train["Sex"].replace(["male", "female"], [0, 1], inplace=True)

# Converting the embark place into numerics labels

train["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)
# Replicating to joined data frame

join_df["Sex"].replace(["male", "female"], [0, 1], inplace=True)



# Replicating to joined data frame

join_df["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)


# Transforming age by range

# Using as base to our range of ages

plt.figure(figsize=(25,6))

sns.barplot(train['Age'],train['Survived'], ci=None)

plt.xticks(rotation=90);

plt.show()



# Creating new field

train["Age_Range"] = 0

train.loc[train["Age"]<=15, "Age_Range"] = 0

train.loc[(train["Age"]>15)&(train["Age"]<=35), "Age_Range"] = 1

train.loc[(train["Age"]>35)&(train["Age"]<=55), "Age_Range"] = 2

train.loc[train["Age"]>55, "Age_Range"] = 3
# Creating new field

join_df["Age_Range"] = 0

join_df.loc[join_df["Age"]<=15, "Age_Range"] = 0

join_df.loc[(join_df["Age"]>15)&(join_df["Age"]<=35), "Age_Range"] = 1

join_df.loc[(join_df["Age"]>35)&(join_df["Age"]<=55), "Age_Range"] = 2

join_df.loc[join_df["Age"]>55, "Age_Range"] = 3
# Dropping features

train.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Parch", "SibSp", "Age"], inplace=True)

join_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Parch", "SibSp", "Age"], inplace=True)
# Changing the columns names

train.columns = ["survived", "p_class", "sex", "embarked", "family_size", "age_range"]

join_df.columns = ["p_class", "sex", "embarked", "family_size", "age_range"]
# Getting the correlation between features

sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 



# Get current figure

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.naive_bayes import GaussianNB 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split 

from sklearn import metrics

import xgboost as xgb

# Splitting the data frames again

print("Test Shape: {}".format(test.shape))

print("Train Shape: {}".format(train.shape))

print("Merged Shape: {}".format(join_df.shape))



test_shape = test.shape[0]

train_shape = train.shape[0]
# Target feature

y = train["survived"]



# Removing the target feature from the remaining dataset

X = join_df[:train_shape]

test = join_df[train_shape:]



# Splitting in test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



print("X_train shape: ", X_train.shape)

print("X_test shape: ", X_test.shape)

print("y_train shape: ", y_train.shape)

print("y_test shape: ", y_test.shape)
# liblinear because its a small dataset

model = LogisticRegression(solver='liblinear')

model.fit(X_train, y_train)



linear_regression_prediction = model.predict(X_test)



print('Logistic regression accuracy: ',metrics.accuracy_score(linear_regression_prediction, y_test))
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)



random_forest_prediction = model.predict(X_test)



print('Random forest accuracy: ', metrics.accuracy_score(random_forest_prediction, y_test))
model = DecisionTreeClassifier()

model.fit(X_train, y_train)



decision_tree_prediction = model.predict(X_test)



print('Decision tree accuracy: ', metrics.accuracy_score(decision_tree_prediction, y_test))
model = GaussianNB()

model.fit(X_train, y_train)



gaussian_prediction = model.predict(X_test)



print('Naive Bayes accuracy: ', metrics.accuracy_score(gaussian_prediction, y_test))
model = xgb.XGBClassifier(n_estimators=100,

                          n_jobs=4,

                          learning_rate=0.03,

                          subsample=0.8,

                          colsample_bytree=0.8)



model.fit(X_train, y_train)

xgb_prediction = model.predict(X_test)



print('XGB prediction: ', metrics.accuracy_score(xgb_prediction, y_test))
test_passenger_id = pd.read_csv('../input/gender_submission.csv')['PassengerId']

# We will use the XGB prediction for the submition

xgb_prediction = model.predict(test)

submission = pd.concat([pd.DataFrame(test_passenger_id), pd.DataFrame({'Survived':xgb_prediction})], axis=1)

submission.to_csv('predictions.csv', index=False)