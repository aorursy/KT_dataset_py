%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv", index_col="PassengerId")
test = pd.read_csv("../input/test.csv", index_col = "PassengerId")
print(train.shape)
print(test.shape)
print("train has {} data | test has {} data".format(train.shape[0], test.shape[0]))
train.head()
train.head()
print("train has {} columns".format(len(train.columns)))
print("test has {} columns".format(len(test.columns)))
print("Target data is Survived")
# check unique data in train
for i in train.columns:
    print("{} has {} unique data".format(i, len(train[i].unique())))
# check unique data in test
for i in test.columns:
    print("{} has {} unique data".format(i, len(test[i].unique())))
#check missing data in train
for i in train.columns:
    print("{0} has {1:.2f}% missing data".format(i, (len(train[train[i].isnull()]) / train.shape[0]) *100)) 
cat_data = ["Pclass","Sex","Ticket","Cabin","Emabarked"]
num_data = ["Age","Fare","SibSp","Parch"]
cat_1 = [14,15,16,17,18,24,25,26,27,30,31,32,33,34,35,36,37]
figure, axe = plt.subplots(nrows =1,ncols =1)
figure.set_size_inches(20,4)
sns.countplot(train["Age"])
#replace NoN data with Mode data
train.loc[train["Age"].isnull(),"Age"] = train["Age"].mode()[0]
test.loc[test["Age"].isnull(),"Age"] = test["Age"].mode()[0]
train.loc[train["Age"].isnull(),"Age"]
#Check deviation of Age
sns.distplot(train["Age"])
train["Age"].describe()

#Too many missing data, it means Cabin data would not be good feature to predict Target label
len(train[train["Cabin"].isnull()])

# replace NaN data with mdoe data
train.loc[train["Embarked"].isnull(),"Embarked"]  = train["Embarked"].mode()[0]
#replace NaN data with mean 
test.loc[test["Fare"].isnull(),"Fare"] = test["Fare"].mean()
#train
le= LabelEncoder()
le.fit(train["Embarked"])
Embarked = le.transform(train["Embarked"])
# One hot encoding
Embarked= np.eye(3)[Embarked]
Embarked = pd.DataFrame(Embarked,columns =["Embarked_C","Embarked_Q","Embarked_S"])
train.reset_index(inplace=True)
train = pd.concat([train,Embarked], axis =1)
train.set_index("PassengerId",inplace=True)
train.head()
#test
le= LabelEncoder()
le.fit(test["Embarked"])
Embarked = le.transform(test["Embarked"])
# One hot encoding
Embarked= np.eye(3)[Embarked]
Embarked = pd.DataFrame(Embarked,columns =["Embarked_C","Embarked_Q","Embarked_S"])
test.reset_index(inplace=True)
test = pd.concat([test,Embarked], axis =1)
test.set_index("PassengerId",inplace=True)
#Encoding sex data in train
le.fit(train["Sex"])
sex = le.transform(train["Sex"])
train["Sex"] = sex.reshape(-1,1)
#Encoding sex data in test
le.fit(test["Sex"])
sex = le.transform(test["Sex"])
test["Sex"] = sex.reshape(-1,1)
train.corr()
# 1 : male, 0: femle
sex_corr = train[["Sex","Survived"]]
grouped = sex_corr.groupby("Sex")["Survived"].aggregate({"sum_of_survior":"sum"})
grouped["count_of_sex"] = sex_corr.groupby("Sex")["Survived"].aggregate({"count_of_sex":"count"})
grouped["s_rate"] = grouped["sum_of_survior"] / grouped["count_of_sex"]
grouped
figure, (axe1,axe2) = plt.subplots(nrows = 1, ncols =2)
figure.set_size_inches(14,4)
sns.barplot(grouped.index,grouped["sum_of_survior"],ax = axe1)
sns.barplot(grouped.index,grouped["s_rate"],ax = axe2)
pclass = train[["Pclass","Survived"]]
grouped = pclass.groupby("Pclass")["Survived"].aggregate({"sum_of_survivor":"sum"})
grouped["count_of_class"] = pclass.groupby("Pclass")["Survived"].aggregate({"count_of_class":"count"})
grouped["s_rate"] = grouped["sum_of_survivor"] / grouped["count_of_class"]
grouped
figure, (axe1,axe2) = plt.subplots(nrows = 1, ncols =2)
figure.set_size_inches(14,4)
sns.barplot(grouped.index,grouped["sum_of_survivor"],ax = axe1)
sns.barplot(grouped.index,grouped["s_rate"],ax = axe2)
age = train.groupby("Age")["Survived"].aggregate({"sum_of_survivor":"sum"})
age["count_of_age"] = train.groupby("Age").size().values
age["s_rate"] = age["sum_of_survivor"] / age["count_of_age"]
age
figure, (axe1,axe2) = plt.subplots(nrows = 1, ncols =2)
figure.set_size_inches(14,4)
sns.pointplot(age.index,age["sum_of_survivor"],ax = axe1)
sns.pointplot(age.index,age["s_rate"],ax = axe2)
train.head()
sns.barplot(train["SibSp"],train["Survived"])
sns.barplot(train["Parch"],train["Survived"])
sns.barplot(train["Embarked"],train["Survived"])
train["Family_size"] = train["Parch"] + train["SibSp"]
test["Family_size"] = test["Parch"] + test["SibSp"]
sns.pointplot(train["Family_size"],train["Survived"])
feature_names = ["Pclass", "Sex", "Fare","Family_size",
                 "Embarked_C", "Embarked_S", "Embarked_Q"]
feature_names
x_train = train[feature_names]

print(x_train.shape)
x_train.head()
x_test = test[feature_names]

print(x_test.shape)
x_test.head()
label_name = "Survived"
y_train = train[label_name]

print(y_train.shape)
y_train.head()
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=71,max_leaf_nodes=6,min_impurity_decrease=0.002818)
model
model.fit(x_train, y_train)
y_predict = cross_val_predict(model,x_train,y_train)
accuracy = accuracy_score(y_predict,y_train,)
print("accuracy = {0:.2f}".format(accuracy))
predictions = model.predict(x_test)
submit = pd.read_csv("../input/gender_submission.csv", index_col="PassengerId")

print(submit.shape)
submit.head()
submit["Survived"] = predictions

print(submit.shape)
submit.head()
submit.to_csv("submit.csv")
