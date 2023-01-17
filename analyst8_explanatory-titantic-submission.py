# have to install any python packages that are missing but all necessary packages are installed

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import os
os.getcwd()
os.chdir('C:\\Users\\sexto\\titaniccontest')
os.getcwd()
# Make code as reproducible as possible which includes where data was downloaded from.
# reading in data with the web link set to a variable is causing problems 

# Instead, manually downloaded titanic data from kaggle website into working directory
# Data downloaded from: https://www.kaggle.com/c/titanic/data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# training dataframe
# size 
sizeTrain = train_df.size
sizeTest = test_df.size
  
# shape 
shapeTrain = train_df.shape 
shapeTest = test_df.shape

# printing size and shape 
print("SizeTrain = {}\nShapeTrain = {}".
format(sizeTrain, shapeTrain))
print("SizeTest = {}\nShapeTest = {}".
format(sizeTest, shapeTest))
train_df.head()
test_df.head()
train_df.tail()
train_df.describe()
train_df.info()
print('_'*40)
test_df.info()
# Combine train and test dataset by adding Survived column to test dataset and with NaN
# Combine them so can clean data and feature engineer
combined_df =  pd.concat(objs=[train_df, test_df], axis=0, sort=False).reset_index(drop=True)
combined_df
# Select all duplicate rows based on one column
# List first and all instance of duplicate names
duplicateRowsName1 = combined_df[combined_df.duplicated(['Name'], keep='last')]
duplicateRowsName2 = combined_df[combined_df.duplicated(['Name'])]
duplicateRowsName3 = pd.concat([duplicateRowsName1, duplicateRowsName2])
sortName = duplicateRowsName3.sort_values(by=['Name'])
sortName
# Are there any null values?
all = len(train_df["Name"])
print ("Total variables for Name are:", all)
null_Name = train_df["Name"].isnull().sum()
print("Missing values for Name are:", null_Name)
a = min(train_df["Name"]), max(train_df["Name"])
print('Min and Max values are:', a)

b = train_df.Name.dtype
print('Data type is:', b)

c= train_df.Name.nunique()
print('Number of unique values is:', c)

# comment out since too many unique
#d= train_df.Name.unique()
#print('Unique values are:', d)
# if too many unique then print first 5 in dataframe instead
e= train_df.filter(like='Name').head(n=5)
print(e)
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train_df["Name"]]
train_df["Title"] = pd.Series(dataset_title)
train_df["Title"].head()
# Count how many of each title there are
sns.set(rc={'figure.figsize':(16,5)})
g = sns.countplot(x="Title",data=train_df)
for p in g.patches:
    g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 
train_df["Title"] = train_df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df["Title"] = train_df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":2, "Mlle":2, "Mrs":2, "Mr":3, "Rare":4})
train_df["Title"] = train_df["Title"].astype(int)
ax = sns.countplot(train_df["Title"], 
                   order = train_df["Title"].value_counts().index)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax = ax.set_xticklabels(["Mr","Miss/Ms", "Mme/Mlle/Mrs","Master","Rare"])
x = sns.catplot(x="Title",y="Survived",data=train_df,kind="bar", height=4, aspect=3.4, order = train_df["Title"].value_counts().index)
x = x.set_xticklabels(["Mr","Miss/Ms", "Mme/Mlle/Mrs","Master","Rare"])
x = x.set_ylabels("survival probability")
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train_df["Name"]]
train_df["Title2"] = pd.Series(dataset_title)
mr_df = train_df[train_df['Title2'].isin(['Don','Rev', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer'])]
sortName2 = mr_df.sort_values(by=['Title2'])
sortName2
mrs_df = train_df[train_df['Title2'].isin(['Lady', 'the Countess'])]
sortName2 = mrs_df.sort_values(by=['Title2'])
sortName2
# Convert to categorical values Title 
train_df["Title2"] = train_df["Title2"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr')
train_df["Title2"] = train_df["Title2"].replace(['Lady', 'the Countess','Countess'], 'Mrs')
train_df["Title2"] = train_df["Title2"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":2, "Mlle":2, "Mrs":2, "Mr":3})
train_df["Title2"] = train_df["Title2"].astype(int)
ax = sns.countplot(train_df["Title2"], 
                   order = train_df["Title2"].value_counts().index)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax = ax.set_xticklabels(["Mr","Miss/Ms", "Mme/Mlle/Mrs","Master"])
g = sns.catplot(x="Title2",y="Survived",data=train_df,kind="bar", order = train_df["Title2"].value_counts().index, height=4, aspect=3.4)
g = g.set_xticklabels(["Mr","Miss/Ms", "Mme/Mlle/Mrs","Master"])
g = g.set_ylabels("survival probability")
# Are there any missing values for Survived and Sex columns?
# Are there any null values?
all = len(train_df["Sex"])
print ("Total variables for Sex are:", all)
null_Survived = train_df["Survived"].isnull().sum()
print("Missing values for Survived are:", null_Survived)
null_Sex = train_df["Sex"].isnull().sum()
print("Missing values for Sex are:", null_Sex)
a = min(train_df["Sex"]), max(train_df["Sex"])
print('Min and Max values are:', a)

b = train_df.Sex.dtype
print('Data type is:', b)

c= train_df.Sex.nunique()
print('Number of unique values is:', c)

d= train_df.Sex.unique()
print('Unique values are:', d)
print("Ensure there are an adequate number of males and females who survived & didn't survive.")
# PassengerId was used because it has no missing values 
sextest1 = train_df[['Sex', 'Survived', 'PassengerId']].groupby(['Sex', 'Survived'], as_index=False).count()
print(sextest1)
print('The lowest number is 81 which is adequate for comparison.')
print('Probability for Survival for males and females:')
sextest = train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(sextest)

sns.set(rc={'figure.figsize':(7,3)})
g = sns.barplot(x="Sex",y="Survived",data=train_df, palette="bwr").set_title("Survival Probabilty by Sex")
plt.ylabel("survival probabilty")
# Are there any null values?
all = len(train_df["Pclass"])
print ("Total variables for Pclass are:", all)

null_Pclass = train_df["Pclass"].isnull().sum()
print("Missing values for Pclass are:", null_Pclass)
a = min(train_df["Pclass"]), max(train_df["Pclass"])
print('Min and Max values are:', a)

b = train_df.Pclass.dtype
print('Data type is:', b)

c= train_df.Pclass.nunique()
print('Number of unique values is:', c)

d= train_df.Pclass.unique()
print('Unique values are:', d)
print("Ensure there are an adequate number of values for Survived in each Pclass to allow for reliable comparison.")
# PassengerId was used because it has no missing values 
Pclasstest2 = train_df[['Pclass', 'Survived', 'PassengerId']].groupby(['Pclass', 'Survived'], as_index=False).count()
print(Pclasstest2)
print('The lowest number is 80 which is adequate for comparison.')
print('Probability for Survival in each class (with 1 being 1st class) is as follows:')
Pclasstest = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(Pclasstest)

# Explore Pclass vs Survived
g = sns.catplot(x="Pclass",y="Survived",data=train_df,kind="bar", height = 6 , 
color = "green")
g.despine(left=True)
g = g.set_ylabels("survival probability")
g = g.fig.suptitle("Survival Probability by Pclass")

# Explore Pclass vs Survived by Sex
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
                   height=6, kind="bar", palette="bwr")
g.despine(left=True)
g = g.set_ylabels("survival probability")
g = g.fig.suptitle("Survival Probability by Pclass & Sex")
# Are there any null values?
all = len(train_df["Embarked"])
print ("Total variables for Embarked are:", all)
null_Embarked = train_df["Embarked"].isnull().sum()
print("Missing values for Embarked are:", null_Embarked)
# string so cant display
#a = min(train_df["Embarked"]), max(train_df["Embarked"])
#print('Min and Max values are:', a)

b = train_df.Embarked.dtype
print('Data type is:', b)

c= train_df.Embarked.nunique()
print('Number of unique values is:', c)

d= train_df.Embarked.unique()
print('Unique values are:', d)
train_df[['Embarked', 'PassengerId']].groupby(['Embarked'], as_index=False).count()
# Fill Embarked nan values with 'S' most frequent value since only 2 values or 0.2%
train_df["Embarked"] = train_df["Embarked"].fillna("S")
g = sns.catplot(x="Embarked", y="Survived",  data=train_df,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_df.head()
# Explore Pclass vs Embarked 
g = sns.catplot("Pclass", col="Embarked",  data=train_df,
                   height=6, kind="count", palette="dark")
g.despine(left=True)
g = g.set_ylabels("Count")
# Are there any null values?
all = len(train_df["Cabin"])
print ("Total variables for Cabin are:", all)
null_Embarked = train_df["Cabin"].isnull().sum()
print("Missing values for Cabin are:", null_Embarked)
# Strings not supported
#a = min(train_df["Cabin"]), max(train_df["Cabin"])
#print('Min and Max values are:', a)

b = train_df.Cabin.dtype
print('Data type is:', b)

c= train_df.Cabin.nunique()
print('Number of unique values is:', c)

d= train_df.Cabin.unique()
print('Unique values are:', d)
# Replace the Cabin number by the type of cabin 'X' if not (in case decide to use X later)
train_df["Cabin2"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train_df['Cabin'] ])
# Delete rows with value of X
cab = train_df[train_df.Cabin2 != 'X']
# chart cabin letter
sns.set(rc={'figure.figsize':(16,5)})
g = sns.countplot(cab["Cabin2"],order=['A','B','C','D','E','F','G','T'])
g = sns.catplot(y="Survived", x="Cabin2", data=cab, kind="bar", order=['A','B','C','D','E','F','G','T'], height=4, aspect=3.4)
g = g.set_ylabels("Survival Probability")
g = g.set_ylabels("Survival Probability")
# Are there any null values?
tick = len(train_df["Ticket"])
print ("Total variables for Ticket are:", tick)
null_Embarked = train_df["Ticket"].isnull().sum()
print("Missing values for Ticket are:", null_Embarked)
a = min(train_df["Ticket"]), max(train_df["Ticket"])
print('Min and Max values are:', a)

b = train_df.Ticket.dtype
print('Data type is:', b)

c= train_df.Ticket.nunique()
print('Number of unique values is:', c)

#d= train_df.Ticket.unique()
#print('Unique values are:', d)
# if too many unique then print first 5 in dataframe instead
e= train_df.filter(like='Ticket').head(n=5)
print(e)
## See if common occurences by extracting the ticket prefix. When there is no prefix it returns X. 
# Replace the Cabin number by the type of cabin 'X' if not
train_df["Ticket"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train_df['Ticket'] ])
# Delete rows with value of X
train_df = train_df[train_df.Ticket != 'X']
# chart ticket letter
sns.set(rc={'figure.figsize':(16,5)})
g = sns.countplot(train_df["Ticket"])
g = sns.catplot(x="Ticket",y="Survived",data=train_df,kind="bar", order = train_df["Ticket"].value_counts().index, height=4, aspect=3.4)
g = g.set_ylabels("survival probability")
# Are there any null values?
fare = len(train_df["Fare"])
print ("Total variables for Fare are:", fare)
fare2 = train_df["Fare"].isnull().sum()
print("Missing values for Fare are:", fare2)
a = min(train_df["Fare"]), max(train_df["Fare"])
print('Min and Max values are:', a)

b = train_df.Fare.dtype
print('Data type is:', b)

c= train_df.Fare.nunique()
print('Number of unique values is:', c)

# comment out since too many unique
#d= train_df.Fare.unique()
#print('Unique values are:', d)
# if too many unique then print first 5 in dataframe instead
e= train_df.filter(like='Fare').head(n=5)
print(e)
# Are there any null values?
par = len(train_df["Parch"])
print ("Total variables for Parch are:", par)
null_Embarked = train_df["Parch"].isnull().sum()
print("Missing values for Parch are:", null_Embarked)
a = min(train_df["Parch"]), max(train_df["Parch"])
print('Min and Max values are:', a)

b = train_df.Parch.dtype
print('Data type is:', b)

c= train_df.Parch.nunique()
print('Number of unique values is:', c)

d= train_df.Parch.unique()
print('Unique values are:', d)
# Are there any null values?
sib = len(train_df["SibSp"])
print ("Total variables for SibSp are:", sib)
null_sib = train_df["SibSp"].isnull().sum()
print("Missing values for SibSp are:", null_sib)
a = min(train_df["SibSp"]), max(train_df["SibSp"])
print('Min and Max values are:', a)

b = train_df.SibSp.dtype
print('Data type is:', b)

c= train_df.SibSp.nunique()
print('Number of unique values is:', c)

d= train_df.SibSp.unique()
print('Unique values are:', d)
# Explore SibSp feature vs Survived
g = sns.catplot(x="SibSp",y="Survived",data=train_df, kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
# Create a family size descriptor from SibSp and Parch
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
g = sns.factorplot(x="Fsize",y="Survived",data = train_df)
g = g.set_ylabels("Survival Probability")
# Are there any null values?
a = len(train_df["Age"])
print ("Total variables for Age are:", a)
null_Embarked = train_df["Age"].isnull().sum()
print("Missing values for Age are:", null_Embarked)
a = min(train_df["Age"]), max(train_df["Age"])
print('Min and Max values are:', a)

b = train_df.Age.dtype
print('Data type is:', b)

c= train_df.Age.nunique()
print('Number of unique values is:', c)

d= train_df.Age.unique()
print('Unique values are:', d)
null_Age = train_df["Age"].isnull().sum()
print("Missing values for Age are:", null_Age)
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
# Looking at Age separted by Pclass, it is significant that under age 50 9n 3rd class had low survival rate. 
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
train_df.head()
# In training dataset, remove rows that are :
# 1. redundant 
# 2. have shown in the Exploratory section are missing too many values & not been able to be cleaned up
# Keep: Sex, Pclass, Title2, Embarked, Parch, FSize
# Remove: Title (similar to Title2), Name (too unique), Cabin (too many values missing), Ticket (too unique),
#         Fare (too unique), SibSp (similar to Fsize), Age (too unique), PassengerId (too unique), Cabin2 (duplicate), Parch (too unique)
X_train = train_df.drop(["Title", "Name", "Cabin", "Ticket", "Fare", "SibSp", "Age", "Survived", "PassengerId", "Cabin2", "Parch"], axis=1)
Y_train = train_df["Survived"]
X_train.shape, Y_train.shape
# Correlation matrix between variables chosen (SibSp Parch Age and Fare values) and Survived to ensure not correlated
fig, ax = plt.subplots(figsize=(7,7))
g = sns.heatmap(train_df[["Sex","Pclass", "Title2","Embarked","Fsize"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm", ax=ax)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!
# Create categorical values for Pclass
X_train["Pclass"] = X_train["Pclass"].astype("category")
X_train = pd.get_dummies(X_train, columns = ["Pclass"],prefix="Pc")
# Create categorical values for Sex
X_train["Sex"] = X_train["Sex"].astype("category")
X_train = pd.get_dummies(X_train, columns = ["Sex"],prefix="Sex")
# Create categorical values for Embarked
X_train["Embarked"] = X_train["Embarked"].astype("category")
X_train = pd.get_dummies(X_train, columns = ["Embarked"],prefix="Embarked")
# Create categorical values for Title2
X_train["Title2"] = X_train["Title2"].astype("category")
X_train = pd.get_dummies(X_train, columns = ["Title2"],prefix="Title")
X_train.head(n=2)
# Prepare test dataset by performing necessary data manipulation on columns and adding columns to ensure matches X_train
# Determine what needs to be done by examining the test_df
X_test = test_df
X_test.head(n=2)
X_test = X_test.drop(["PassengerId", "Age", "Ticket", "Fare", "Cabin"], axis=1)
# Ensure no missing values for Pclass, Sex, Embarked, Name, SibSp, Parch
null_Pclass = X_test["Pclass"].isnull().sum()
print("Missing values for Pclass are:", null_Pclass)

null_Sex = X_test["Sex"].isnull().sum()
print("Missing values for Sex are:", null_Sex)

null_Embarked = X_test["Embarked"].isnull().sum()
print("Missing values for Embarked are:", null_Embarked)

null_Name = X_test["Name"].isnull().sum()
print("Missing values for Title are:", null_Name)

null_SibSp = X_test["SibSp"].isnull().sum()
print("Missing values for SibSp are:", null_SibSp)

null_Parch = X_test["Parch"].isnull().sum()
print("Missing values for Parch are:", null_Parch)
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in test_df["Name"]]
X_test["Title"] = pd.Series(dataset_title)
X_test["Title"].head()
# Ensure no unique title in test_df that were not in train_df
c= X_test.Title.nunique()
print('Number of unique values is:', c)

d= X_test.Title.unique()
print('Unique values are:', d)
# Convert Titles to main categories
# Add Dona to Mrs
# From wiki: Dona may refer to: Feminine form for don (honorific) a Spanish, Portuguese
# Convert to categorical values Title
X_test["Title"] = X_test["Title"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr')
X_test["Title"] = X_test["Title"].replace(['Lady', 'the Countess','Countess', 'Dona'], 'Mrs')
X_test["Title"] = X_test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":2, "Mlle":2, "Mrs":2, "Mr":3})
X_test["Title"] = X_test["Title"].astype(int)
# Create a family size descriptor from SibSp and Parch
X_test["Fsize"] = X_test["SibSp"] + X_test["Parch"] + 1
X_test = X_test.drop(["Name", "SibSp", "Parch"], axis=1)
# Create categorical values for Pclass
X_test["Pclass"] = X_test["Pclass"].astype("category")
X_test = pd.get_dummies(X_test, columns = ["Pclass"],prefix="Pc")
# Create categorical values for Sex
X_test["Sex"] = X_test["Sex"].astype("category")
X_test = pd.get_dummies(X_test, columns = ["Sex"],prefix="Sex")
# Create categorical values for Embarked
X_test["Embarked"] = X_test["Embarked"].astype("category")
X_test = pd.get_dummies(X_test, columns = ["Embarked"],prefix="Embarked")
# Create categorical values for Title
X_test["Title"] = X_test["Title"].astype("category")
X_test = pd.get_dummies(X_test, columns = ["Title"],prefix="Title")
X_test.head(n=2)
# Suppress future warnings from sci-kit learn package
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Random Forest
# set random seed for reproducibility
import random
random.seed(1234)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_predRF = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
# K-nearest neighbor
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_predKNN = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_predLR = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_predSVC = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_predGNB = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_predPer = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_predLSVC = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_predSGD = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_predDT = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_predRF
    })
submission.to_csv('submission.csv', index=False)