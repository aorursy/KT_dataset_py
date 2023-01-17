import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
dataframes = [train, test]
test_id = test["PassengerId"]
print(train.shape, test.shape)
train.describe().transpose()
test.describe().transpose()
print(train.info(), test.info())
#Checking for null values

print(train.isnull().sum())
print("-----------------------------")
print(test.isnull().sum())
#Checking correlation between variables

sns.catplot(x ="Sex", hue ="Survived", kind ="count", data = train)
# We can see that females survived. We will check other variables are related

sns.catplot(x ="Pclass", hue ="Survived",kind ="count", data = train)
# Age also plays a major role in survival - we will check the age range for survival

sns.violinplot(x ="Sex", y ="Age", hue ="Survived", data = train, split = True)
#We will check the relation between the family members and survival rate
#the total of parents, children and siblings is family
#Adding total family members

for data in dataframes:
    data['Fam_mem'] = data['SibSp'] + data['Parch']

    # Creating a variable alone to check if they have any family or not

    data['Alone'] = data['Fam_mem'].map(lambda x: 1 if int(x) == 0 else 0)

train.head()
sns.catplot(x ="Fam_mem", hue ="Survived",kind ="count", data = train)
sns.factorplot(x ='Alone', y ='Survived', data = train) 
# The fare paid can also be an important factor in survival as first class passengers had more survival rate
# We have to make the number of fares into 3 broader groups as the classes

for data in dataframes:
    data['Fare_groups'] = pd.qcut(data['Fare'], 3) 
    
train.head()
sns.catplot(x ="Fare_groups", hue ="Survived",kind ="count", data = train)
#We can check how the embarked place is related with survival

sns.catplot(x ="Embarked", hue ="Survived",kind ="count", data = train)
#We can see that there are missing values for age, and fare has one missing value.
#We have to check how age is distributed

sns.distplot(train['Age'])
plt.show()
#Age is normally distributed - we can use mean or median

for data in dataframes:
    data['Age'] = data['Age'].fillna(data['Age'].median())

#Only one value missing for Fare and fare groups

    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Fare_groups'] = data['Fare_groups'].fillna(data['Fare_groups'].mode()[0])

print(train.info())
print("--------------------------------")
print(test.info())
#Embarked has two missing values. Filling it with mode

for data in dataframes:
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

print(train.info())
print("--------------------------------")
print(test.info())
#Since there are 3 categories in Embarked we can transform it to numerical values

from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()

for data in dataframes:
    data['Embarked_New'] = lab_enc.fit_transform(data['Embarked'])

train.head()
#Checking the names there are many titles, so we will split it 

for data in dataframes:
    data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]

train.head()
print(train['Title'].value_counts())
print("________________________________")
print(test['Title'].value_counts())
#We will now group all the titles into major categories

for data in dataframes:
    data['Mr'] = data['Title'].map(lambda x: 1 if str(x) == 'Mr' else 0)
    data['Miss'] = data['Title'].map(lambda x: 1 if str(x) in ['Miss', 'Mlle', 'Ms'] else 0)
    data['Mrs'] = data['Title'].map(lambda x: 1 if str(x) in ['Mrs', 'Mme'] else 0)
    data['Master'] = data['Title'].map(lambda x: 1 if str(x) == 'Master' else 0)
    data['Officer'] = data['Title'].map(lambda x: 1 if str(x) in ['Dr', 'Major', 'Rev', 'Col', 'Capt'] else 0)
    data['Royalty'] = data['Title'].map(lambda x: 1 if str(x) not in ['Mr', 'Miss', 'Mlle', 
                                                                      'Mrs', 'Ms', 'Mme', 
                                                                      'Master', 'Dr', 'Major', 'Rev', 'Col', 'Capt'] else 0)

train.head()
# Changing Male and female to binary

for data in dataframes:
    data['Sex'] = data['Sex'].map(lambda x: 0 if str(x) == 'male' else 1)
    
train.head()
#We have to standardise age and fare

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

for data in dataframes:
    data[['Age', 'Fare']] = ss.fit_transform(data[['Age', 'Fare']])

train.head()
print(train.info())
print("--------------------------------")
print(test.info())
for data in dataframes:
    data = data.drop(['Ticket', 'Title', 'Name', 'Cabin', 'Embarked', 'Fare_groups', 'PassengerId'], axis = 1, inplace = True)

print(train.info())
print("--------------------------------")
print(test.info())
#now to check how all variables are related

plt.figure(figsize = (14,10))
sns.heatmap(train.corr(), annot = True)
plt.show()
#Splitting data into dependent and independent variables

x_train = train.drop('Survived', axis = 1)
y_train = train.Survived

print(x_train.shape)
print(y_train.shape)
x_test = test.copy()
print(x_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc_classifier = classifier.score(x_train, y_train)*100
acc_classifier
#cross validation

from sklearn.model_selection import cross_val_score
print(cross_val_score(classifier, x_train, y_train, cv=5))
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

y_pred = dec_tree.predict(x_test)

acc_dec_tree = dec_tree.score(x_train, y_train)*100
acc_dec_tree
#cross validation
print(cross_val_score(dec_tree, x_train, y_train, cv=5))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 3, max_features = 0.5, n_jobs = -1)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

acc_rfc = rfc.score(x_train, y_train)*100
acc_rfc
#cross validation
print(cross_val_score(rfc, x_train, y_train, cv=5))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = knn.score(x_train, y_train)*100
acc_knn
#cross validation
print(cross_val_score(knn, x_train, y_train, cv=5))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

acc_gnb = gnb.score(x_train, y_train)*100
acc_gnb
#cross validation
print(cross_val_score(gnb, x_train, y_train, cv=5))
models = pd.DataFrame({
'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes'],
'Score': [acc_classifier, acc_dec_tree, acc_rfc, acc_knn, acc_gnb]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({"PassengerId": test_id, "Survived": y_pred})
submission.to_csv('D:\\Naveen\\Data Science\\Nuclei - Online\\Projects\\Internship Project 2\\submission.csv', index=False)
