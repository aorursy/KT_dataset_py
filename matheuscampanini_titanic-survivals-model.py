# Importing essential libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Loading datasets

df_test = pd.read_csv("../input/titanic/test.csv")

df_train = pd.read_csv("../input/titanic/train.csv")
# Checking train dataset first 5 rows

df_train.head()
# Checking test dataset first 5 rows

df_test.head()
# Datasets size

print('Train Dataset number of rows:', df_train.shape[0],'\nTrain Dataset number of columns:', df_train.shape[1], '\n')

print('Test Dataset number of rows:', df_test.shape[0],'\nTest Dataset number of columns:', df_test.shape[1], '\n')
# Dataset description

df_train.describe()
# Train dataset histogram

df_train.hist(figsize=(12,12));
# Checking for Missing Values

df_train.isnull().sum()
df_train.info()
# Checking for the most common town

pd.value_counts(df_train['Embarked'])
# Fill empty and NaNs values with NaN

df_train["Embarked"] = df_train["Embarked"].fillna("S")

age_median = df_train['Age'].median()

df_train['Age'] = df_train['Age'].fillna(age_median)
# Checking if there are still any missing values

df_train.isnull().sum()
# Heat map of the dataset features

g = sns.heatmap(df_train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# Survival percentage by sex

df_train[['Sex', 'Survived']].groupby(['Sex']).mean()
# Survivals vs Sex/Pclass/Embarked

fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(12,4))

sns.barplot(x='Sex', y='Survived', data=df_train, ax=axis1)

sns.barplot(x='Pclass', y='Survived', data=df_train, ax=axis2)

sns.barplot(x='Embarked', y='Survived', data=df_train, ax=axis3);
# Survivals vs Age

age_survived = sns.FacetGrid(df_train, col='Survived')

age_survived.map(sns.distplot, 'Age');
# Creating dummy variables dataset for the Sex feature

df_sex_dummies = pd.get_dummies(df_train.Sex)

df_sex_dummies.head()
# Adding the Sex dummy variables dataset to the train dataset

df_train[['female', 'male']] = df_sex_dummies[['female', 'male']]

df_train.head()
# Creating dummy variables dataset for the Pclass feature

df_Pclass_get_dummies = pd.get_dummies(df_train['Pclass'])

df_Pclass_get_dummies.head()
# Adding the Pclass dummy variables dataset to the train dataset

df_train[['1_Class', '2_Class', '3_Class']] = df_Pclass_get_dummies[[1,2,3]]

df_train.head()
# Taking the Title of each passenger from the Name feature

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_train["Name"]]

df_train["Title"] = pd.Series(dataset_title)

df_train.head()
# Analising the most commom titles

g = sns.countplot(x="Title",data=df_train)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# Replacing the least commom titles by "Rare"

df_train["Title"] = df_train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Mme'], 'Rare')
# Creating dummy variables dataset for the Title feature

df_title_get_dummies = pd.get_dummies(df_train['Title'])

df_title_get_dummies.head()
# Adding the Title dummy variables dataset to the train dataset

df_train[['Master', 'Miss', 'Mr', 'Mrs', 'Ms', 'Rare']] = df_title_get_dummies[['Master', 'Miss', 'Mr', 'Mrs', 'Ms', 'Rare']]

df_train.head()
# Creating dummy variables dataset for the Embarked feature

df_embarked_get_dummies = pd.get_dummies(df_train['Embarked'])

df_embarked_get_dummies.head()
# Adding the Embarked dummy variables dataset to the train dataset

df_train[['Cherbourg', 'Queenstown', 'Southampton']] = df_embarked_get_dummies[['C', 'Q', 'S']]
# Taking off the columns data will not be used in the model

df_train = df_train.drop(['Pclass','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis = 1)

df_train.head()
# Last check for missing values

df_train.isnull().sum()
# Defining the Dependent Variables

X_train = df_train.iloc[:,2:]

df_X_train = pd.DataFrame(X_train)

df_X_train.head()
# Defining the Independent Variable

y_train = df_train.iloc[:,1]

df_y_train = pd.DataFrame(y_train)

df_y_train.head()
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

# Importing Libraries for the model

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



kfold = StratifiedKFold(n_splits=10)



# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy",cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
# Test dataser preprocessing

# Fill empty and NaNs values with NaN

df_test["Embarked"] = df_test["Embarked"].fillna("S")

age_fare_median = df_test[['Age', 'Fare']].median()

df_test[['Age', 'Fare']] = df_test[['Age', 'Fare']].fillna(age_fare_median)
# Checking for missing values in the test dataset

df_test.isnull().sum()
# Creating dummy variables dataset for the Sex feature

df_sex_get_dummies_test = pd.get_dummies(df_test['Sex'])

df_sex_get_dummies_test.head()
# Adding the Sex dummy variable dataset to the test dataset

df_test[['female', 'male']] = df_sex_get_dummies_test

df_test.head()
# Creating dummy variables dataset for the Pclass feature

df_Pclass_get_dummies_test = pd.get_dummies(df_test['Pclass'])

df_Pclass_get_dummies_test.head()
# Adding the Pclass dummy variable dataset to the test dataset

df_test[['1_Class', '2_Class', '3_Class']] = df_Pclass_get_dummies_test[[1,2,3]]

df_test.head()
# Taking the passenger title from the name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in df_test["Name"]]

df_test["Title"] = pd.Series(dataset_title)

df_test.head()
# Creating dummy variables dataset for the Pclass feature

df_title_get_dummies_test = pd.get_dummies(df_test['Title'])

df_title_get_dummies_test.head()
# Replacing the least commom titles by "Rare"

df_test["Title"] = df_test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Mme'], 'Rare')

df_title_get_dummies_test = pd.get_dummies(df_test['Title'])

df_title_get_dummies_test.head()
# Adding the Title dummy variable dataset to the test dataset

df_test[['Master', 'Miss', 'Mr', 'Mrs', 'Ms', 'Rare']] = df_title_get_dummies_test[['Master', 'Miss', 'Mr', 'Mrs', 'Ms', 'Rare']]

df_test.head()
# Creating dummy variables dataset for the Embarked feature

df_embarked_get_dummies_test = pd.get_dummies(df_test['Embarked'])

df_embarked_get_dummies_test.head()
# Adding the Embarked dummy variable dataset to the test dataset

df_test[['Cherbourg', 'Queenstown', 'Southampton']] = df_embarked_get_dummies_test[['C', 'Q', 'S']]

df_test.head()
# Taking off the columns data will not be used in the model

df_test = df_test.drop(['Pclass','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis = 1)

df_test.head()
# Defining the Dependent Variables

X_test = df_test.iloc[:,1:]

df_X_test = pd.DataFrame(X_test)

df_X_test.head()
X_test.shape
X_train.shape
# Feature Scaling

X_test = sc_X.transform(X_test)
# Making the predictions

y_pred = gsSVMC.predict(X_test)
# Output of the program

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)