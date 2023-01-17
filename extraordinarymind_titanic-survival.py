# Import required packages

import pandas as pd

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

# Read titanic data

train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
# Test data

test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
# Shape of training and test data

print("Train_data:",train_data.shape)

print("Test_data:",test_data.shape)
train_data.columns
train_data.info()
# Check missing values

train_data.isnull().sum()
# Summary stats of the numerical data

train_data.describe()
# Convert Survived and Pclass to categorical

# train_data['Survived'] = train_data['Survived'].astype(object)

# train_data['Pclass'] = train_data['Pclass'].astype(object)
# Summary stats of categorical data

train_data.describe(include = ['O'])
train_data['Survived'].value_counts()
train_data[['Sex','Survived']].groupby(['Sex']).mean()
train_data[['Pclass','Survived']].groupby(['Pclass']).mean()
train_data[['SibSp','Survived']].groupby(['SibSp']).mean()
g = sns.FacetGrid(train_data,col = 'Survived')

g.map(plt.hist, 'Age', bins = 20)
# Remove ticket (too many values) and Cabin (Too many missing values)

train_df = train_data.drop(['Ticket','Cabin'], axis = 1)

test_df = test_data.drop(['Ticket','Cabin'], axis = 1)
# Combine train and test dataframe

combine = [train_df, test_df]

"After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)  
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
train_df[['Title','Survived']].groupby(['Title']).mean()
# Convert categorical columns to ordinal

title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()    
# Drop PassengerId, Name from dataframe

train_df = train_df.drop(['Name','PassengerId'], axis = 1)

test_df = test_df.drop(['Name'], axis = 1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
# Converting a categorical feature to numerical

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)

train_df.head()    
# Imputing missing values in age with mean value

for dataset in combine:

    dataset['Age'] = dataset['Age'].fillna((dataset['Age'].mean()))
# Create age bins and determine correlations with survived

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# Replace age with ordinals

for dataset in combine:

    dataset.loc[dataset['Age'] <=16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
# Drop ageband feature

train_df = train_df.drop(['AgeBand'], axis = 1)

combine = [train_df, test_df]

train_df.head()
# Find mode value in Embarked column

mode_embarked = train_df['Embarked'].dropna().mode()[0]

mode_embarked
# Replace missing value in Embarked with mode value

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(mode_embarked)
# Converting categorical feature into numerical

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)



train_df.head()
# Impute missing values in test_df Fare column 

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)

test_df.head()
# Divide the data into train and test

X_train = train_df.drop("Survived", axis = 1)

Y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis = 1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train,Y_train)*100,2)

print(acc_log)
# predictions on test data

test_df['predictions'] = Y_pred

submission = test_df[['PassengerId','predictions']]

submission.head()
submission.to_csv("logistic_predictions.csv",index=False)
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support vector machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100,2)

acc_svc
# predictions on test data

test_df['predictions'] = Y_pred

#test_df[['PassengerId','predictions']].to_csv("svm_predictions.csv",index=False)
# K-nearest neighbor

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train,Y_train)*100,2)

acc_knn
# prepare submissions on test data

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],

                           "Survived": Y_pred

                          })

#submission.to_csv("knn_submissions.csv",index = False)
# Gaussian naive bayes

gaussian = GaussianNB()

gaussian.fit(X_train,Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train,Y_train) * 100, 2)

acc_gaussian