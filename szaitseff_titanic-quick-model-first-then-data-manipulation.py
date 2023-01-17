# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Import train and test data:
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# Split imported train dataset into train and dev (cross_validation) data
train_data, dev_data = train_test_split(train_data, test_size=0.25)
# review examples of data 
train_data.head()
# review information on data, pay attention to missing values and dtypes
train_data.info()
print("_" * 40)
dev_data.info()
print("_" * 40)
test_data.info()
# look up the passenger with missing Fare value in the test_data: 
test_data[test_data.Fare.isnull()]
# make an estimate of the missing Fare value for the above passenger in 3rd class embarked at S: "13.9"
test_data[["Pclass", "Fare", "Embarked"]].groupby(["Pclass", "Embarked"]).mean()
# exclude data of weak correlation with survival and with many missing values (don't spend time on new features for this base model):
X_train = train_data.drop(["PassengerId", "Survived", "Name", "Age", "Ticket", "Cabin"], axis=1)
Y_train = train_data["Survived"]
X_dev = dev_data.drop(["PassengerId", "Survived", "Name", "Age", "Ticket", "Cabin"], axis=1)
Y_dev = dev_data["Survived"]
X_test = test_data.drop(["PassengerId", "Name", "Age", "Ticket", "Cabin"], axis=1)
X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape, X_test.shape
# convert alpha-numerical data to numbers in Sex and Embarked, and fill in null data in Embarked and Fare
X_full = [X_train, X_dev, X_test]
for dataset in X_full:
    dataset["Sex"] = dataset["Sex"].map({"female": "1", "male": "0"}).astype("int")
    dataset["Embarked"] = dataset["Embarked"].fillna("S").map({"S": "0", "C": "1", "Q": "2"}).astype("int")
X_test["Fare"] = X_test["Fare"].fillna(13.9)
# Train a Logistic regresson model and predict survival on dev data
logit = LogisticRegression()
logit.fit(X_train, Y_train)
acc_logit_train = round(logit.score(X_train, Y_train)*100, 2)
acc_logit_dev = round(logit.score(X_dev, Y_dev)*100, 2)
print(f"logit: train accuracy = {acc_logit_train}, dev accuracy = {acc_logit_dev}")
# Train a Support Vextor Machine n model and predict survival on dev data
svc = SVC(C=1.0, kernel='rbf', gamma='auto')
svc.fit(X_train, Y_train)
acc_svc_train = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc_dev = round(svc.score(X_dev, Y_dev) * 100, 2)
print(f"svc: train accuracy = {acc_svc_train}, dev accuracy = {acc_svc_dev}")
# Let's now create new features and compare their effect on the model performance. 
# Let's start with extracting titles from Names and adding them as a new feature:
combine_data = [train_data, dev_data, test_data]
for dataset in combine_data:
    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
pd.crosstab(train_data['Title'], train_data['Sex'])
pd.crosstab(test_data['Title'], test_data['Sex'])
# Replace rare titles with more common ones
for dataset in combine_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Map titles to categories (numbers) for fitting the model
for dataset in combine_data:
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)
# Add the Title feature to our model data
X_train["Title"] = train_data["Title"]
X_dev["Title"] = dev_data["Title"]
X_test["Title"] = test_data["Title"]
X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape, X_test.shape

# Now we can train logit on expanded data and the effect on accuracy:
logit.fit(X_train, Y_train)
acc_logit_train = round(logit.score(X_train, Y_train)*100, 2)
acc_logit_dev = round(logit.score(X_dev, Y_dev)*100, 2)
print(f"logit: train accuracy = {acc_logit_train}, dev accuracy = {acc_logit_dev}")
svc.fit(X_train, Y_train)
acc_svc_train = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc_dev = round(svc.score(X_dev, Y_dev) * 100, 2)
print(f"svc: train accuracy = {acc_svc_train}, dev accuracy = {acc_svc_dev}")
# Let's normalize Fares data with mean and standard deviation of X_train:
mu = X_train["Fare"].mean()
sigma = (((X_train["Fare"]-mu)**2).mean())**0.5
for subset in X_full:
    subset["Fare"] = (subset["Fare"] - mu)/sigma
print(mu, sigma)
# Now we can train logit with normalized "Fare" data and see the effect on accuracy:
logit.fit(X_train, Y_train)
acc_logit_train = round(logit.score(X_train, Y_train)*100, 2)
acc_logit_dev = round(logit.score(X_dev, Y_dev)*100, 2)
print(f"logit: train accuracy = {acc_logit_train}, dev accuracy = {acc_logit_dev}")
# Support Vector Machines
svc.fit(X_train, Y_train)
acc_svc_train = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc_dev = round(svc.score(X_dev, Y_dev) * 100, 2)
print(f"svc: train accuracy = {acc_svc_train}, dev accuracy = {acc_svc_dev}")
### Choose predictions on test dataset:
#Y_pred = logit.predict(X_test)
Y_pred = svc.predict(X_test)

### Form the submission file:
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred}).sort_values(by="PassengerId")
submission.to_csv("submission.csv", index=False)
