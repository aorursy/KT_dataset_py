# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import re

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # data visualization



from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))  





training_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

gender_data = pd.read_csv("../input/titanic/gender_submission.csv")



# Any results you write to the current directory are saved as output.
# Split the target from the dataset

X_train = training_data.drop(['Survived'], axis=1)

y_train = training_data["Survived"]



X_test = test_data.copy()

y_test = gender_data['Survived']



# Reindex the columns

X_train = X_train.reindex(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", "Ticket", "Cabin", "PassengerId"])

X_test = X_test.reindex(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", "Ticket", "Cabin", "PassengerId"])
def missing_data(X):

    total_missing_values = X.isnull().sum().sort_values(ascending=False)

    percent_1 = X.isnull().sum()/X.isnull().count()*100

    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

    missing_data = pd.concat([total_missing_values, percent_2], axis=1, keys=['Total', '%'])

    print(missing_data.head(5))



print("----- Before cleaning -----")

# Print columns sorted by the most NaN values

print(missing_data(X_train))

print(missing_data(X_test))
# Create 'Title' column and drop the Name column

datasets = [X_train, X_test]



titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in datasets:

    # We can import all titles because it have the same pattern in every row: '[LAST_NAME], [TITLE.], [FIRST NAMES]'.

    # So we get the word before the dot and put it to our new column 'Title'.    

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer', 'Countess', 'Lady', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Ms', 'Mlle'], 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

    dataset['Title'] = dataset['Title'].map(titles)

    dataset['Title'] = dataset['Title'].astype(int)

    # Drop the Name column because we extract everything we needed

    dataset.drop('Name', axis=1, inplace=True)
# Create 'Deck' column using the first letter of the Cabin column

dep = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 8}



for dataset in datasets:

    # When the value is not missing, the Cabin column always start with a letter and 1 to 3 numbers.

    # So we extract the letter for every row and give a number according to the letter (dict dep)

    # We give the NaN values a letter.

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile('([a-zA-Z])+').search(x).group())

    dataset['Deck'] = dataset['Deck'].map(dep)

    dataset['Deck'].fillna(0, inplace=True)

    dataset['Deck'] = dataset['Deck'].astype(int)

    # Drop Cabin because we don't need it anymore

    dataset.drop('Cabin', axis=1, inplace=True)
# Create 'relatives' and 'is_alone' based on SibSp and Parch columns (simple addition)



for dataset in datasets:

    # SibSp : Siblings/Spouses

    # Parch : Parents/childrens

    # relatives column gives us how many family members are in the boat

    # is_alone column gives us if the passenger is alone (1) or not (0)

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'is_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'is_alone'] = 1

    dataset['is_alone'] = dataset['is_alone'].astype(int)
for dataset in datasets:

    # Convert to a categorical column

    dataset.loc[(dataset.Sex == "male"), 'Sex'] = 0

    dataset.loc[(dataset.Sex == "female"), 'Sex'] = 1
for dataset in datasets:

    # Fill the NaN values with "S" because it's the most common case (only 2 missing values here)

    dataset["Embarked"].fillna("S", inplace=True)

    # Convert to a categorical column

    dataset.loc[(dataset.Embarked == "S"), "Embarked"] = 0

    dataset.loc[(dataset.Embarked == "Q"), "Embarked"] = 1

    dataset.loc[(dataset.Embarked == "C"), "Embarked"] = 2
# Fill the NaN values of the Age column



for dataset in datasets:

    # Fill the NaN values with random values near the mean (around the peak in a Gaussian Distribution) 

    Age_avg = dataset['Age'].mean()

    Age_std = dataset['Age'].std()

    Age_null_count = dataset['Age'].isnull().sum()

    Age_null_random_list = np.random.randint(Age_avg - Age_std, Age_avg + Age_std, size=Age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = Age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

# To cut the Age column in 6 parts balanced 

# pd.qcut(X_train.Age, 6)

    dataset.loc[(dataset.Age <= 18), 'Age'] = 0

    dataset.loc[(dataset.Age > 18) & (dataset.Age <= 23), 'Age'] = 1

    dataset.loc[(dataset.Age > 23) & (dataset.Age <= 29), 'Age'] = 2

    dataset.loc[(dataset.Age > 29) & (dataset.Age <= 35), 'Age'] = 3

    dataset.loc[(dataset.Age > 35) & (dataset.Age <= 42), 'Age'] = 4

    dataset.loc[(dataset.Age > 42), 'Age'] = 5
# We only have one missing value from the fare column (X_test), so we take the mean of the column

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)



for dataset in datasets:

    # To cut the Fare column in 6 parts balanced

# pd.qcut(X_train.Fare, 6)

    dataset.loc[(dataset.Fare <= 7.775), 'Fare'] = 0

    dataset.loc[(dataset.Fare > 7.775) & (dataset.Fare <= 8.662), 'Fare'] = 1

    dataset.loc[(dataset.Fare > 8.662) & (dataset.Fare <= 14.454), 'Fare'] = 2

    dataset.loc[(dataset.Fare > 14.454) & (dataset.Fare <= 26), 'Fare'] = 3

    dataset.loc[(dataset.Fare > 26) & (dataset.Fare <= 52.369), 'Fare'] = 4

    dataset.loc[(dataset.Fare > 52.369), 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

    # Drop Ticket column because the data is too messy and we can't find some patterns

    # and PassengerId columns because it brings nothing

    dataset.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)
print("----- After cleaning -----")



# Last check of missing values

print(missing_data(X_train))

print(missing_data(X_test))
# Last check of data



print("----- X_train -----")

print(X_train)

print()

print("----- X_test -----")

print(X_test)
# Trying different Supervised Learning Classification Algorithms



SVC_model = SVC(gamma='scale')

SVC_model.fit(X_train, y_train)

SVC_score = round(accuracy_score(SVC_model.predict(X_train), y_train) * 100, 2)



KNN_model = KNeighborsClassifier()

KNN_model.fit(X_train, y_train)

KNN_score = round(accuracy_score(KNN_model.predict(X_train), y_train) * 100, 2)



DC_model = DecisionTreeClassifier()

DC_model.fit(X_train, y_train)

DC_score = round(accuracy_score(DC_model.predict(X_train), y_train) * 100, 2)



XGB_model = XGBClassifier() 

XGB_model.fit(X_train, y_train)

XGB_score = round(accuracy_score(XGB_model.predict(X_train), y_train) * 100, 2)



RF_model = RandomForestClassifier(n_estimators=100)

RF_model.fit(X_train, y_train)

RF_score = round(accuracy_score(RF_model.predict(X_train), y_train) * 100, 2)



GNB_model = GaussianNB()

GNB_model.fit(X_train, y_train)

GNB_score = round(accuracy_score(GNB_model.predict(X_train), y_train) * 100, 2)
# Ranking dataframe of the models

ranking = pd.DataFrame({'Model': ['Support Vector Classification', 'KNN', 'Decision Classifier', 'Random Forest', 'Naive Bayes', 'XGBClassifer'], 'Score': [SVC_score, KNN_score, DC_score, RF_score, GNB_score, XGB_score]})

ranking = ranking.sort_values(by='Score', ascending=False).set_index('Score')

ranking
# Out-of-bag accuracy

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1, n_jobs=-1)

rf.fit(X_train, y_train)



# Cross_val_score gives all the scores according to the cv parameter, so we compute the mean of all the scores to have a global vision of the score



print("Cross val score:", round(cross_val_score(rf, X_train, y_train, cv=8, scoring='accuracy').mean(), 3), '(+/-', round(cross_val_score(rf, X_train, y_train, cv=8, scoring='accuracy').std(), 2), '%)')

print("Oob score:", round(rf.oob_score_, 4)*100, "%")
# Importances of every feature

importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(rf.feature_importances_, 3)})

importances = importances.sort_values('importance', ascending=False).set_index('feature')

importances
# parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [100, 400, 700, 1000, 1500], 'min_samples_leaf': [1, 5, 10, 25, 50, 70], 'min_samples_split': [2, 4, 10, 12, 16, 18, 25, 35]}

#clf = GridSearchCV(rf, parameters, n_jobs=-1, cv=5)

#clf.fit(X_train, y_train)

# Gives the parameters with the best scores

#clf.best_params_
# Fit the best parameters on a new model



rf = RandomForestClassifier(criterion="gini", min_samples_leaf=1, min_samples_split=35, n_estimators=1000, oob_score=True, random_state=1, n_jobs=-1)

rf.fit(X_train, y_train)



# Oob_score_ gives us a general accuracy

print("oob score:", round(rf.oob_score_, 4)*100, '%')
predictions = cross_val_predict(rf, X_train, y_train, cv=5)



print("Precision:", precision_score(y_train, predictions))

print("Recall:", recall_score(y_train, predictions))

print("F1:", f1_score(y_train, predictions))



y_scores = rf.predict_proba(X_train)

# Get the column with probability of 1 (example: 0.75 is the probability of output being 1)

y_scores = y_scores[:, 1]
precision, recall, threshold = precision_recall_curve(y_train, y_scores)



def plot_precision_and_recall(precision, recall, threshold):

    # Plot it to get more insights about precision and recall

    plt.plot(threshold, precision[:-1], "r-", label="precision")

    plt.plot(threshold, recall[:-1], "b", label="recall")

    plt.xlabel("threshold")

    plt.legend(loc="upper right")

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
r_a_score = roc_auc_score(y_train, y_scores)



# Gives us a good idea of how well the model perfoms. 

print("ROC-AUC-Score:", r_a_score)



preds = rf.predict(X_test)

submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': preds})

submission.to_csv('Titanic Predictions Final Submission.csv', index=False)