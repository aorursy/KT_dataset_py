# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import re



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression,Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score,GridSearchCV, cross_val_predict

from sklearn.metrics import confusion_matrix,precision_score, recall_score,f1_score, precision_recall_curve, roc_auc_score







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
missing_values = ["n/a", "na", "--"]

# Train dataset

train_data_path = '../input/titanic/train.csv'

train_data = pd.read_csv(train_data_path, na_values = missing_values)

print("Data set loaded. Number of train examples: ", len(train_data))



#Y_train = train_data.pop('Survived')



# Test dataset

test_data_path = '../input/titanic/test.csv'

test_data = pd.read_csv(test_data_path)

print("Data set loaded. Number of test examples: ", len(test_data))



# Gender Submission

gender_submission_data_path = '../input/titanic/gender_submission.csv'

gender_submission_data = pd.read_csv(gender_submission_data_path)

print("Data set loaded. Number of gender_submission examples: ", len(gender_submission_data))
train_data.head()
train_data.columns.values
train_data.describe()
train_data.dtypes
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



women = train_data[train_data['Sex']=='female']

men = train_data[train_data['Sex']=='male']



ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[0], kde = False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = 'not survived', ax = axes[0], kde = False)

ax.legend()

ax.set_title('Female')



ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = 'not survived', ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_data, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_data)
data = [train_data, test_data]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

train_data['not_alone'].value_counts()
axes = sns.factorplot('relatives','Survived', data=train_data, aspect = 2.5, )
train_data = train_data.drop(['PassengerId'], axis=1)
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_data, test_data]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

    

# we can now drop the cabin feature

train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)
data = [train_data, test_data]



for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)



train_data["Age"].isnull().sum()
train_data['Embarked'].describe()
common_value = 'S'

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_data.info()
data = [train_data, test_data]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
genders = {"male": 0, "female": 1}

data = [train_data, test_data]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
data = [train_data, test_data]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train_data = train_data.drop(['Name'], axis=1)

test_data = test_data.drop(['Name'], axis=1)
train_data['Ticket'].describe()
train_data = train_data.drop(['Ticket'], axis=1)

test_data = test_data.drop(['Ticket'], axis=1)
train_data.Embarked.value_counts()
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train_data, test_data]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



train_data['Age'].value_counts()
train_data['Fare'].value_counts()
pd.qcut(train_data['Fare'],6)
data = [train_data, test_data]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_data, test_data]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
data = [train_data, test_data]

for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
train_data.head()
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# KNN 

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
importances.plot.bar()
train_data = train_data.drop("not_alone", axis=1)

test_data = test_data.drop("not_alone", axis=1)



train_data = train_data.drop("Parch", axis=1)

test_data = test_data.drop("Parch", axis=1)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)



random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)

confusion_matrix(Y_train, predictions)
print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))
f1_score(Y_train, predictions)
# getting the probabilities of our predictions

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(Y_train, y_scores)

def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
r_a_score = roc_auc_score(Y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
Y_prediction
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':Y_prediction})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)