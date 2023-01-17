from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
# Data wrangling and EDA
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train_test = [train, test]
train.head()
test.head()
train.info()
train.describe()
train.describe(include = ['O'])
test.info()
test.describe()
test.describe(include = ['O'])
survived = train[train.Survived == 1]
passengers_survived = len(survived)
survived_percent = round(float(len(survived))/len(train)*100.0, 2)

died = train[train.Survived == 0] 
passengers_died = len(died)
died_percent = round(float(len(died))/len(train)*100.0, 2)

passengers_total = len(train)

print("Survived: {} passengers ({}%)".format(passengers_survived, survived_percent))
print("Died: {} passengers ({}%)".format(passengers_died,died_percent))
print("Total: {} passengers".format(passengers_total))
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass', y='Survived', data=train)
male = train[train.Sex == 'male']
passengers_male = len(male)
male_percent = round(float(len(male))/len(train)*100.0, 2)

female = train[train.Sex == 'female'] 
passengers_female = len(female)
female_percent = round(float(len(female))/len(train)*100.0, 2)

passengers_total = len(train)

print("Male: {} passengers ({}%)".format(passengers_male, male_percent))
print("Female: {} passengers ({}%)".format(passengers_female, female_percent))
print("Total: {} passengers".format(passengers_total))
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Sex', y='Survived', palette = 'Blues_d', data=train)
sns.factorplot('Sex', 'Survived', hue='Pclass', aspect=2, data=train)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=True)
sns.barplot(x='SibSp', y='Survived', ci=None, palette = 'Blues_d',data=train)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=True)
sns.barplot(x='Parch', y='Survived', ci=None, palette = 'Blues_d', data=train)
s = train[train.Embarked == 'S']
passengers_s = len(s)
s_percent = round(float(len(s))/len(train)*100.0, 2)

c = train[train.Embarked == 'C'] 
passengers_c= len(c)
c_percent = round(float(len(c))/len(train)*100.0, 2)

q = train[train.Embarked == 'Q'] 
passengers_q= len(q)
q_percent = round(float(len(q))/len(train)*100.0, 2)

passengers_total = len(train)

print("S: {} passengers ({}%)".format(passengers_s, s_percent))
print("C: {} passengers ({}%)".format(passengers_c, c_percent))
print("Q: {} passengers ({}%)".format(passengers_q, q_percent))
print("Total: {} passengers".format(passengers_total))
sns.barplot(x='Embarked', y='Survived', data=train)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
sns.displot(x='Age', data=train, bins=range(0,81,1), hue = 'Survived',multiple = 'stack', binwidth = 2, aspect = 2)
sns.displot(x='Age', data=train, bins=range(0,81,1), col = 'Sex', hue = 'Sex', binwidth = 2, aspect = 2)
sns.displot(x='Age', data=train, bins=range(0,81,1), col = 'Survived', hue = 'Sex', multiple = 'stack', binwidth = 2, aspect = 2)
sns.displot(x='Age', data=train, bins=range(0,81,1), col = 'Pclass', hue = 'Pclass', multiple = 'stack', binwidth = 2, aspect = 2)
sns.displot(x='Age', data=train, bins=range(0,81,1), col = 'Survived', hue = 'Pclass', multiple = 'stack', binwidth = 2, aspect = 2)
sns.displot(x='Age', data=train, bins=range(0,81,1), col = 'Embarked', hue = 'Embarked', multiple = 'stack', binwidth = 2, aspect = 2)
sns.displot(x='Age', data=train, bins=range(0,81,1), col = 'Survived', hue = 'Embarked', multiple = 'stack', binwidth = 2, aspect = 2)
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
train_test = [train, test]
train.head()
test.head()
for dataset in train_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
for dataset in train_test:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt',
                                                 'Col', 'Don', 'Dr',
                                                 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
train_test = [train, test]
train.head()
test.head()
for dataset in train_test:
    dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
train.head()
test.head()
embarked_mode = train['Embarked'].dropna().mode()[0]

for dataset in train_test:
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)
for dataset in train_test:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
test.head()
for dataset in train_test:
    mean_age = dataset['Age'].mean()
    std_age = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint( (mean_age - std_age), (mean_age + std_age), size = age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['AgeBand'] = pd.cut(train['Age'], 5)

train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()
for dataset in train_test:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train = train.drop(['AgeBand'], axis=1)
train_test = [train, test]
train.head()
test.head()
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train['FareBand'] = pd.qcut(train['Fare'], 5)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in train_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
train = train.drop(['FareBand'], axis=1)
train_test = [train, test]
train.head()
test.head()

for dataset in train_test:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Family', ascending=True)
for dataset in train_test:
    dataset['Alone'] = 0
    dataset.loc[dataset['Family'] == 1, 'Alone'] = 1

train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp', 'Family'], axis=1)
test = test.drop(['Parch', 'SibSp', 'Family'], axis=1)
train_test = [train, test]
train.head()
test.head()
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
Y_pred_logistic_regression = logistic_regression.predict(X_test)
logistic_regression_score = round(logistic_regression.score(X_train, Y_train) * 100, 2)
print(logistic_regression_score)
coefficients = pd.DataFrame(train.columns.delete(0))
coefficients.columns = ['Feature']
coefficients['Correlation'] = pd.Series(logistic_regression.coef_[0])
coefficients.sort_values(by='Correlation', ascending=False)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
knn_score = round(knn.score(X_train, Y_train) * 100, 2)
print(knn_score)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
svc_score = round(svc.score(X_train, Y_train) * 100, 2)
print(svc_score)
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
nb_score = round(nb.score(X_train, Y_train) * 100, 2)
print(nb_score)
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_perceptron = perceptron.predict(X_test)
perceptron_score = round(perceptron.score(X_train, Y_train) * 100, 2)
print(perceptron_score)
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
linear_svc_score = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(linear_svc_score)
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
sgd_score = round(sgd.score(X_train, Y_train) * 100, 2)
print(sgd_score)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
decision_tree_score = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(decision_tree_score)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
random_forest_score = round(random_forest.score(X_train, Y_train) * 100, 2)
print(random_forest_score)
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
Y_pred_clf = clf.predict(X_train)
selected_clf_score = round(clf.score(X_train, Y_train) * 100, 2)

class_names = ['Survived', 'Not Survived']
true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

c_matrix = confusion_matrix(Y_train, Y_pred_clf)
np.set_printoptions(precision=2)

c_matrix_percent = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]

df_c_matrix = pd.DataFrame(c_matrix,index = true_class_names,columns = predicted_class_names)
df_c_matrix_percent = pd.DataFrame(c_matrix_percent, index = true_class_names, columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_c_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_c_matrix_percent, annot=True)
models = pd.DataFrame(
    {
        'Model': [
            'Logistic Regression',
            'Support Vector Machines (SVM)',
            'Linear SVC',
            'k-Nearest Neighbors (KNN)',
            'Decision Tree',
            'Random Forest',
            'Gaussian Naive Bayes Classifier',
            'Perceptron',
            'Stochastic Gradient Decent'
        ],
        'Score': [
            logistic_regression_score,
            svc_score,
            linear_svc_score,
            knn_score,
            decision_tree_score,
            random_forest_score,
            nb_score,
            perceptron_score,
            sgd_score]
    }
)

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_decision_tree
    }
)

submission.to_csv('submission.csv', index=False)