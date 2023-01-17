import pandas as pd

DATA_DIR = '../input/'

train_df = pd.read_csv(DATA_DIR + 'train.csv')
test_df = pd.read_csv(DATA_DIR + 'test.csv')
full_df = [train_df, test_df]

train_df.info()
print('_'*40)
test_df.info()
import matplotlib.pyplot as plt
import seaborn as sns

# Decrease the density of x labels
def decrease_density(ax, rate):
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % rate == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

# Set the grid plot
f, ax_arr  = plt.subplots(3, 3, figsize=(20, 20))

# Build the barplot
sns.factorplot(x="Pclass", data=train_df, kind="count", ax=ax_arr[0][0])
sns.factorplot(x="Sex", data=train_df, kind="count", ax=ax_arr[0][1])
g = sns.factorplot(x="Age", data=train_df, kind="count", ax=ax_arr[0][2])
decrease_density(ax_arr[0][2], 10)
sns.factorplot(x="SibSp", data=train_df, kind="count", ax=ax_arr[1][0])
sns.factorplot(x="Parch", data=train_df, kind="count", ax=ax_arr[1][1])
sns.factorplot(x="Ticket", data=train_df, kind="count", ax=ax_arr[1][2])
decrease_density(ax_arr[1][2], 100)
sns.factorplot(x="Fare", data=train_df, kind="count", ax=ax_arr[2][0])
decrease_density(ax_arr[2][0], 50)
sns.factorplot(x="Cabin", data=train_df, kind="count", ax=ax_arr[2][1])
decrease_density(ax_arr[2][1], 10)
sns.factorplot(x="Embarked", data=train_df, kind="count", ax=ax_arr[2][2])

# Delete the seaborn plots
for i in range(2,11):
    plt.close(i)
# Drop non relevant features
new_train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
new_test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(new_train_df.isnull().sum())
print('_'*40)
new_test_df.isnull().sum()
# Fill the missing values with median or mode
new_train_df['Age'].fillna(new_train_df['Age'].median(), inplace = True)
new_test_df['Age'].fillna(new_test_df['Age'].median(), inplace = True)
new_train_df['Embarked'].fillna(new_train_df['Embarked'].mode()[0], inplace = True)
new_test_df['Fare'].fillna(new_test_df['Fare'].median(), inplace = True)
# Build the Family features
new_train_df['Family'] = new_train_df ['SibSp'] + new_train_df['Parch'] + 1
new_test_df['Family'] = new_test_df['SibSp'] + new_test_df['Parch'] + 1
new_train_df = new_train_df.drop(['SibSp', 'Parch'], axis=1)
new_test_df = new_test_df.drop(['SibSp', 'Parch'], axis=1)

# Build the Title features
new_train_df['Title'] = train_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
new_test_df['Title'] = test_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
from sklearn.preprocessing import LabelEncoder

# Convert categorical variables Sex, Embarked and Title into dummy variables
label = LabelEncoder()
new_train_df['Sex'] = label.fit_transform(new_train_df['Sex'])
new_test_df['Sex'] = label.fit_transform(new_test_df['Sex'])
new_train_df['Embarked'] = label.fit_transform(new_train_df['Embarked'])
new_test_df['Embarked'] = label.fit_transform(new_test_df['Embarked'])
new_train_df['Title'] = label.fit_transform(new_train_df['Title'])
new_test_df['Title'] = label.fit_transform(new_test_df['Title'])

# Convert continuous variables Age and Fare to categorical variables
new_train_df['Age'] = pd.cut(new_train_df['Age'], 10)
new_test_df['Age'] = pd.cut(new_test_df['Age'], 10)
new_train_df['Fare'] = pd.cut(new_train_df['Fare'], 10)
new_test_df['Fare'] = pd.cut(new_test_df['Fare'], 10)

new_train_df['Age'] = label.fit_transform(new_train_df['Age'])
new_test_df['Age'] = label.fit_transform(new_test_df['Age'])
new_train_df['Fare'] = label.fit_transform(new_train_df['Fare'])
new_test_df['Fare'] = label.fit_transform(new_test_df['Fare'])
new_train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
new_train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
new_train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
new_train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean()
new_train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
new_train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
new_train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=new_train_df)
sns.barplot(x='Embarked', y='Survived', hue='Sex', data=new_train_df)
sns.pointplot(x='Age', y='Survived', hue='Sex', data=new_train_df)
sns.barplot(x='Fare', y='Survived', hue='Sex', data=new_train_df)
sns.barplot(x='Family', y='Survived', hue='Sex', data=new_train_df)
# Inport machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create the training features set and labels set as well as the test features set
X_train = new_train_df.drop("Survived", axis=1)
Y_train = new_train_df["Survived"]
X_val  = new_test_df

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)

acc_log = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_log
coeff_df = pd.DataFrame(new_train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_svc
# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_knn
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_gaussian
# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_perceptron
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_sgd
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_decision_tree
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_random_forest
from sklearn.metrics import make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV

svc = SVC()

# Choose some parameter combinations to try
parameters = {
    'C': [0.3, 0.7, 1.0, 1.3, 1.7, 2],
    'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
    'degree': [1, 2, 3, 4, 5],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(svc, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
acc_svc = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_svc
from sklearn.cross_validation import KFold
import numpy as np

X_train_values = new_train_df.drop("Survived", axis=1).values
Y_train_values = new_train_df["Survived"].values

kf = KFold(X_train.shape[0], n_folds=10)
outcomes = []
fold = 0

for train_index, test_index in kf:
    fold += 1
    X_train, X_test = X_train_values[train_index], X_train_values[test_index]
    y_train, y_test = Y_train_values[train_index], Y_train_values[test_index]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
    print("Fold {0} accuracy: {1}".format(fold, accuracy))     
mean_outcome = np.mean(outcomes)
print("Mean Accuracy: {0}".format(mean_outcome))
final_pred = clf.predict(X_val)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": final_pred
    })

# submission.to_csv('./submission.csv', index=False)
