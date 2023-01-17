import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
train_data = pd.read_csv("../input/train.csv")

# first five train_data samples

train_data.head()
# summary of the training dataset

train_data.info()
# changing the datatype of the Pclass column

train_data['Pclass'] = train_data['Pclass'].astype('object')
train_data.isna().sum()
# filling the nan values in the age column with mean age

train_data['Age'] = train_data['Age'].fillna(round(train_data['Age'].mean()))

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
survival = pd.crosstab(train_data['Survived'], train_data['Sex'], normalize = True)

print(survival)

survival.plot(kind = 'Bar', stacked = True)
sns.countplot(y = "Pclass", hue = "Survived", data = train_data)
sns.countplot(y= 'Embarked', hue = "Survived", data = train_data)
numerical = list(set(['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']))

corr_matrix = train_data[numerical].corr()

sns.heatmap(corr_matrix, annot = True)
X = train_data.drop(["Survived", "Name", "Cabin", "Ticket", 'Sex', 'Embarked'], axis = 1)

embarked_sex = train_data[['Sex', 'Embarked']]

embarked_sex = pd.get_dummies(embarked_sex, prefix = ["Sex", "Embarked"])

y = train_data.Survived

X = pd.concat([X,embarked_sex], axis = 1)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.40)
my_model1 = RandomForestClassifier(random_state = 1)

pipeline1 = make_pipeline(my_model1)

pipeline1.fit(X_train, y_train)

y_val1 = pipeline1.predict(X_test)

print(accuracy_score(y_test, y_val1))
rf_model_on_full_data = RandomForestClassifier(random_state = 1)

rf_model_on_full_data.fit(X,y)
test_data = pd.read_csv("../input/test.csv")

test_data = test_data.drop(["Name", "Cabin", "Ticket"], axis = 1)

test_data = pd.get_dummies(test_data, prefix = ["Sex", "Embarked"])

test_data.head()
test_data.isna().sum()
test_data['Age'] = test_data['Age'].fillna(round(test_data.Age.mean()))

test_data['Fare'] = test_data['Fare'].fillna(round(test_data.Fare.mean()))
test_predictions1 = rf_model_on_full_data.predict(test_data)
output1 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions1})

output1.to_csv("submission1.csv", index = False)
my_scalar = StandardScaler()

my_model2 = KNeighborsClassifier(n_neighbors = 10, algorithm = 'ball_tree',

                              leaf_size = 20, weights = 'uniform', metric = 'manhattan')

pipeline2= make_pipeline(my_scalar, my_model2)

pipeline2.fit(X_train, y_train)

y_val2 = pipeline2.predict(X_test)

print(accuracy_score(y_test, y_val2))
knn_on_full_model = KNeighborsClassifier(n_neighbors = 10, algorithm = 'ball_tree',

                              leaf_size = 20, weights = 'uniform', metric = 'manhattan')

knn_pipe = make_pipeline(my_scalar, knn_on_full_model)

knn_pipe.fit(X, y)
test_predictions2 = knn_pipe.predict(test_data)
output2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions2})

output2.to_csv("submission2.csv", index = False)
my_model3 = DecisionTreeClassifier(criterion = 'gini', random_state = 17, max_depth = 6)

pipeline3 = make_pipeline(my_model3)

pipeline3.fit(X_train, y_train)

y_val3 = pipeline3.predict(X_test)

print(accuracy_score(y_test, y_val3))
tree_param = {'max_depth': range(1,11),

             'max_features': range(4,8)}

tree_grid = GridSearchCV(my_model3, tree_param, cv = 5, n_jobs = -1, verbose = True)

tree_grid.fit(X_train, y_train)

print(tree_grid.best_params_)

print(tree_grid.best_score_)

print(accuracy_score(y_test, tree_grid.predict(X_test)))
tree_model_on_full_data = DecisionTreeClassifier(criterion = 'gini',random_state = 1)

tree_param = {'max_depth': range(1,11),

             'max_features': range(4,8)}

tree_grid = GridSearchCV(tree_model_on_full_data, tree_param, cv = 5, n_jobs = -1, verbose = True)

tree_grid.fit(X, y)

print(tree_grid.best_params_)

print(tree_grid.best_score_)
test_predictions3 = tree_grid.predict(test_data)
output3 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions3})

output3.to_csv("submission3.csv", index = False)