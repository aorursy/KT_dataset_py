# Load dependencies.
import numpy as np
import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Load train and test data.
test_data_raw = pd.read_csv('/kaggle/input/titanic/test.csv', sep=',')
train_data_raw = pd.read_csv('/kaggle/input/titanic/train.csv', sep=',')

# Take a quick look into the data.
train_data_raw.head(8)
# Remove useless columns fill missing age values with mean.
train_data = train_data_raw.drop(['Cabin', 'Fare', 'Name', 'Ticket', 'Embarked'], axis=1)
train_data['Age'] = train_data.fillna(train_data.mean())['Age']
# Calculate survival rate by gender.
gender_rate = train_data[['Sex', 'Survived']].groupby('Sex').mean()

# Draw bar chart.
plt.figure(figsize=(3,4))
plt.title("Survival rate by gender")
sns.barplot(x=gender_rate.index, y=gender_rate['Survived'])
# Calculate survival rate by travel class.
class_rate = train_data[['Pclass', 'Survived']].groupby('Pclass').mean()

# Draw bar chart.
plt.figure(figsize=(4,4))
plt.title("Survival rate by travel class")
sns.barplot(x=class_rate.index, y=class_rate['Survived'])
# Calculate survival rate by age.
age_rate = train_data[['Age', 'Survived']].groupby('Age').mean()

# Draw line chart.
plt.figure(figsize=(10,4))
plt.title("Survival rate by age")
sns.lineplot(data=age_rate)
# Calculate survival rate by gender and travel class.
gender_class_rate = train_data[['Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass']).mean()

# Draw bar chart.
plt.figure(figsize=(8,4))
plt.title("Survival rate by gender and by travel class")
sns.barplot(x=gender_class_rate.index, y=gender_class_rate['Survived'])
# Make a model in which all men died and all women survived.
pred_survived = pd.Series(0, index=train_data.index)
pred_survived[
  train_data[train_data.Sex == 'female'].index
] = 1

accuracy_score(y_true = train_data['Survived'], y_pred = pred_survived)
# Make a model in which only whealty women survived.
pred_survived = pd.Series(0, index=train_data.index)
pred_survived[
  train_data[(train_data.Sex == 'female') & (train_data.Pclass <= 2)].index
] = 1

accuracy_score(y_true = train_data['Survived'], y_pred = pred_survived)
# Make a model following the 'women and children first' code of conduct.
pred_survived = pd.Series(0, index=train_data.index)
pred_survived[
  train_data[(train_data.Sex == 'female') | (train_data.Age <= 8)].index
] = 1

accuracy_score(y_true = train_data['Survived'], y_pred = pred_survived)
# Separate output from input columns.
X = train_data.drop('Survived', axis=1)
y = train_data['Survived'].copy()

# Convert categorical values into indicator variables.
X.loc[X['Sex'] == 'male', 'Sex'] = 0
X.loc[X['Sex'] == 'female', 'Sex'] = 1

# Separate data into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
# Create and train a classifier using a decision tree.
titanic_tree_classifier = DecisionTreeClassifier(max_leaf_nodes=7, random_state=0)
titanic_tree_classifier.fit(X_train, y_train)

# Make predictions.
y_pred_dt = titanic_tree_classifier.predict(X_test)

# Check the accuracy of the decision tree predictions.
accuracy_score(y_true = y_test, y_pred = y_pred_dt)
# Create and train a classifier using a random forest.
titanic_forest_classifier = RandomForestClassifier(n_estimators=3, max_leaf_nodes=4, random_state=324)  
titanic_forest_classifier.fit(X_train, y_train)

# Make predictions (again).
y_pred_rf = titanic_forest_classifier.predict(X_test)

# Check the accuracy of the random forest predictions.
accuracy_score(y_true = y_test, y_pred = y_pred_rf)
# Make the predictions using the random forest.
test_data = test_data_raw.drop(['Cabin', 'Fare', 'Name', 'Ticket', 'Embarked'], axis=1)
test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1
test_data = test_data.fillna(test_data.mean())

tree_test_predictions = titanic_forest_classifier.predict(test_data)

# Generate the submission file (to be uploaded to Kaggle).
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': tree_test_predictions})
output.to_csv('my_submission.csv', index=False)
print("The submission was successfully saved!")
