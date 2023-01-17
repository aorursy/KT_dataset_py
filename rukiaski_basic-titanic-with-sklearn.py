# data processing, CSV file I/O (e.g. pd.read_csv)

# For convention, pandas always import using alias pd

import pandas as pd 



# To visualize graphs

import matplotlib.pyplot as plt



# Models used

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



# Checking available directories

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Load data with pandas

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data_complete = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()

# View the top rows of our dataset. Default value is 5. 

# If want to you view more or less, you can especify the value like 

# train_data.head(10)
test_data_complete.head()
# Remove columns that will not be use 

train_data = train_data.drop(['Name','Ticket', 'Cabin'], axis=1)

test_data_complete = test_data_complete.drop(['Name', 'Ticket', 'Cabin'], axis=1)



# I could use also

# train_data.drop(['Name','Ticket', 'Cabin'], axis=1, inplace=True)

# Using this, I don't need to save the modifications in a variable, because parameter inplace=True

# meaning that all modifications is automatically saved in data set.
# Tranform data using one-hot encoding.

# This part is important because machine learning models works with numbers, and in this dataset some columns

# has values objects, like Sex (male or female) and Embarked (Q, S, C).



train_data = pd.get_dummies(train_data)

test_data_complete = pd.get_dummies(test_data_complete)
train_data.head()
test_data_complete.head()
# Verifying null values

train_data.isnull().sum().sort_values(ascending=False)
#Filling null values with mean

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data_complete.isnull().sum().sort_values(ascending=False)
test_data_complete['Age'] = test_data_complete['Age'].fillna(test_data_complete['Age'].mean())

test_data_complete['Fare'] = test_data_complete['Fare'].fillna(test_data_complete['Fare'].mean())
# Separating features and targets

x_features = train_data.drop(['PassengerId','Survived'], axis=1)

y_targets = train_data['Survived']
x_features.head()
tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=0)

tree_classifier.fit(x_features, y_targets)
forest_classifier = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=0)

forest_classifier.fit(x_features, y_targets)
gbc_classifier = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=0)

gbc_classifier.fit(x_features, y_targets)
# Naive Bayes Classifier

naive_classifier = GaussianNB()

naive_classifier.fit(x_features, y_targets)
svc_classifier = SVC(C=5,gamma=10)

svc_classifier.fit(x_features, y_targets)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

knn_classifier.fit(x_features, y_targets)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(7,2), activation='relu',solver='adam', max_iter=300,

                               random_state=0, batch_size=250)

mlp_classifier.fit(x_features, y_targets)
results = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test_target = results['Survived']

test_data = test_data_complete.drop('PassengerId', axis=1)
tree_score_test = tree_classifier.score(test_data, test_target)

forest_score_test = forest_classifier.score(test_data, test_target)

gbc_score_test = gbc_classifier.score(test_data, test_target)

naive_score_test = naive_classifier.score(test_data, test_target)

svc_score_test = svc_classifier.score(test_data, test_target)

knn_score_test = knn_classifier.score(test_data, test_target)

mlp_score_test = mlp_classifier.score(test_data, test_target)
# Using matplotlib to visualize the best score

names = ['Decision Tree', 'Random Forest', 'GBC', 'Naive Bayes', 'SVC', 'KNN', 'MLP']

values = [tree_score_test, forest_score_test, gbc_score_test, naive_score_test, svc_score_test, 

         knn_score_test, mlp_score_test]



plt.bar(names,values)

plt.tick_params(axis ='x', rotation = 90)

plt.show()
submission = pd.DataFrame()

submission['PassengerId'] = test_data_complete['PassengerId']

submission['Survived'] = forest_classifier.predict(test_data)
submission.to_csv('submission.csv', index=False)

print("Submission was successfully saved!")