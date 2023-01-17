# General
import numpy as np 
import pandas as pd 
import os

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
combined = [titanic_train, titanic_test]
titanic_train.head()
print(titanic_train.isna().sum())
print('-' * 50)
print(titanic_test.isna().sum())
# For fariness we fill empty values in test set with values for training data
for dataset in combined:
    dataset['Embarked'] = dataset.Embarked.fillna(titanic_train.Embarked.value_counts().idxmax())
# Create hasCabin variable 
for dataset in combined:
    dataset['hasCabin'] = dataset.Cabin.apply(lambda x: 0 if pd.isnull(x) else 1)
# Deal with empty Fare entry in test dataset - get average Fare for each class and fill that value with the result
mean_fare_by_class = titanic_train.groupby(['Pclass']).mean()['Fare']
def fill_empty_fare(x):
    if pd.isnull(x['Fare']):
        return mean_fare_by_class[x['Pclass']]
    else:
        return x['Fare']

for dataset in combined:
    dataset['Fare'] = dataset.apply(lambda x: fill_empty_fare(x), axis=1)
# Ok, what's left is Age column. As explained, I will take average of age after grouping by multiple categories - Sex, Pclass and Parch, 
# as I believe that the number of parents or children on-board does influence the age of a person
mean_age_by_categories = titanic_train.groupby(['Sex', 'Pclass', 'Parch']).mean()['Age']
def fill_empty_age(x):
    if pd.isnull(x['Age']):
        try:
            return mean_age_by_categories.loc[x['Sex'], x['Pclass'], x['Parch']]
        except:
            # To avoid indexing error, we need to get Parch that exists in the traning data, so we pick the closest available
            closest_family = min(mean_age_by_categories\
                                 .reset_index('Parch')\
                                 .filter(regex='\\b'+x['Sex']+'\\b', axis=0)['Parch']\
                                 .unique(), 
                                 key=lambda y: abs(y-x['Parch']))
            return mean_age_by_categories.loc[x['Sex'], x['Pclass'], closest_family]
    else:
        return x['Age']
    
for dataset in combined:
    dataset['Age'] = dataset.apply(lambda x: fill_empty_age(x), axis=1)
# Now that we have used Parch to extrapolate the not available ages, we can combine it with SibSp to create FamilySize variable
for dataset in combined:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
# First lets split Fare into 5 categories
fare_brackets = sorted(pd.qcut(titanic_train.Fare, 5).unique())
for dataset in combined:
    for i, bracket in enumerate(fare_brackets):
        if i == len(fare_brackets) - 1:
            dataset.loc[(dataset.Fare > bracket.left), 'FareBracket'] = i
        else:
            dataset.loc[(dataset.Fare > bracket.left) & (dataset.Fare <= bracket.right), 'FareBracket'] = i
            
    dataset['FareBracket'] = dataset['FareBracket'].astype(int)
# C = Cherbourg, Q = Queenstown, S = Southampton
g = sns.FacetGrid(titanic_train, hue='Embarked', col='Pclass', margin_titles=True, palette={'S': 'green', 'C': 'blue', 'Q': 'red'}, height=10)
g = g.map(plt.scatter, 'FareBracket', 'Age', edgecolor='w').add_legend()
g = sns.FacetGrid(titanic_train, hue='Survived', col='Pclass', margin_titles=True, palette={1: 'blue', 0: 'red'}, height=10)
g = g.map(plt.scatter, 'FareBracket', 'Age', edgecolor='w').add_legend()
g = sns.FacetGrid(titanic_train, hue='Survived', col='Sex', margin_titles=True, palette={1: 'blue', 0: 'red'}, height=10)
g = g.map(plt.scatter, 'FamilySize', 'Age', edgecolor='w').add_legend()
g = sns.FacetGrid(titanic_train, hue='Survived', col='hasCabin', margin_titles=True, palette={1: 'blue', 0: 'red'}, height=10)
g = g.map(plt.scatter, 'Fare', 'Age', edgecolor='w').add_legend()
# First let's drop out the unncessary columns
for dataset in combined:
    dataset = dataset.drop(['Name', 'SibSp', 'Parch', 'Cabin', 'Ticket', 'hasCabin'], axis=1, inplace=True)
titanic_train.head()
# Based on the conducted analysis, I'll convert Embarked to ordinal variables to give point of embarked with highest survival
# rate the biggest weights
titanic_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
embarked_mappings = {'C': 1, 'Q': 2, 'S': 3}
for dataset in combined:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mappings)
sex_mappings = {'male': 0, 'female': 1}
for dataset in combined:
    dataset['Sex'] = dataset['Sex'].map(sex_mappings)
# Since majority of our variables are ordinal, we don't want to encode them into one-hot, because that would kill the
# connections between them. We will use Random Forest, thus no need to feature scaling either.
# Pclass and Fare seem to connected, so it might be best to drop one (Fare?). Let's try some methods
X_train = titanic_train.drop(['PassengerId', 'Survived', 'FareBracket'], axis=1)
X_test = titanic_test.drop(['FareBracket'], axis=1)

y_train = titanic_train['Survived']
X_train.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(X_train) 
X_train.loc[:,:] = scaled_values
X_test = X_test.drop('PassengerId', axis=1)
scaled_values_test = scaler.transform(X_test)
X_test.loc[:,:] = scaled_values_test
# X_test = pd.concat([X_test, titanic_test['PassengerId']], axis=1)
# 98.2% accuracy means that the model is extremely overfit to the train data. Let's divide it into train/test so we can
# get more accurate scores
X_train_splitted, X_test_splitted, y_train_splitted, y_test_splitted = train_test_split(X_train, y_train, test_size=.2)
X_train.head()
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_splitted, y_train_splitted)

predictions = clf.predict(X_train_splitted)
acc = round(accuracy_score(predictions, y_train_splitted) * 100, 2)
print('Accuracy on the train dataset is: {}%'.format(acc))

predictions = clf.predict(X_test_splitted)
acc = round(accuracy_score(predictions, y_test_splitted) * 100, 2)
print('Accuracy on the test dataset is: {}%'.format(acc))
# Overfitting confirmed. Let's use grid search to find perfect parameters for our model
clf = RandomForestClassifier()

parameters = {'max_features': ['log2', 'sqrt', 'auto'],
              'n_estimators': [2, 4, 10, 50, 100, 500],
              'min_samples_leaf': [2, 4, 10, 50, 100, 500]}

grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train_splitted, y_train_splitted)
best_clf = grid_search.best_estimator_

predictions = best_clf.predict(X_train_splitted)
acc = round(accuracy_score(predictions, y_train_splitted) * 100, 2)
print('Accuracy on the train dataset is: {}%'.format(acc))

predictions = best_clf.predict(X_test_splitted)
acc = round(accuracy_score(predictions, y_test_splitted) * 100, 2)
print('Accuracy on the test dataset is: {}%'.format(acc))
# Slightly better but still not satisfactory...
feature_importances = pd.DataFrame(best_clf.feature_importances_,
                                   index = X_train_splitted.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances
# Ok, that didn't work well, with a bit less than 80% accuracy on the test set.
# Let's try something new. How about adding age brackets, considering FareBrackets rather than fare
# Preparing sub-family categories and changing the ordinal variables to dummy variables
# # Lets split Age into 7 categories
# age_brackets = sorted(pd.qcut(titanic_train.Age, 7).unique())
# age_brackets
# for dataset in combined:
#     for i, bracket in enumerate(age_brackets):
#         if i == 0:
#             dataset.loc[(dataset.Age <= bracket.right), 'AgeBracket'] = i
#         elif i == len(age_brackets) - 1:
#             dataset.loc[(dataset.Age > bracket.left), 'AgeBracket'] = i
#         else:
#             dataset.loc[(dataset.Age > bracket.left) & (dataset.Age <= bracket.right), 'AgeBracket'] = i
            
#     dataset['AgeBracket'] = dataset['AgeBracket'].astype(int)
# for dataset in combined:
#     dataset['isAlone'] = 0
#     dataset['smallFamily'] = 0
#     dataset['mediumFamily'] = 0
#     dataset['bigFamily'] = 0
# for dataset in combined:
#     dataset.loc[(dataset.FamilySize == 0), 'isAlone'] = 1
#     dataset.loc[((dataset.FamilySize > 0) & (dataset.FamilySize <= 3)), 'smallFamily'] = 1
#     dataset.loc[((dataset.FamilySize > 3) & (dataset.FamilySize <= 6)), 'mediumFamily'] = 1
#     dataset.loc[(dataset.FamilySize > 6), 'bigFamily'] = 1
# # Let's remap embarked so we can easier find the dummies columns names
# embarked_mappings = {1: 'C', 2: 'Q', 3: 'S'}
# agebracket_mappings = {0: 'Baby', 1: 'Child', 2: 'Youth', 3: 'Teen', 4: 'Adult', 5: 'Elderly', 6: 'Ancient'}
# farebracket_mappings = {0: 'Cheap', 1: 'Standard', 2: 'Pricey', 3: 'Expensive', 4: 'Fortune'}
# class_mappings = {1: 'High', 2: 'Medium', 3: 'Low'}

# for dataset in combined:
#     dataset['Embarked'] = dataset['Embarked'].map(embarked_mappings)
#     dataset['AgeBracket'] = dataset['AgeBracket'].map(agebracket_mappings)
#     dataset['FareBracket'] = dataset['FareBracket'].map(farebracket_mappings)
#     dataset['Pclass'] = dataset['Pclass'].map(class_mappings)
# titanic_all = pd.concat([titanic_train, titanic_test], axis=0, sort=False)
# titanic_all.head()
# def convert_to_dummies(titanic_all, category):
#     dummies = pd.get_dummies(titanic_all[category])
#     titanic_all = pd.concat([titanic_all, dummies], axis=1, sort=False)
#     titanic_all.drop(category, axis=1, inplace=True)
    
#     return titanic_all
# titanic_all = convert_to_dummies(titanic_all, 'AgeBracket')
# titanic_all = convert_to_dummies(titanic_all, 'Embarked')
# titanic_all = convert_to_dummies(titanic_all, 'FareBracket')
# titanic_all = convert_to_dummies(titanic_all, 'Pclass')
# titanic_all.head()
# titanic_train = titanic_all[:len(titanic_train)]
# titanic_test = titanic_all[len(titanic_train):]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
# X_train = titanic_train.drop(['PassengerId', 'Survived', 'Age', 'Fare'], axis=1)
# y_train = titanic_train['Survived'].astype(int)

# X_test = titanic_test.drop(['PassengerId', 'Survived', 'Age', 'Fare'], axis=1)
# X_train.head()
# X_train_splitted, X_test_splitted, y_train_splitted, y_test_splitted = train_test_split(X_train, y_train, test_size=.2)
# clf = RandomForestClassifier()

# parameters = {'max_features': ['log2', 'sqrt', 'auto'],
#               'n_estimators': [2, 4, 10, 50, 100, 500],
#               'min_samples_leaf': [2, 4, 10, 50, 100, 500]}

# grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
# grid_search.fit(X_train_splitted, y_train_splitted)
# best_clf = grid_search.best_estimator_

# predictions = best_clf.predict(X_train_splitted)
# acc = round(accuracy_score(predictions, y_train_splitted) * 100, 2)
# print('Accuracy on the train dataset is: {}%'.format(acc))

# predictions = best_clf.predict(X_test_splitted)
# acc = round(accuracy_score(predictions, y_test_splitted) * 100, 2)
# print('Accuracy on the test dataset is: {}%'.format(acc))
# feature_importances = pd.DataFrame(best_clf.feature_importances_,
#                                    index = X_train_splitted.columns,
#                                    columns=['importance']).sort_values('importance', ascending=False)
# feature_importances
clf = RandomForestClassifier()

parameters = {'max_features': ['log2', 'sqrt', 'auto'],
              'n_estimators': [2, 4, 10, 50, 100, 500],
              'min_samples_leaf': [2, 4, 10, 50, 100, 500]}

grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

random_forest_predictions = best_clf.predict(X_train)
acc = round(accuracy_score(random_forest_predictions, y_train) * 100, 2)
print('Random Forest accuracy on the train dataset is: {}%'.format(acc))

random_forest_predictions = best_clf.predict(X_test)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

parameters = {'penalty': ['l1','l2'], 
              'C': [0.001,0.01,0.1,1,10,100,1000]}

grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

logistic_regression_predictions = best_clf.predict(X_train)
acc = round(accuracy_score(logistic_regression_predictions, y_train) * 100, 2)
print('Logistic Regression accuracy on the train dataset is: {}%'.format(acc))

logistic_regression_predictions = best_clf.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

parameters = {'n_neighbors': list(range(1, 31))}

grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

knn_predictions = best_clf.predict(X_train)
acc = round(accuracy_score(knn_predictions, y_train) * 100, 2)
print('Logistic Regression accuracy on the train dataset is: {}%'.format(acc))

knn_predictions = best_clf.predict(X_test)
from sklearn.svm import SVC

clf = SVC()

parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}

grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

svc_predictions = best_clf.predict(X_train)
acc = round(accuracy_score(svc_predictions, y_train) * 100, 2)
print('Logistic Regression accuracy on the train dataset is: {}%'.format(acc))

svc_predictions = best_clf.predict(X_test)
from sklearn.neural_network import MLPClassifier
import itertools

clf =  MLPClassifier(solver='adam', 
                     alpha=1e-5,
                     hidden_layer_sizes=(10, 20, 15))

# Training takes ages on CPU, thus no grid search
# parameters = {
#     'solver': ['lbfgs', 'adam'],
#     'alpha': [x for x in 10.0 ** -np.arange(1, 7)],
#     'hidden_layer_sizes': [x for x in itertools.product((10, 20, 50, 100), repeat=3)],
#     'activation': ['relu']
# }

# grid_search = GridSearchCV(clf, parameters, cv=5, scoring=make_scorer(accuracy_score))
# grid_search.fit(X_train, y_train)

# best_clf = grid_search.best_estimator_

# nn_predictions = best_clf.predict(X_train)

clf.fit(X_train, y_train)
nn_predictions = clf.predict(X_train)
acc = round(accuracy_score(nn_predictions, y_train) * 100, 2)
print('Neural Network accuracy on the train dataset is: {}%'.format(acc))

nn_predictions = clf.predict(X_test)
# Since SVC prediction was rather low, lets use another random forest instead, so it's vote get higher weight
predictions = pd.DataFrame({
    'RandomForestPred': random_forest_predictions,
    'LogisticRegressionPred': logistic_regression_predictions,
    'KNNPred': knn_predictions,
    'RandomForestPred2': random_forest_predictions,
    'NNPred': nn_predictions
})

predictions['FinalPrediction'] = predictions.mode(axis=1)
predictions.head()
ids = titanic_test['PassengerId'].astype(int)
output = pd.DataFrame({
    'PassengerId': ids, 
    'Survived': predictions['FinalPrediction']})

output.to_csv('submission_6.csv', index=False)
output.head()
