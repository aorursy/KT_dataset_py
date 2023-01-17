#First, the necessary tools:

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

print(os.listdir("../input"))
#Reading the data

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')



#Saving the PassengerID to the submission, since it's required...

submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']
#inspecting the data:

train.head()
print(train.info())
#Here I'm removing things that I think that wouldn't be useful

# (of course this is just a supposition, in fact, they are useful with proper use)

train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace = True)

test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace = True)



#Get_dummyes convert categorical variable into dummy/indicator variables

data_train = pd.get_dummies(train)

data_test = pd.get_dummies(test)



#"Filnna" fills empty values, I'm replacing they for the mean of the values.

# (but, again, there are better ways to do this... this is just an example)

data_train['Age'].fillna(data_train['Age'].mean(), inplace = True)

data_test['Age'].fillna(data_test['Age'].mean(), inplace = True)

data_test['Fare'].fillna(data_test['Fare'].mean(), inplace = True)
# Calculate correlation matrix

corr = data_train.corr()



# Plot heatmap of correlation matrix

plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True)

plt.yticks(rotation=0); plt.xticks(rotation=90)  # fix ticklabel directions

plt.tight_layout()  # fits plot area to the plot, "tightly"

plt.show()  # show the plot

plt.clf()  # clear the plot area
# Separating the data into variables to analyze (x) and the result (y) that we expect

x = data_train.drop('Survived', axis=1)

y = data_train['Survived']
# Parametrizing the method and testing using cross-validation

# You can change the parameters to get better results; ^^

classifier_rf = RandomForestClassifier(

                criterion='gini',

                max_depth=50,

                n_estimators=100,

                n_jobs=-1)

    

scores_rf = cross_val_score(classifier_rf, x, y, scoring='accuracy', cv=5)

print(scores_rf.mean())
#this result is for the training... we want to know for tests, so:

from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
#Here we do some parameter optimization



from sklearn.model_selection import ParameterGrid



# Create a dictionary of hyperparameters to search

grid = {'n_estimators': [150, 100, 50], 'max_depth': [10, 25, 50], 'max_features': [4, 6, 8], 'random_state': [42], 'criterion': ['gini']}

test_scores = []



# Loop through the parameter grid, set the hyperparameters, and save the scores

for g in ParameterGrid(grid):

    classifier_rf.set_params(**g)  # ** is "unpacking" the dictionary

    classifier_rf.fit(train_X, train_y)

    test_scores.append(classifier_rf.score(test_X, test_y))



# Find best hyperparameters from the test score and print

best_idx = np.argmax(test_scores)

print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
#ok, let's fit the model with the best parameters in test



classifier_rf = RandomForestClassifier(

                criterion='gini',

                max_depth=10,

                max_features=4,

                n_estimators=50,

                random_state=42,

                n_jobs=-1)



# Training the model with the data

classifier_rf.fit(x, y)

predictions = classifier_rf.predict(test_X)

print(confusion_matrix(test_y,predictions))

print(classification_report(test_y,predictions))
# Get feature importances from our random forest model

importances = classifier_rf.feature_importances_



# Get the index of importances from greatest importance to least

sorted_index = np.argsort(importances)[::-1]

aux = range(len(importances))



# Create tick labels

labels = x.columns.values

plt.figure(figsize=(10,10))

plt.bar(aux, importances[sorted_index], tick_label=labels)



# Rotate tick labels to vertical

plt.xticks(rotation=90)

plt.show()
mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

       beta_2=0.999, early_stopping=False, epsilon=1e-08,

       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',

       learning_rate_init=0.001, max_iter=300, momentum=0.9,

       nesterovs_momentum=True, power_t=0.5, random_state=42,

       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,

       verbose=False, warm_start=False)



mlp.fit(x,y)
predictions = mlp.predict(test_X)

print(confusion_matrix(test_y,predictions))

print(classification_report(test_y,predictions))
# Creating a submission

submission['Survived'] = classifier_rf.predict(data_test)

submission.to_csv('submission.csv', index=False)