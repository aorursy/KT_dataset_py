# Importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
print(train.info())
print('-----------------------------------------------------------\n')
print(test.info())
plt.hist(train['Embarked'])
plt.show()
train['Embarked'] = train['Embarked'].fillna('S')
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
# Create dataframe that holds survived passengers
survival = train[train['Survived'] == 1]

# Create dataframe for all observations with ages that are not null
train_with_age = train[train['Age'].notnull()]

# Preparing the standard sizes and background color
plt.rcParams['figure.figsize'] = 20, 5
sns.set_style('darkgrid')
# Create dataframe to represent some basic statistics
gender_total = pd.DataFrame(columns=['Gender', 'Passengers', 'Survival'])

# Adding numbers to the above dataframe
for gender in ['male', 'female']:
    percentage = str(round(len(survival[survival['Sex'] == gender]) / len(train[train['Sex'] == gender]) * 100, 2))
    gender_total.loc[len(gender_total)] = gender, len(train[train['Sex'] == gender]), (percentage + ' %')

# Create histogram 
plt.hist(train['Sex'], bins=9, rwidth=0.9,  color = 'steelblue', label='Total')
plt.hist(survival['Sex'], bins=9, rwidth=0.9, color = 'darkseagreen', label='Survived')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.grid(axis='x')

# Create comparison charts (Box and swarm plots)
f, axes = plt.subplots(2, 2, figsize=(17, 10))
a1 = sns.boxplot(x='Sex', y='Age', data=train_with_age, hue='Survived', palette=['steelblue','darkseagreen'], showfliers=False, ax=axes[0, 0])
a2 = sns.swarmplot(x='Sex', y='Age', data=train_with_age, hue='Survived', palette=['steelblue','darkseagreen'], dodge=True, ax=axes[1, 0])
a3 = sns.boxplot(x='Sex', y='Fare', data=train, hue='Survived', palette=['steelblue','darkseagreen'], showfliers=False, ax=axes[0, 1])
a4 = sns.swarmplot(x='Sex', y='Fare', data=train, hue='Survived',palette=['steelblue','darkseagreen'], dodge=True, ax=axes[1, 1])

# Create comparison charts (Box and swarm plots)
b1 = a1.legend(loc='upper right')
b2 = a2.legend(loc='upper right')
b3 = a3.legend(loc='upper right')
b4 = a4.legend(loc='upper right')

plt.show()
gender_total
# Create dataframe to represent some basic statistics
embarkation_port = pd.DataFrame(columns=['Embarkation Port', 'Passengers', 'Port Survival', 'Port Survival (F)', 'Port Survival (M)'])

# Adding numbers to the above dataframe
for embark in ['C', 'Q', 'S']:
    percentage = str(round(len(survival[survival['Embarked'] == embark]) / len(train[train['Embarked'] == embark]) * 100, 2))
    female = str(round(len(survival[(survival['Embarked'] == embark) & (survival['Sex'] == 'female')]) / len(train[(train['Embarked'] == embark)]) * 100, 2))
    male = str(round(len(survival[(survival['Embarked'] == embark) & (survival['Sex'] == 'male')]) / len(train[(train['Embarked'] == embark)]) * 100, 2))
    embarkation_port.loc[len(embarkation_port)] = embark, len(train[train['Embarked'] == embark]), (percentage + ' %'), (female + ' %'), (male + ' %')

# Create histogram 
plt.hist(train['Embarked'], bins=9, rwidth=0.9,  color = 'steelblue', label='Total')
plt.hist(survival['Embarked'], bins=9, rwidth=0.9, color = 'darkseagreen', label='Survived')
plt.xlabel('Embarkation Port')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.grid(axis='x')

# Create comparison charts (Box and swarm plots)
f, axes = plt.subplots(2, 2, figsize=(17, 10))
a1 = sns.boxplot(x='Embarked', y='Age', data=train_with_age, hue='Survived', palette=['steelblue','darkseagreen'], showfliers=False, ax=axes[0, 0])
a2 = sns.swarmplot(x='Embarked', y='Age', data=train_with_age, hue='Survived', palette=['steelblue','darkseagreen'], dodge=True, ax=axes[1, 0])
a3 = sns.boxplot(x='Embarked', y='Fare', data=train, hue='Survived', palette=['steelblue','darkseagreen'], showfliers=False, ax=axes[0, 1])
a4 = sns.swarmplot(x='Embarked', y='Fare', data=train, hue='Survived',palette=['steelblue','darkseagreen'], dodge=True, ax=axes[1, 1])

# Adjust comparison charts' legends locations
b1 = a1.legend(loc='upper right')
b2 = a2.legend(loc='upper right')
b3 = a3.legend(loc='upper right')
b4 = a4.legend(loc='upper right')

plt.show()
display(embarkation_port)
# Create dataframe to represent some basic statistics
p_class = pd.DataFrame(columns=['Pclass', 'Passengers', 'Class Survival', 'Class Survival (F)', 'Class Survival (M)'])

# Adding numbers to the above dataframe
for pclass in range(1, 4):
    percentage = str(round(len(survival[survival['Pclass'] == pclass]) / len(train[train['Pclass'] == pclass]) * 100, 2))
    female = str(round(len(survival[(survival['Pclass'] == pclass) & (survival['Sex'] == 'female')]) / len(train[(train['Pclass'] == pclass)]) * 100, 2))
    male = str(round(len(survival[(survival['Pclass'] == pclass) & (survival['Sex'] == 'male')]) / len(train[(train['Pclass'] == pclass)]) * 100, 2))
    p_class.loc[len(p_class)] = pclass, len(train[train['Pclass'] == pclass]), (percentage + ' %'), (female + ' %'), (male + ' %')

# Create histogram 
plt.hist(train['Pclass'], bins=9, rwidth=0.9,  color = 'steelblue', label='Total')
plt.hist(survival['Pclass'], bins=9, rwidth=0.9, color = 'darkseagreen', label='Survived')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.legend(loc='upper left')
plt.grid(axis='x')

# Create comparison charts (Box and swarm plots)
f, axes = plt.subplots(2, 2, figsize=(17, 10))
a1 = sns.boxplot(x='Pclass', y='Age', data=train_with_age, hue='Survived', palette=['steelblue','darkseagreen'], showfliers=False, ax=axes[0, 0])
a2 = sns.swarmplot(x='Pclass', y='Age', data=train_with_age, hue='Survived', palette=['steelblue','darkseagreen'], dodge=True, ax=axes[1, 0])
a3 = sns.boxplot(x='Pclass', y='Fare', data=train, hue='Survived', palette=['steelblue','darkseagreen'], showfliers=False, ax=axes[0, 1])
a4 = sns.swarmplot(x='Pclass', y='Fare', data=train, hue='Survived',palette=['steelblue','darkseagreen'], dodge=True, ax=axes[1, 1])

# Create comparison charts (Box and swarm plots)
b1 = a1.legend(loc='upper right')
b2 = a2.legend(loc='upper right')
b3 = a3.legend(loc='upper right')
b4 = a4.legend(loc='upper right')

plt.show()
display(p_class)
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']
# Merging both datasets (train and test) data
combined = [train, test]

# Loop over each observation to extract the title from the name
for dataset in combined:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
# Now, let's have a look on which titles were more likely to survive.
pd.crosstab(train['Title'], train[train['Survived'] == 1]['Survived'])
# Aggregating the unfrequented titles under (Others)
for dataset in combined:
    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
       'Jonkheer', 'Dona'], 'Others')

# Merging the datasets (train & test) again after the previous amendments
combine = pd.concat(combined, ignore_index=True)
for dataset in combined:
    dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
    
for dataset in combined:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

for dataset in combined:
    dataset['Title'] = dataset['Title'].map( {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Others': 5} ).astype(int)
    
combine = pd.concat(combined, ignore_index=True)
for dataset in combined:
    # Setting counter to mark each "Fare" group with a specific number
    counter = 0
    
    # Grouping "Fares" below 5 dollars
    dataset.loc[dataset['Fare'] <= 5, 'Fare'] = counter
    counter += 1
    
    # Grouping "Fares" between 5 and 30 dollars
    for number in range(5, 30, 5):
        dataset.loc[(dataset['Fare'] > number) & (dataset['Fare'] <= (number + 5)), 'Fare'] = counter
        counter += 1
                                  
    # Grouping "Fares" between 30 and 200 dollars
    for number in range(30, 200, 10):
        dataset.loc[(dataset['Fare'] > number) & (dataset['Fare'] <= (number + 10)), 'Fare'] = counter
        counter += 1
                                        
    # Grouping "Fares" between 200 and 300 dollars
    for number in range(200, 300, 20):
        dataset.loc[(dataset['Fare'] > number) & (dataset['Fare'] <= (number + 20)), 'Fare'] = counter
        counter += 1

    # Grouping "Fares" between 300 and 400 dollars
    for number in range(300, 400, 50):
        dataset.loc[(dataset['Fare'] > number) & (dataset['Fare'] <= (number + 50)), 'Fare'] = counter
        counter += 1
                        
    # Grouping "Fares" above 400 dollars
    dataset.loc[(dataset['Fare'] > 400), 'Fare'] = counter
for dataset in combined:
    # Setting counter to mark each "Age" group with a specific number
    counter = 0

    # Grouping "Ages" below 10 years
    dataset.loc[ dataset['Age'] <= 10, 'Age'] = counter
    counter += 1

    # Grouping "Ages" between 10 and 80 years
    for number in range(10, 80, 10):
        dataset.loc[(dataset['Age'] > number) & (dataset['Age'] <= (number + 10)), 'Age'] = counter
        counter += 1
combine = pd.concat(combined, ignore_index=True)

# Creating predictors (X_train) and dependent (y_train) dataframes containing all observations with existing ages
X_train = combine[combine['Age'].notnull()].loc[:, ['Embarked', 'Family', 'Fare', 'Pclass', 'Sex', 'Title']].values
y_train = combine[combine['Age'].notnull()].loc[:, 'Age'].values

# Creating classifier object using XGBoost
classifier = XGBClassifier(colsample_bytree=0.34, learning_rate=0.1, max_depth=3, min_child_weight=5.01, 
    n_estimators=105, reg_lambda=0.000001, subsample=0.6)

# Fitting the classifier
classifier.fit(X_train, y_train)
# Creating predictors dataframe containing all observations with missing ages
X_age = combine[combine['Age'].isnull()].loc[:, ['Embarked', 'Family', 'Fare', 'Pclass', 'Sex', 'Title']].values

# Predicting missing ages
X_prediction = classifier.predict(X_age)

# Adding the related (PassengerIDs) for all of the missing ages to be able to locate them in the training and test dataframes and replace the corresponding missing ages.
PassengerId = combine[combine['Age'].isnull()].iloc[:, 7].values

for id in (range(0, len(PassengerId))):
    # Looping through the missing ages in train dataframe and replace them with the corresponding predictions obtained from the random forest model. 
    for row in range(0, len(train)):
        if train.iloc[row, 0] == PassengerId[id]:
            train.iloc[row, 5] = X_prediction[id]

    # Looping through the missing ages in test dataframe and replace them with the corresponding predictions obtained from the random forest model.
    for row in range(0, len(test)):
        if test.iloc[row, 0] == PassengerId[id]:
            test.iloc[row, 4] = X_prediction[id]

combine = pd.concat(combined, ignore_index=True)
# Create X, y vectors.
X_train = train.loc[:, ['Embarked', 'Sex', 'Title', 'Pclass', 'Age', 'Fare', 'Family']].values
y_train = train.loc[:, 'Survived'].values

# Create a dataframe that will hold each model's prediction accuracy calculated using cross validation.
accuracy_dataframe = pd.DataFrame(columns=['Model', 'K_Fold_Score'])

# Create function to check the best hyperparameters for each algorithm
def checker (algo, parameters, x, y):
    grid_search = GridSearchCV(estimator = algo, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
    grid_search = grid_search.fit(x, y)
    
    # summarize the result
    print("\nBest score is: %.2f %s using %s\n" % ((round((grid_search.best_score_) * 100, 2)), '%', grid_search.best_params_))
parameters = dict(C= [i/10 for i in range(1, 31)], solver= ['newton-cg', 'lbfgs', 'liblinear'])
checker(LogisticRegression(), parameters, X_train, y_train)
# Fitting Logistic Regression to the Training set.
classifier = LogisticRegression(C = 0.2, solver='newton-cg')
classifier.fit(X_train, y_train)

# Applying K-Fold Cross Validation.
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
K_fold = str(round((accuracy.mean() * 100), 2))

# Adding the name of the model and K_fold result to the accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'Logistic Regression', (K_fold + ' %')

parameters = dict(n_neighbors = range(1,51), metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], p=[1, 2])
checker(KNeighborsClassifier(), parameters, X_train, y_train)
# Fitting the classifier to the Training set
classifier = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 32, p = 1)
classifier.fit(X_train, y_train)

# Applying K-Fold Cross Validation
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
K_fold = str(round((accuracy.mean() * 100), 2))

# Adding the name of the model and K_fold result to the accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'K-NN', (K_fold + ' %')
parameters = dict(C = [i/100 for i in range(100, 131)], kernel = ['linear', 'rbf', 'sigmoid'], gamma = [0.01, 0.1, 1])
checker(SVC(), parameters, X_train, y_train)
# Fitting SVM to the Training set
classifier = SVC( C=1.19, gamma=0.1, kernel = 'rbf')
classifier.fit(X_train, y_train)

# Applying K-Fold Cross Validation
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
K_fold = str(round((accuracy.mean() * 100), 2))

# Adding the name of the model and K_fold result to the accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'SVM', (K_fold + ' %')
# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Applying K-Fold Cross Validation
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
K_fold = str(round((accuracy.mean() * 100), 2))

# Adding the name of the model and K_fold result to the accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'Naive Bayesn', (K_fold + ' %')
parameters = dict(n_estimators=range(60, 76), criterion = ['gini', 'entropy'], max_depth = range(4, 7), random_state=[3])
checker(RandomForestClassifier(), parameters, X_train, y_train)
# Fitting Random Forest to the Training set (The below hyperparameters were the result of the Grid Search if grid search was running on Spyder)
classifier = RandomForestClassifier(criterion = 'entropy', max_depth=4, n_estimators = 61, random_state=3)
classifier.fit(X_train, y_train)

# Applying K-Fold Cross Validation
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
K_fold = str(round((accuracy.mean() * 100), 2))

# Adding the name of the model and K_fold result to the accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'Random Forest', (K_fold + ' %')
parameters = dict(n_estimators=range(80, 101), learning_rate=[0.01],
max_depth=[3], gamma = [0.00001], subsample=[i/10 for i in range(3, 8)], colsample_bytree=[i/10 for i in range(8, 11)],
            reg_lambda=[0.000001])
checker(XGBClassifier(), parameters, X_train, y_train)
# Fitting Random Forest to the Training set (The below hyperparameters were the result of the Grid Search if grid search was running on Spyder)
classifier = XGBClassifier(colsample_bytree=0.8, gamma=1e-05, learning_rate=0.01, max_depth=3, n_estimators=89, 
                          reg_lambda=1e-06, subsample=0.6)
classifier.fit(X_train, y_train)

# Applying K-Fold Cross Validation
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
K_fold = str(round((accuracy.mean() * 100), 2))

# Adding the name of the model and K_fold result to the accuracy_dataframe
accuracy_dataframe.loc[len(accuracy_dataframe)] = 'XGBoost', (K_fold + ' %')
accuracy_dataframe = accuracy_dataframe.sort_values(['K_Fold_Score'], ascending=False)
accuracy_dataframe.reset_index(drop=True)
# Predicting the test data
test_dataset = test.loc[:, ['Embarked', 'Sex', 'Title', 'Pclass', 'Age', 'Fare', 'Family']].values

# Predict the survival of test dataset
prediction = {'PassengerId': test['PassengerId'], 'Survived': classifier.predict(test_dataset)}
# Creating prediction file
solution = pd.DataFrame(prediction)
solution.to_csv('Final_Solution.csv', index=False)