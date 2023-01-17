import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



# Import Competition Datasets

comp_train = pd.read_csv("../input/titanic/train.csv")

comp_test = pd.read_csv("../input/titanic/test.csv")



full = pd.concat([comp_train, comp_test], axis=0, ignore_index=True)



print('Train instances:', comp_train.shape[0])

print('Test instances:', comp_test.shape[0])

print('Number of features:', comp_train.shape[1])



# Drop PassengerId

full.drop('PassengerId', axis=1, inplace=True)



# Take glance at the data set

full.head()
# Calculate proportion of passengers that survived

display(comp_train.Survived.value_counts(normalize=True))
# Calculate Gender proportions of passengers

display(comp_train.Sex.value_counts(normalize=True))

sns.countplot(x='Survived', hue='Sex', data=comp_train)

plt.show()
display(comp_train.groupby('Pclass').Survived.value_counts(normalize=True))

sns.countplot(x='Survived', hue='Pclass', data=comp_train)

plt.show()
sns.kdeplot(full['Age'][full['Survived'] == 1], label='Survived', color='green')

sns.kdeplot(full['Age'][full['Survived'] == 0], label='Perished', color='red'); 
full.info()
# Fill by median

missing_val_median = ['Age', 'Fare']



for col  in missing_val_median:

    full[col] = full[col].fillna(value=full[col].median())

    

# Fill by mode

missing_val_median = ['Embarked']



for col  in missing_val_median:

    full[col] = full[col].fillna(value=full[col].mode()[0])
full['Pclass'] = full['Pclass'].astype(str)

full['TravelAlone'] = np.where(full['SibSp'] + full['Parch'] > 0, '1', '0')

full.drop(['Name', 'Ticket'], axis=1, inplace=True)



full.info()
print(full.var())



sns.kdeplot(full['Fare'])

plt.show()
full['logFare'] = np.log(full['Fare'] + 1)

full['AgeScaled'] = (full['Age'] - full['Age'].mean()) / full['Age'].std()



full.drop(['Age', 'Fare', 'Cabin'], axis=1, inplace=True)
print(full.var())

print(full.describe())
full_dummies = pd.get_dummies(full, drop_first=True)



full_dummies.info()
from sklearn.model_selection import train_test_split



train = full_dummies[full_dummies['Survived'].notnull()]

test = full_dummies[full_dummies['Survived'].isnull()]



print(train.shape)

print(test.shape)



# Create train and test sets

# Stratify along Survived to ensure the split remains representative of the original dataset

X_train, X_test, y_train, y_test = train_test_split(

    train.drop('Survived', axis=1), 

    train['Survived'], 

    test_size=0.2, 

    stratify=train['Survived'],

    random_state=12)



print(y_train.value_counts(normalize=True))

print(y_test.value_counts(normalize=True))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



logreg = LogisticRegression(max_iter=3000)



# Define parameter search grid

parameters = [#{'penalty': ['elasticnet'], 'l1_ratio': np.linspace(0,1,10), 'C': [0.01, 0.1, 0.1, 1, 10], 'solver': ['saga']},

              {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 0.1, 1, 10], 'solver': ['liblinear']},

              {'penalty': ['none']}]



# Pass the pipeline and parameter dict to GridSearchCV

cv = GridSearchCV(logreg, param_grid=parameters, cv=5)



# Fit model to the training set

cv.fit(X_train, y_train)



# View best cross-validated fit

print(cv.best_params_)

print(cv.best_score_)



# Check training set accuracy

print('Training set accuracy:', cv.score(X_train, y_train))

# Check test set accuracy

print('Test set accuracy:', cv.score(X_test, y_test))
from sklearn.model_selection import cross_val_score



out_logreg = LogisticRegression(penalty=cv.best_params_['penalty'], C=cv.best_params_['C'], solver=cv.best_params_['solver'], max_iter=800)



# Obtain Cross-Validated Accuracy for the model on the entire training set

cross_val_score(out_logreg, X=train.drop('Survived', axis=1), y=train['Survived'], cv=5)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()



# Define parameter search grid

rf_parameters = [{'n_estimators': [100, 200, 250, 300, 400], 'criterion': ['gini', 'entropy']}]



# Pass the pipeline and parameter dict to GridSearchCV

rf_cv = GridSearchCV(rfc, param_grid=rf_parameters, cv=10)



# Fit model to the training set

rf_cv.fit(X_train, y_train)



# View best cross-validated fit

print(rf_cv.best_params_)

print(rf_cv.best_score_)



# Check training and test set accuracy

print('Training set accuracy:', rf_cv.score(X_train, y_train))

print('Test set accuracy:', rf_cv.score(X_test, y_test))
out_rf = RandomForestClassifier(n_estimators=rf_cv.best_params_['n_estimators'], criterion=rf_cv.best_params_['criterion'])



# Obtain Cross-Validated Accuracy for the model on the entire training set

cross_val_score(out_rf, X=train.drop('Survived', axis=1), y=train['Survived'], cv=10)
# Refit model to the entire training set and make predictions

out_rf.fit(train.drop('Survived', axis=1), train['Survived'])



y_pred = out_rf.predict(test.drop('Survived', axis=1))
sub = pd.DataFrame()

sub['PassengerId'] = comp_test['PassengerId']

sub['Survived'] = y_pred

sub['Survived'] = sub['Survived'].astype(int)

sub.to_csv('cv_logreg_submission.csv', index=False)
print(sub.info())

display(sub.head())