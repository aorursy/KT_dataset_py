from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import re
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
def write_to_submission_file(predicted_labels, out_file, train_num=891,
                    target='Survived', index_label="PassengerId"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(train_num + 1,
                                                  train_num + 1 +
                                                  predicted_labels.shape[0]),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
X = pd.read_csv("../input/train.csv")
y = X['Survived']
Z = pd.read_csv("../input/test.csv")
X_orig = X.copy()
Z_orig = Z.copy()
X.head()
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Embarked'].fillna('S', inplace=True)

Z['Age'].fillna(Z['Age'].median(), inplace=True)
Z['Fare'].fillna(Z['Fare'].median(), inplace=True)
X = pd.concat([X, pd.get_dummies(X['Pclass'], prefix="PClass"),
                  pd.get_dummies(X['Sex'], prefix="Sex"),
                  pd.get_dummies(X['SibSp'], prefix="SibSp"),
                  pd.get_dummies(X['Parch'], prefix="Parch"),
                  pd.get_dummies(X['Embarked'], prefix="Embarked")], axis=1)

Z = pd.concat([Z, pd.get_dummies(Z['Pclass'], prefix="PClass"),
                  pd.get_dummies(Z['Sex'], prefix="Sex"),
                  pd.get_dummies(Z['SibSp'], prefix="SibSp"),
                  pd.get_dummies(Z['Parch'], prefix="Parch"),
                  pd.get_dummies(Z['Embarked'], prefix="Embarked")], axis=1)
X.drop(['Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1, inplace=True)
Z.drop(['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Parch_9'], axis=1, inplace=True)
X.head()
X.shape
Z.head()
Z.shape
# Binning Age and Fare

X['Age_cat'] = pd.qcut(X.Age, q=4, labels=False)

X.drop(['Age'], axis=1, inplace=True)


Z['Age_cat'] = pd.qcut(Z.Age, q=4, labels=False)

Z.drop(['Age'], axis=1, inplace=True)


X['Fare_cat'] = pd.qcut(X.Fare, q=4, labels=False)

X.drop(['Fare'], axis=1, inplace=True)


Z['Fare_cat'] = pd.qcut(Z.Fare, q=4, labels=False)

Z.drop(['Fare'], axis=1, inplace=True)
# Use Cabin

X['has_Cabin'] = ~X_orig.Cabin.isnull()


Z['has_Cabin'] = ~Z_orig.Cabin.isnull()
# Extract Title

X_title = pd.DataFrame(index = X_orig.index)

X_title['Title'] = X_orig.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

X_title['Title'] = X_title['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
X_title['Title'] = X_title['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                             'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

X = pd.concat([X, pd.get_dummies(X_title['Title'], prefix='Title')], axis=1)


Z_title = pd.DataFrame(index = Z_orig.index)

Z_title['Title'] = Z_orig.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

Z_title['Title'] = Z_title['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
Z_title['Title'] = Z_title['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                             'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

Z = pd.concat([Z, pd.get_dummies(Z_title['Title'], prefix='Title')], axis=1)
X.head().T
Z.head().T
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
decision_tree_params = {'max_depth': list(range(1, 5)),
                        'min_samples_leaf': list(range(1, 5))}

decision_tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=17),
                                  decision_tree_params, verbose=True, n_jobs=-1, cv=skf)
decision_tree_grid.fit(X, y)

print('Best decision tree params:', decision_tree_grid.best_params_)
print('Best decision tree cross validation score:', decision_tree_grid.best_score_)
decision_tree_predictions = decision_tree_grid.best_estimator_.predict(Z)

write_to_submission_file(decision_tree_predictions, 'titanic_decision_tree.csv')
random_forest_params = {'max_features': list(range(18, 22)),
                        'max_depth': [1, 2, 3, 4, 5, 6]}

random_forest = RandomForestClassifier(n_estimators=300, random_state=17,
                                       oob_score=True, class_weight='balanced', n_jobs=-1)

random_forest_grid = GridSearchCV(random_forest, random_forest_params, verbose=True, n_jobs=-1, cv=skf)
random_forest_grid.fit(X, y)

print('Best random tree params:', random_forest_grid.best_params_)
print('Best random tree cross validation score:', random_forest_grid.best_score_)
random_forest_grid.best_estimator_
score = cross_val_score(random_forest_grid.best_estimator_, X, y, cv=skf)
score
np.std(score)
random_forest_predictions = random_forest_grid.best_estimator_.predict(Z)

write_to_submission_file(random_forest_predictions, 'titanic_random_forest.csv')
c_values = np.logspace(-2, 3, 500)

logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1, n_jobs=-1)
logit_searcher.fit(X, y)

print('Best C:', logit_searcher.C_)
logit_predictions = logit_searcher.predict(Z)

write_to_submission_file(logit_predictions, 'titanic_logit.csv')