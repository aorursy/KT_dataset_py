import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import platform

import warnings



from xgboost import XGBClassifier



from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, VotingClassifier)

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.feature_selection import SelectFromModel



from sklearn.preprocessing import (StandardScaler, OrdinalEncoder)



from sklearn.metrics import (classification_report, confusion_matrix)



from sklearn.model_selection import (GridSearchCV, learning_curve,

                                     cross_val_score, train_test_split)





warnings.filterwarnings('ignore')



print('Python version: {}'.format(platform.python_version()))

print('NumPy version: {}'.format(np.__version__))

print('pandas version: {}'.format(pd.__version__))
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
total_num = train_data.shape[0] + test_data.shape[0]

test_pct = round(test_data.shape[0] * 100 / total_num)



print('Number of entries in the training dataset: {}'.format(train_data.shape[0]))

print('Number of entries in the test dataset: {}'.format(test_data.shape[0]))



print('Percentage of entries for testing: {}%'.format(test_pct))
train_data.head()
# List of features to view descriptive statistics

features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
train_data[features].describe()
train_data[features].describe(include=['O'])
test_data.head()
test_data[features].describe()
test_data[features].describe(include=['O'])
train_data['Survived'].value_counts(normalize=True)
train_data.isna().sum()
def num_isna(feature):

    return round(train_data[feature].isna().sum() * 100 / train_data.shape[0], 3)



features_na = train_data.columns[train_data.isna().any()].tolist()



for feature in features_na:

    print('Percentage of NA values for {}: {}%'.format(feature, num_isna(feature)))
test_data.isna().sum()
features_na = test_data.columns[test_data.isna().any()].tolist()



for feature in features_na:

    print('Percentage of NA values for {}: {}%'.format(feature, num_isna(feature)))
sns.catplot(x='Sex', y='Survived', kind='bar', data=train_data)
sns.distplot(train_data['Age'])
sns.countplot(train_data['Parch'])
sns.countplot(train_data['SibSp'])
sns.countplot(train_data['Pclass'])
sns.catplot(x='Sex', y='Survived', hue='Pclass', kind='point', data=train_data)
sns.catplot(x='Sex', y='Survived', hue='Embarked', kind='bar', data=train_data)
sns.catplot(x='Parch', y='Survived', hue='Sex', kind='point', data=train_data)

sns.catplot(x='SibSp', y='Survived', hue='Sex', kind='point', data=train_data)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train_data)
sns.catplot(x='Pclass', y='Age', hue='Sex', kind='swarm', data=train_data)
sns.catplot(x='Embarked', y='Age', hue='Sex', kind='swarm', data=train_data)

sns.catplot(x='Embarked', y='Age', hue='Pclass', kind='swarm', data=train_data)
sns.catplot(x='Sex', y='Fare', hue='Survived', kind='boxen', data=train_data)

sns.catplot(x='Pclass', y='Fare', hue='Survived', kind='boxen', data=train_data)

sns.catplot(x='Embarked', y='Fare', hue='Survived', kind='boxen', data=train_data)
sns.relplot(x='Age', y='Fare', hue='Survived', size='Pclass', data=train_data)
features_na = ['Cabin', 'Age', 'Embarked', 'Fare']
# We calculate the values and apply the tranpose to invert the rows/columns for easier manipulation

train_data_stats = train_data.describe().T

test_data_stats = test_data.describe().T
train_data_stats.head()
test_data_stats.head()
def age_dist(df, stats):

    return np.random.randint(stats.at['Age', 'mean'] - stats.at['Age', 'std'],

                           stats.at['Age', 'mean'] + stats.at['Age', 'std'],

                           size=df['Age'].isna().sum())





train_data['Cabin'] = train_data['Cabin'].fillna('N')



train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])



train_data['Age'][train_data['Age'].isna()] = age_dist(train_data, train_data_stats)
train_data.isna().sum()
test_data[test_data['Fare'].isna()]
train_data[train_data['Pclass'] == 3].describe()
test_data['Cabin'] = test_data['Cabin'].fillna('N')



test_data['Fare'] = test_data['Fare'].fillna(test_data[test_data['Pclass'] == 3]['Fare'].median())



test_data['Age'][test_data['Age'].isna()] = age_dist(test_data, test_data_stats)
test_data.isna().sum()
pd.cut(train_data['Age'].astype(int), 5).cat.categories
pd.qcut(train_data['Fare'], 4).cat.categories
for df in [train_data, test_data]:

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



    df.loc[ df['Age'] <= 16, 'Age'] = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[df['Age'] > 64, 'Age'] = 4

    df['Age'] = df['Age'].astype(int)



    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2

    df.loc[df['Fare'] > 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)
train_data.head()
for df in [train_data, test_data]:

    df['Surname'] = df['Name'].str.split(',')

    df['Surname'] = df['Surname'].apply(lambda x: list(x)[0])

    df['Family'] = df.agg('{0[Surname]}:{0[FamilySize]}'.format, axis=1)
train_data.head()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ''



for df in [train_data, test_data]:

    df['Title'] = df['Name'].apply(get_title)



    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')
for df in [train_data, test_data]:

    df['AdultMale'] = ((df['Age'] > 0) & (df['Sex'] == 'male')).astype(int)
for df in [train_data, test_data]:

    df['Deck'] = df['Cabin'].str[0]

    df['Cabin_Extra'] = df['Cabin'].str.contains(' ').astype(int)
train_data[train_data['Cabin'].str.startswith('F ')]
train_data[train_data['Cabin'] != 'N']['Cabin'].values
train_data.groupby('Deck')['Survived'].value_counts(normalize=True).unstack()
sns.catplot(x='Deck', y='Survived', kind='bar', data=train_data)
train_data.groupby(['Cabin_Extra'])['Survived'].value_counts(normalize=True).unstack()
train_data.groupby(['Sex', 'Deck'])['Survived'].value_counts(normalize=True).unstack()
train_data.groupby(['Sex', 'Deck', 'Cabin_Extra'])['Survived'].value_counts(normalize=True).unstack()
train_data.head()
test_data.head()
plt.figure(figsize=(14,12))

sns.heatmap(train_data.corr(), annot=True)

plt.show()
def encode_features(df):

    df['Deck'] = df['Deck'].apply(ord)



    return df



for df in [train_data, test_data]:

    df = encode_features(df)
train_data = pd.get_dummies(train_data, columns=['Sex', 'Pclass', 'Embarked', 'Title', 'Age', 'Fare'])

test_data = pd.get_dummies(test_data, columns=['Sex', 'Pclass', 'Embarked', 'Title', 'Age', 'Fare'])
PassengerId = test_data['PassengerId']
drop_elements = ['PassengerId', 'Name', 'Family', 'Ticket', 'Cabin', 'Cabin_Extra', 'Surname', 'Family']



train_data = train_data.drop(drop_elements, axis=1)

test_data = test_data.drop(drop_elements, axis=1)
train_data.head()
X = train_data.drop('Survived', axis=1)

y = train_data['Survived']

X_test = test_data.copy()
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# Copy dataframe to later inspect and compare true values and predictions

X_validate_df = X_validate.copy()



standard_scaler = StandardScaler()

standard_scaler.fit(X_train)



X_train = standard_scaler.transform(X_train)

X_validate = standard_scaler.transform(X_validate)
model_results = pd.DataFrame(columns=['Score', 'Cross-validation score'])
# Parameters obtained with grid search

params = {'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}



random_forest = RandomForestClassifier(random_state=0, n_jobs=-1).fit(X_train, y_train)



score = random_forest.score(X_train, y_train)

cross_val_score_mean = cross_val_score(random_forest, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Random Forest'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
decision_tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)



score = decision_tree.score(X_train, y_train)

cross_val_score_mean = cross_val_score(decision_tree, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Decision Tree'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
# Parameters obtained with grid search

params = {'C': 10, 'dual': True, 'penalty': 'l2', 'solver': 'liblinear', 'tol': 0.0001}



logistic_regression = LogisticRegression(random_state=0, n_jobs=-1).fit(X_train, y_train)



score = logistic_regression.score(X_train, y_train)

cross_val_score_mean = cross_val_score(logistic_regression, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Logistic Regression'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
# Parameters obtained with grid search

params = {'n_neighbors': 8, 'weights': 'uniform'}



knn = KNeighborsClassifier().fit(X_train, y_train)



score = knn.score(X_train, y_train)

cross_val_score_mean = cross_val_score(knn, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['K-nearest Neighbors'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
gaussian_nb = GaussianNB().fit(X_train, y_train)



score = gaussian_nb.score(X_train, y_train)

cross_val_score_mean = cross_val_score(gaussian_nb, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Gaussian Naive Bayes'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
# Parameters obtained with grid search

params = {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}



svc = SVC(random_state=0).fit(X_train, y_train)



score = svc.score(X_train, y_train)

cross_val_score_mean = cross_val_score(svc, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Support Vector Machine'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
# Parameters obtained with grid search

params = {'activation': 'tanh', 'early_stopping': False, 'learning_rate': 'constant', 'learning_rate_init': 0.001}



mlp = MLPClassifier(early_stopping=True, random_state=0, max_iter=300).fit(X_train, y_train)



score = mlp.score(X_train, y_train)

cross_val_score_mean = cross_val_score(mlp, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Multi-layer Perceptron'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
# Parameters obtained with grid search

params = {'colsample_bytree': 0.7, 'max_depth': 5, 'n_estimators': 200,'objective': 'binary:logistic', 'reg_alpha': 1.2, 'reg_lambda': 1.3, 'subsample': 0.8}



xgboost_model = XGBClassifier(random_state=0, nthread=-1).fit(X_train, y_train)



score = xgboost_model.score(X_train, y_train)

cross_val_score_mean = cross_val_score(xgboost_model, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['XGBoost'] = [score, cross_val_score_mean]



print([score, cross_val_score_mean])
model_results
selected_estimator = svc
use_feature_selection = False



selected_features = None



if use_feature_selection:

    selector = SelectFromModel(estimator=selected_estimator)

    selector = selector.fit(X_train, y_train)



    selected_features = X.columns[selector.get_support()].to_list()



    print('Best features: {}'.format(selected_features))



    X_train = selector.transform(X_train)

    X_validate = selector.transform(X_validate)
# hyper parameters for Random Forest

forest_grid = {

  'n_estimators': [100, 200, 500, 1000],

  'criterion': ['gini', 'entropy'],

  'max_depth': [2, 4, 6, 8, None],

  'min_samples_split': [5, 2, 1, 0.2, .05],

  'min_samples_leaf': [5, 1, 0.2, .05],

}



# hyper parameters for K-nearest Neighbors

knn_grid = {'n_neighbors': list(range(1,11)), 'weights': ['uniform', 'distance']}



# hyper parameters for Support Vector Machine

svm_grid = {'C':[1,10,100,1000], 'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}



# hyper parameters for Logistic Regression

logistic_grid = {

  'C':[1,10,100,1000], 'penalty':['l1','l2','elasticnet'],

  'dual': [False, True], 'tol':[1e-4, 1e-5],

  'solver': ['newton-cg','lbfgs','liblinear','sag','saga']

}



# hyper parameters for Multi-layer Perceptron

mlp_grid = {

  'activation':['identity','logistic','tanh','relu'],

  'learning_rate': ['constant','invscaling','adaptive'],

  'learning_rate_init': [0.003, 0.001, 0.0001],

  'early_stopping':[False, True]

}



# hyper parameters for XGBoost

xgboost_grid = {

  'n_estimators': [200, 500, 1000, 2000],

  'colsample_bytree': [0.7, 0.8],

  'max_depth': [5, 10, 15, 20],

  'reg_alpha': [1.1, 1.2, 1.3],

  'reg_lambda': [1.1, 1.2, 1.3],

  'subsample': [0.7, 0.8, 0.9],

  'objective': ['binary:logistic'],

}
def grid_search(estimator, param_grid, X_train, X_test, y_train, y_test):



    # We can re-run the grid search with the other parameter grids

    tune_model = GridSearchCV(estimator, param_grid=param_grid, cv=10, n_jobs=-1)

    tune_model.fit(X_train, y_train)



    print(type(estimator))



    print("\nGrid scores on development set:\n")



    means = tune_model.cv_results_['mean_test_score']

    stds = tune_model.cv_results_['std_test_score']



    print("%0.3f (+/-%0.03f) for %r\n" % 

        (means[tune_model.best_index_], stds[tune_model.best_index_] * 2, tune_model.cv_results_['params'][tune_model.best_index_]))



    print("Detailed classification report:\n")

    y_pred = tune_model.predict(X_test)



    print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))



    return tune_model.best_estimator_
selected_estimator = grid_search(selected_estimator, svm_grid, X_train, X_validate, y_train, y_validate)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(facecolor=(1, 1, 1), figsize=(12, 8))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

      estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                    train_scores_mean + train_scores_std, alpha=0.1,

                    color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                    test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

            label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

            label="Cross-validation score")



    plt.legend(loc="best")



    return plt
plot_learning_curve(selected_estimator, 'Learning Curve', X_train, y_train, (0.7, 1.01), cv=10, n_jobs=-1)

plt.show()
voting_estimators = [('rf', random_forest), ('lreg', logistic_regression), ('knn', knn), ('svm', svc), ('mlp', mlp), ('xgboost', xgboost_model)]



voting = VotingClassifier(voting_estimators, 'hard').fit(X_train, y_train)



score = voting.score(X_train, y_train)

cross_val_score_mean = cross_val_score(voting, X_train, y_train, cv=10, n_jobs=-1).mean()



model_results.loc['Voting Hard'] = [score, cross_val_score_mean]



model_results
final_score = cross_val_score(selected_estimator, X_train, y_train, cv=10)



print('Accuracy: {:.3f} (+/- {:.2f})'.format(final_score.mean(), final_score.std() * 2))
y_pred = selected_estimator.predict(X_validate)



# Confusion matrix for the Selected Model

sns.heatmap(confusion_matrix(y_validate, y_pred), annot=True, fmt='d', cmap='Blues')
final_score = cross_val_score(voting, X_train, y_train, cv=10)



print('Accuracy: {:.3f} (+/- {:.2f})'.format(final_score.mean(), final_score.std() * 2))
y_pred = voting.predict(X_validate)



# Confusion matrix for the Voting Hard Model

sns.heatmap(confusion_matrix(y_validate, y_pred), annot=True, fmt='d', cmap='Blues')
y_pred = selected_estimator.predict(X_validate)
X_validate_df['Survived'] = y_validate

X_validate_df['Prediction'] = y_pred
X_validate_df[(X_validate_df['Survived'] != X_validate_df['Prediction']) & (X_validate_df['Survived'] == 1)].head()
X_validate_df[(X_validate_df['Survived'] != X_validate_df['Prediction']) & (X_validate_df['Survived'] == 0)].head()
if use_feature_selection:

    X = X[selected_features]

    X_test = X_test[selected_features]
X = standard_scaler.fit_transform(X)

X_test = standard_scaler.transform(X_test)