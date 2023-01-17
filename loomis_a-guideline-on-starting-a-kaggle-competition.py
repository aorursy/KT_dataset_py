#Import Necessary Packages:


import csv
import sys
import timeit

#Data wrangling:
import pandas as pd
import numpy as np

#Visualization:
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Imputer
from impyute.imputation.cs import mice, fast_knn

# Imports for Modelling
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import accuracy_score, make_scorer
# Changing the seaborn style for better visibility in the notebook
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
#Read in the necessary data:
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
#Data size:
print(train.shape)
print(test.shape)
#What are the variables?
print(train.columns)
train.Survived.value_counts()
#Variable of interest: Survived
print(train['Survived'].head())
print(train['Survived'].describe())
train.Survived.hist()
plt.show()
#Describe the train data
train.describe()
# First idea: Different Ticket classes may have different chances of survival because of the placement of the cabins on the ship
sns.countplot(x = 'Pclass', data = train, hue = 'Survived')
plt.show()
# Did passengers from different entry ports have different survival rates?
g = sns.countplot(x = 'Embarked', hue = 'Survived', data = train)
g.set_xticklabels(['Southhampton','Cherbourg', 'Queenstown'])
plt.show()
# How is the age distribution?
sns.catplot(y = 'Age', x = 'Survived',kind = 'box', data = train)
plt.show()
# How is the distribution between males and females?
sns.countplot(x = 'Sex', hue = 'Survived', data = train)
plt.show()
# Relationship between Number of siblings and parents and the ticket categorie
sns.barplot(x = 'Pclass', y= 'SibSp', hue = 'Survived', data = train)
sns.barplot(x = 'Pclass', y = 'Parch', hue = 'Survived', data = train)
plt.show()
print(train.corr())
# Create age classes:
sns.distplot(train['Age'], bins = 16)
plt.show()
# Age groups:
def calc_age_group(data):
    data['Age_groups'] = pd.cut(data.Age, [0, 15, 40, 60, 81], right=False, labels=[0, 1, 2, 3])

    return data
# Categorical to Binary
def bin_to_cat(data):
    data = pd.get_dummies(data, prefix=['Sex', 'Pclass', 'Embarked'], drop_first=True,
                          columns=['Sex', 'Pclass', 'Embarked'])

    return data
# Check for missing variables:
print('Missing variables in the training set:')
print(train.isna().sum())
print('Missing variables in the testing set:')
print(test.isna().sum())
def drop_missing_values(data):
    # Drop name column
    data = data.drop('Name', axis=1)
    # Missing values:
    # Cabin - 687 missing values - drop column
    data = data.drop('Cabin', axis=1)

    # Embarked - 2 missing values -> drop rows
    data = data.dropna(axis=0, subset=['Embarked'])
    return data
def impute_knn(tr, ts):
    sys.setrecursionlimit(100000)
    imputed_tr = pd.DataFrame(fast_knn(tr.values), columns=tr.columns)
    imputed_ts = pd.DataFrame(fast_knn(ts.values), columns=ts.columns)

    return imputed_tr, imputed_ts


def impute_mice(tr, ts):
    imputed_tr = pd.DataFrame(mice(tr.values), columns=tr.columns)
    imputed_ts = pd.DataFrame(mice(ts.values), columns=ts.columns)

    return imputed_tr, imputed_ts
grouped_class = train.groupby('Pclass')['Age']
first = grouped_class.get_group(1)
second = grouped_class.get_group(2)
third = grouped_class.get_group(3)
first.hist(alpha = 0.3, label = 'first')
#second.hist(alpha = 0.3, label = 'second')
third.hist(alpha = 0.3, label = 'third')
plt.legend()
plt.show()
# Install Impyute
!pip install impyute
import csv
import sys
import numpy as np
import pandas as pd
import timeit

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Imputer
from impyute.imputation.cs import mice, fast_knn

# Imports for Modelling
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import accuracy_score, make_scorer



# Feature Engineering
def drop_missing_values(data):
    # Drop name column
    data = data.drop('Name', axis=1)
    # Missing values:
    # Cabin - 687 missing values - drop column
    data = data.drop('Cabin', axis=1)

    # Embarked - 2 missing values -> drop rows
    data = data.dropna(axis=0, subset=['Embarked'])
    return data


# Ticket:
def single_ticket(data):
    count = data.Ticket.value_counts()
    data = data.assign(Ticket_count=0)
    for index, row in data.iterrows():
        tik = row['Ticket']
        num = count[tik]
        data.loc[index, ['Ticket_count']] = num

    data = data.drop('Ticket', axis=1)
    return data


# Categorical to Binary
def bin_to_cat(data):
    data = pd.get_dummies(data, prefix=['Sex', 'Pclass', 'Embarked'], drop_first=True,
                          columns=['Sex', 'Pclass', 'Embarked'])

    return data


# Age:
# Missing values:
def impute_knn(tr, ts):
    sys.setrecursionlimit(100000)
    imputed_tr = pd.DataFrame(fast_knn(tr.values), columns=tr.columns)
    imputed_ts = pd.DataFrame(fast_knn(ts.values), columns=ts.columns)

    return imputed_tr, imputed_ts


def impute_mice(tr, ts):
    imputed_tr = pd.DataFrame(mice(tr.values), columns=tr.columns)
    imputed_ts = pd.DataFrame(mice(ts.values), columns=ts.columns)

    return imputed_tr, imputed_ts


# Age groups:
def calc_age_group(data):
    data['Age_groups'] = pd.cut(data.Age, [0, 15, 40, 60, 81], right=False, labels=[0, 1, 2, 3])

    return data


def pipeline(data):
    data = drop_missing_values(data)
    data = single_ticket(data)
    data = bin_to_cat(data)

    return data


def removeIds(train, test):
    train_ids = train['PassengerId']
    test_ids = test['PassengerId']
    train = train.drop(['PassengerId'], axis=1)
    test = test.drop(['PassengerId'], axis=1)
    return train_ids, test_ids, train, test


train_ids, test_ids, train, test = removeIds(train, test)

train = pipeline(train)
test = pipeline(test)

tr_mice, ts_mice = impute_mice(train, test)


# tr_knn, ts_knn = impute_knn(train, test)

def scale(train, test):
    X_train = train.drop(['Survived'], axis=1)
    y_train = train[['Survived']]
    Scaler = StandardScaler()
    Scaler.fit(X_train)
    X_train_scaled = Scaler.transform(X_train)
    X_test_scaled = Scaler.transform(test)
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)

    return X_train_scaled, y_train, X_test_scaled


X_train_scaled, y_train, X_test_scaled = scale(tr_mice, ts_mice)


# Modelling

def createModel():
    RandomForest = RandomForestClassifier(random_state=123)
    GradientBoost = GradientBoostingClassifier(random_state=123)
    AdaBoost = AdaBoostClassifier(random_state=123)
    naive_bayes = GaussianNB()
    LogisitcR = LogisticRegression(random_state=123)
    svc = SVC(random_state=123)

    param_grid = []

    Voter = VotingClassifier(
        estimators=[('RandomForest', RandomForest), ('GradientBoost', GradientBoost),
                    ('Adaboost', AdaBoost)],
        voting='soft')

    First_layer = [('LogisticR', LogisitcR), ('NaiveB', naive_bayes), ('SVC', svc)]

    Stack = StackingClassifier(estimators=First_layer, final_estimator=Voter)
    return Stack


model = createModel()


# Create Train, Test & Validation set
def createSplits(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    return X_train, X_val, y_train, y_val


X_train, X_val, y_train, y_val = createSplits(X_train_scaled, y_train)

# Hyperparameter Tuning
params = {'final_estimator__RandomForest__max_depth': [4, 6, 8, 10, 20],
          'final_estimator__RandomForest__n_estimators': [50, 100, 200, 500],
          'final_estimator__GradientBoost__max_depth': [4, 6, 8, 10, 20],
          'final_estimator__GradientBoost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
          'final_estimator__GradientBoost__n_estimators': [50, 100, 200, 500],
          'final_estimator__Adaboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
          'final_estimator__Adaboost__n_estimators': [50, 100, 200, 500],

          }


def rndSearch(X_train, y_train, model, params, n):
    rndSearch = RandomizedSearchCV(model, param_distributions=params, n_iter=n, cv=5, n_jobs=-1, scoring='accuracy')
    rndSearch.fit(X_train, y_train)
    best_model = rndSearch.best_estimator_
    best_params = rndSearch.best_params_
    train_score = rndSearch.best_score_

    return best_model, best_params, train_score


# start = timeit.default_timer() - Evaluate the runtime of the random search
best_model, best_params, train_score = rndSearch(X_train, y_train.values, model, params, 50)
# stop = timeit.default_timer()
# print('Time: ', stop - start)

# Training - Evaluation
def evaluate_stratified(model, train):
    fold_metrics = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    for train_index, test_index in skf.split(train, train['Survived']):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        # Train a model
        model.fit(cv_train.drop('Survived', axis=1), cv_train['Survived'])
        # Make predictions
        predictions = model.predict(cv_test.drop('Survived', axis=1))
        # Calculate the metric
        metric = accuracy_score(cv_test['Survived'], predictions)
        fold_metrics.append(metric)
    # Compute overall evaluation score
    score = np.mean(fold_metrics) + np.std(fold_metrics)
    std = np.std(fold_metrics)
    return score, std


# train_score, train_score_std = evaluate_stratified(model, pd.concat([X_train, y_train], axis=1))


# Evaluation on Validation Set:

def validation(model, X_val, y_val):
    prediction = model.predict(X_val)
    metric = accuracy_score(y_val, prediction)
    return metric


#val_score = validation(best_model, X_val, y_val)


# Submission

def return_submission(model, X_test, test_id):
    submission = pd.DataFrame(model.predict(X_test), columns=['Survived'])
    submission = pd.concat([test_id, submission], axis=1)

    return submission


# submission = return_submission(model, X_test_scaled, test_ids)


def submission_to_csv(submission, filename):
    submission = submission.astype(int)
    submission.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print('Finished!')

# submission_to_csv(submission, 'model_1.csv')
