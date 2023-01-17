import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display



# Pretty display for notebooks

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv', sep=',')

display(train.head())



test = pd.read_csv('../input/test.csv', sep=',')

display(test.head())
def describe_dataset(dataset, threshold=0.90):

    ds = dataset.isnull().sum(axis=0).reset_index()

    ds.columns = ['feature_name', 'missing_count']

    ds['missing_ratio'] = ds['missing_count'] / dataset.shape[0]

    return ds
missing_data = describe_dataset(train)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds)
missing_data = describe_dataset(test)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds)
fig = plt.figure(figsize=(8,8))

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, scale="count")

plt.show()
fig = plt.figure(figsize=(8,8))

sns.violinplot(x="Sex", y="Pclass", hue="Survived", data=train, split=True, scale="count")

plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(x="Embarked", y="Pclass", hue="Survived", data=train, split=True, scale="count")

plt.show()
features = list(train.columns.values)

features.remove('PassengerId')

features.remove('Name')

corr = train.loc[:, features].corr(method='spearman')



ds = corr.sort_values('Survived', ascending=False)

display(ds['Survived'])
corralation_matrix = corr.round(2)



fig = plt.figure(figsize=(12,12));

sns.heatmap(corralation_matrix, annot=True, center=0, cmap = sns.diverging_palette(250, 10, as_cmap=True), ax=plt.subplot(111))

plt.show()
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()



def encode_string_labels(dataset, target):

    nullvalues = dataset[target].isnull()

    dataset[target] = dataset[target].astype(str)

    encoder = LabelEncoder()

    

    dataset[target] = encoder.fit_transform(dataset[target].values)

    # restore the NaN values

    dataset.loc[nullvalues, target] = np.nan



features_to_be_encoded = [

    'Sex',

    'Embarked'

]



for feature in features_to_be_encoded:

    encode_string_labels(train, feature)
# helper function to impute missing age by class and sex

def impute_age(dataset):

    for pclass in [1,2,3]:

        for sex in [0,1]:

            ds = dataset[dataset['Pclass'] == pclass]

            ds = ds[ds['Sex'] == sex]

            median = ds['Age'].median()

            dataset.loc[

                (dataset['Age'].isnull()) &

                (dataset['Pclass'] == pclass) &

                (dataset['Sex'] == sex),

                'Age'] = median
impute_age(train)

train.loc[(train['Embarked'].isnull()), 'Embarked'] = train['Embarked'].median()
ds = train[train['Survived'] == 1]

display(ds.describe())
target = 'Survived'



features = [

    'Pclass',

    'Sex',

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'Embarked'

]
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
def train_model(model, features, target):

    # split the dataset into training and test data

    X = train.loc[:, features]

    y = train[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # train the model and create predictions for the test data

    model.fit(X_train, y_train)  

    predictions_test = model.predict(X_test)

    score = accuracy_score(y_test, predictions_test)

    print('accuracy: {}'.format(score))
def display_feature_importance(model, features, visualise=False):

    ds = pd.DataFrame()

    ds['features'] = features

    ds['importance'] = model.feature_importances_

    ds = ds.sort_values('importance', ascending=False)

    if visualise:

        fig, ax = plt.subplots(figsize=(18,8))

        sns.set_color_codes('muted')

        sns.barplot(x='importance', y='features', data=ds, label='Weight', color='b')

        ax.legend(ncol=2, loc='lower right', frameon=True)

        #ax.set(xlim=(0, 0.3), ylabel='', xlabel='Feature importance')

        sns.despine(left=True, bottom=True)

        plt.show()

    display(ds)
clf = GradientBoostingClassifier(random_state=33)

train_model(clf, features, target)

display_feature_importance(clf, features, visualise=True)
tuning = False

if tuning:

    param_grid = {

        'n_estimators': range(50, 301, 50),

        'learning_rate': [0.05, 0.02, 0.01],

        'min_samples_split': range(50, 201, 50),

        'max_depth': [2, 3, 4, 5, 6],

        'max_features': ['sqrt']

    }



    clf = GradientBoostingClassifier(random_state=33)

    grid = GridSearchCV(clf, param_grid=param_grid, cv=ShuffleSplit(train_size=0.80, n_splits=10, random_state=1), verbose=1, n_jobs=4)

    train_model(grid, features, target)

    

    display(grid.best_params_)
clf = GradientBoostingClassifier(

    learning_rate=0.02,

    max_depth=3,

    max_features='sqrt',

    min_samples_split=200,

    n_estimators=250,

    random_state=33)

train_model(clf, features, target)

display_feature_importance(clf, features, visualise=True)
X = train.loc[:, features]

y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



errors = pd.DataFrame()

errors['accuracy'] = [accuracy_score(y_test, y_pred) for y_pred in clf.staged_predict(X_test)]
best_n_estimators = np.argmax(errors['accuracy'])



plt.figure(figsize=(18,8))

plt.axvline(best_n_estimators, color='r')

plt.scatter(range(errors.shape[0]), errors['accuracy'].values)

plt.ylabel('Accuracy')

plt.xlabel('Estimators')

plt.show()
clf = GradientBoostingClassifier(

    learning_rate=0.02,

    max_depth=3,

    max_features='sqrt',

    min_samples_split=200,

    n_estimators=250,

    random_state=33)

train_model(clf, features, target)

display_feature_importance(clf, features, visualise=True)
def encode_titles(dataset):

    dataset.loc[dataset['Name'].str.contains('Mr.'), 'Title'] = 'Mr'

    dataset.loc[dataset['Name'].str.contains('Mrs.'), 'Title'] = 'Mrs'

    dataset.loc[dataset['Name'].str.contains('Miss.'), 'Title'] = 'Miss'

    dataset.loc[dataset['Name'].str.contains('Ms.'), 'Title'] = 'Ms'

    dataset.loc[dataset['Name'].str.contains('Mme.'), 'Title'] = 'Mme'

    dataset.loc[dataset['Name'].str.contains('Mlle.'), 'Title'] = 'Mlle'

    dataset.loc[dataset['Name'].str.contains('Master.'), 'Title'] = 'Master'

    dataset.loc[dataset['Name'].str.contains('Rev.'), 'Title'] = 'Rev'

    dataset.loc[dataset['Name'].str.contains('Don'), 'Title'] = 'Don'

    dataset.loc[dataset['Name'].str.contains('Dr.'), 'Title'] = 'Dr'

    dataset.loc[dataset['Name'].str.contains('Capt.'), 'Title'] = 'Capt'

    dataset.loc[dataset['Name'].str.contains('Major.'), 'Title'] = 'Major'

    dataset.loc[dataset['Name'].str.contains('Col.'), 'Title'] = 'Col'

    dataset.loc[dataset['Name'].str.contains('Jonkheer.'), 'Title'] = 'Jonkheer'

    dataset.loc[dataset['Name'].str.contains('Countess.'), 'Title'] = 'Countess'

    encode_string_labels(dataset, 'Title')



encode_titles(train)
train['Alone'] = 1

train.loc[(train['SibSp'] != 0), 'Alone'] = 0

train.loc[(train['Parch'] != 0), 'Alone'] = 0
features2 = [

    'Pclass',

    'Sex',

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'Embarked',

    'Alone',

    'Title'

]
tuning = False

if tuning:

    param_grid = {

        'n_estimators': range(50, 301, 50),

        'learning_rate': [0.05, 0.02, 0.01],

        'min_samples_split': range(50, 201, 50),

        'max_depth': [2, 3, 4, 5, 6],

        'max_features': ['sqrt']

    }



    clf = GradientBoostingClassifier(random_state=33)

    grid = GridSearchCV(clf, param_grid=param_grid, cv=ShuffleSplit(train_size=0.80, n_splits=10, random_state=1), verbose=1, n_jobs=8)

    train_model(grid, features2, target)

    

    display(grid.best_params_)
clf = GradientBoostingClassifier(

    learning_rate=0.01,

    max_depth=5,

    max_features='sqrt',

    min_samples_split=150,

    n_estimators=250,

    random_state=33)

train_model(clf, features2, target)

display_feature_importance(clf, features2, visualise=True)
features_to_be_encoded = [

    'Sex',

    'Embarked'

]



for feature in features_to_be_encoded:

    encode_string_labels(test, feature)

    

encode_titles(test)
impute_age(test)

test.loc[(test['Fare'].isnull()), 'Fare'] = test['Fare'].median()
test['Alone'] = 1

test.loc[(test['SibSp'] != 0), 'Alone'] = 0

test.loc[(test['Parch'] != 0), 'Alone'] = 0
X = test.loc[:, features2]

predictions = clf.predict(X)
def prepare_submission(test, predictions, filename='submission.csv'):

    submission = pd.DataFrame()

    submission['PassengerId'] = test['PassengerId']

    submission['Survived'] = predictions

    submission.to_csv(filename, index=False)

    display(submission.head())
prepare_submission(test, predictions)
import xgboost as xgb



xgb_params = {

    'n_estimators': 300,

    'learning_rate': 0.2,

    'max_depth': 5,

    'min_child_weight': 6,

    'reg_alpha': 0.9,

    'subsample': 0.80,

    'objective': 'binary:logistic',

    'rate_drop': 0.2,

    'skip_drop': 0.8,

    'eval_metric': 'auc',

    'booster': 'dart',

    'lambda': 0.2,

    'gamma': 0.02,

    'nthread': 4,

    'seed': 42,

    'silent': 1

}



X_train, X_test, y_train, y_test = train_test_split(

    train.loc[:, features2],

    train[target],

    test_size=0.20,

    random_state=33)



dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test)



watchlist = [(dtrain, 'train')]

model_xgb = xgb.train(xgb_params, dtrain, 300, watchlist, early_stopping_rounds=10, maximize=False, verbose_eval=50)



predicted_test_xgb = model_xgb.predict(dtest)

y_pred = np.array(predicted_test_xgb)

y_pred[y_pred > 0.5] = 1

y_pred[y_pred <= 0.5] = 0



score = accuracy_score(y_test, y_pred)

print('accuracy: {}'.format(score))
X = test.loc[:, features2]

X = xgb.DMatrix(X)



predicted_test_xgb = model_xgb.predict(X)

predictions = np.array(predicted_test_xgb)

predictions[predictions > 0.5] = 1

predictions[predictions <= 0.5] = 0

predictions = predictions.astype(int, copy=False)
prepare_submission(test, predictions, filename='submission_xgb2.csv')