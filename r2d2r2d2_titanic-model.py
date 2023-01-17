from typing import Any, Union
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import re
from statistics import mode
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# dataset
dataset = pd.read_csv('../input/titanic/train.csv')

# submission data
X_test_submission = pd.read_csv('../input/titanic/test.csv')

dataset.info()
X_test_submission.info()
dataset.isnull().sum()
X_test_submission.isnull().sum()
dataset.loc[dataset.Age.isnull(), 'Age'] = dataset.groupby("Pclass").Age.transform('median')
dataset.isnull().sum()
X_test_submission.loc[X_test_submission.Age.isnull(), 'Age'] = dataset.groupby("Pclass").Age.transform('median')
X_test_submission.isnull().sum()
dataset['Cabin'] = dataset['Cabin'].fillna('U')
dataset.isnull().sum()
X_test_submission['Cabin'] = X_test_submission['Cabin'].fillna('U')
X_test_submission.isnull().sum()
dataset.Embarked.value_counts()
X_test_submission.Embarked.value_counts()
dataset['Embarked'] = dataset['Embarked'].fillna(mode(dataset['Embarked']))
dataset.isnull().sum()
X_test_submission['Embarked'] = X_test_submission['Embarked'].fillna(mode(X_test_submission['Embarked']))
X_test_submission['Fare'] = X_test_submission['Fare'].fillna(mode(X_test_submission['Fare']))

X_test_submission.isnull().sum()
def display_heatmap_na(df):
    print(df.isnull().sum())
    plt.style.use('seaborn')
    plt.figure()
    sns.heatmap(df.isnull(), yticklabels = False, cmap='plasma')
    plt.title('Null Values in Training Set')

display_heatmap_na(dataset)

display_heatmap_na(X_test_submission)
# dataset['Fare']  = dataset.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
dataset.Cabin.value_counts()
X_test_submission.Cabin.value_counts()
# dataset['Cabin'] = dataset['Cabin'].fillna('U')
# display_heatmap_na(dataset)
dataset.Sex.unique()
X_test_submission.Sex.unique()
dataset['Sex'][dataset['Sex'] == 'male'] = 0
dataset['Sex'][dataset['Sex'] == 'female'] = 1
X_test_submission['Sex'][X_test_submission['Sex'] == 'male'] = 0
X_test_submission['Sex'][X_test_submission['Sex'] == 'female'] = 1
dataset.Embarked.unique()
X_test_submission.Embarked.unique()
encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(dataset[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
dataset = dataset.join(temp)
dataset.drop(columns='Embarked', inplace=True)

# display_heatmap_na(dataset)
encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(X_test_submission[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
X_test_submission = X_test_submission.join(temp)
X_test_submission.drop(columns='Embarked', inplace=True)

# display_heatmap_na(dataset)
# deleting outliers
dataset= dataset[dataset['Age'] < 70]
dataset= dataset[dataset['Fare'] < 500]

dataset.columns
dataset.Cabin.tolist()[0:20]
X_test_submission.Cabin.tolist()[0:20]
dataset['Cabin'] = dataset['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
dataset.Cabin.unique()
X_test_submission['Cabin'] = X_test_submission['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
X_test_submission.Cabin.unique()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
dataset['Cabin'] = dataset['Cabin'].map(cabin_category)
dataset.Cabin.unique()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
X_test_submission['Cabin'] = X_test_submission['Cabin'].map(cabin_category)
X_test_submission.Cabin.unique()
dataset.Name
X_test_submission.Name
dataset['Name'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
dataset['Name'].unique().tolist()
X_test_submission['Name'] = X_test_submission.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
X_test_submission['Name'].unique().tolist()
dataset.rename(columns={'Name' : 'Title'}, inplace=True)
dataset['Title'] = dataset['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess',
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

dataset['Title'].value_counts(normalize = True) * 100
X_test_submission.rename(columns={'Name' : 'Title'}, inplace=True)
X_test_submission['Title'] = X_test_submission['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess',
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

X_test_submission['Title'].value_counts(normalize = True) * 100
dataset = dataset.reset_index(drop=True)

encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(dataset[['Title']]).toarray())
dataset = dataset.join(temp)
dataset.drop(columns='Title', inplace=True)

# display_heatmap_na(dataset)
X_test_submission = X_test_submission.reset_index(drop=True)

encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(X_test_submission[['Title']]).toarray())
X_test_submission = X_test_submission.join(temp)
X_test_submission.drop(columns='Title', inplace=True)

# display_heatmap_na(dataset)
dataset['familySize'] = dataset['SibSp'] + dataset['Parch'] + 1

X_test_submission['familySize'] = X_test_submission['SibSp'] + X_test_submission['Parch'] + 1
fig = plt.figure(figsize = (15,4))

ax1 = fig.add_subplot(121)
ax = sns.countplot(dataset['familySize'], ax = ax1)

# calculate passengers for each category
labels = (dataset['familySize'].value_counts())
# add result numbers on barchart
for i, v in enumerate(labels):
    ax.text(i, v+6, str(v), horizontalalignment = 'center', size = 10, color = 'black')

plt.title('Passengers distribution by family size')
plt.ylabel('Number of passengers')

ax2 = fig.add_subplot(122)
d = dataset.groupby('familySize')['Survived'].value_counts(normalize = True).unstack()
d.plot(kind='bar', color=["#3f3e6fd1", "#85c6a9"], stacked='True', ax = ax2)
plt.title('Proportion of survived/drowned passengers by family size (train data)')
plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))
plt.xticks(rotation = False)

plt.tight_layout()
# drop redundant features
dataset = dataset.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)

# drop unused columns
survived = dataset['Survived']
dataset = dataset.drop(['PassengerId', 'Survived'], axis = 1)

dataset.head()
# drop redundant features
X_test_submission = X_test_submission.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)

# drop unused columns
X_test_submission = X_test_submission.drop(['PassengerId'], axis = 1)

dataset.head()
from sklearn.model_selection import train_test_split, learning_curve

X_train, X_test, y_train, y_test = train_test_split(dataset, survived, test_size = 0.2, random_state=2)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
X_test_submission_scaled = scaler.transform(X_test_submission)
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

X_train_scaled, y_train = SMOTE(k_neighbors=(np.unique(y_train, return_counts=True)[1][1] - 1)).fit_resample(X_train_scaled, y_train)

def evaluation(model, scoring='accuracy'):

    model.fit(X_train_scaled, y_train)
    ypred = model.predict(X_test_scaled)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))

    N, train_score, val_score = learning_curve(model,
                                                X_train_scaled,
                                                y_train,
                                                cv=4,
                                                scoring=scoring,
                                                train_sizes=np.linspace(0.1, 1, 10))


    plt.figure(figsize=(8, 6))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# delete infinite values

def assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)

def remove_infinite_values(matrix):
    # find min and max values for each column, ignoring nan, -inf, and inf
    mins = [np.nanmin(matrix[:, i][matrix[:, i] != -np.inf]) for i in range(matrix.shape[1])]
    maxs = [np.nanmax(matrix[:, i][matrix[:, i] != np.inf]) for i in range(matrix.shape[1])]

    # go through matrix one column at a time and replace  + and -infinity 
    # with the max or min for that column
    for i in range(log_train_arr.shape[1]):
        matrix[:, i][matrix[:, i] == -np.inf] = mins[i]
        matrix[:, i][matrix[:, i] == np.inf] = maxs[i]

# assert_all_finite(X_train)
# assert_all_finite(y_train)

# remove_infinite_values(X_train)
# remove_infinite_values(y_train)

random_forest = RandomForestClassifier().fit(X_train_scaled, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(random_forest.score(X_train_scaled, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(random_forest.score(X_test_scaled, y_test)))


preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))

random_forest_pipeline = make_pipeline(preprocessor, RandomForestClassifier())

evaluation(random_forest_pipeline, scoring='accuracy')
random_forest_pipeline.get_params()

# Set our parameter grid
param_grid = {
    'randomforestclassifier__criterion' : ['gini', 'entropy'],
    'randomforestclassifier__n_estimators': [20, 50, 100, 300, 500, 1000],
    'randomforestclassifier__max_features': ['auto', 'log2', 'sqrt'],
    'randomforestclassifier__max_depth' : [3, 4, 5, 6, 7, 10],
    'randomforestclassifier__min_samples_leaf': [5, 8, 10, 20, 50],
    'randomforestclassifier__min_samples_split': [1, 2, 3, 4, 5, 8, 10],
    'randomforestclassifier__n_jobs': [5, 8, 10, 20, 50],
    'pipeline__selectkbest__k': [3, 4, 5, 8, 10],
    'pipeline__polynomialfeatures__degree': [2, 3, 4, 5, 8]
}


def randomized_search_cv():
    # n_iter -> combien de fois l'algorithme va-t-il tester les combinaisons
    grid = RandomizedSearchCV(random_forest_pipeline, param_grid, scoring='recall', cv=4,
                          n_iter=60)
    grid.fit(X_train_scaled, y_train)
    print(grid.best_params_)
    y_pred = grid.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    return grid


# grid = randomized_search_cv()


# Model Accuracy, how often is the classifier correct?
# print("Accuracy:", accuracy_score(y_test, y_pred))
# evaluation(grid.best_estimator_, scoring='accuracy')
'''
{'randomforestclassifier__n_jobs': 10, 
'randomforestclassifier__n_estimators': 500, 
'randomforestclassifier__min_samples_split': 2, 
'randomforestclassifier__min_samples_leaf': 8, 
'randomforestclassifier__max_features': 'log2', 
'randomforestclassifier__max_depth': 3, 
'randomforestclassifier__criterion': 'gini', 
'pipeline__selectkbest__k': 3, 
'pipeline__polynomialfeatures__degree': 4}
'''

random_forest = RandomForestClassifier(n_jobs= 10,
                                        criterion = 'gini',
                                        max_depth = 3,
                                        max_features = 'log2',
                                        min_samples_leaf = 8,
                                        min_samples_split = 2,
                                        n_estimators = 500)

random_forest.fit(X_train_scaled, y_train)
y_pred = random_forest.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred) * 100
evaluation(random_forest, scoring='accuracy')
preprocessor = make_pipeline(PolynomialFeatures(4, include_bias=False), SelectKBest(f_classif, k=3))

random_forest_pipeline = make_pipeline(preprocessor, RandomForestClassifier())

evaluation(random_forest_pipeline, scoring='accuracy')
model = random_forest.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

evaluation(model, scoring='f1')
y_pred_submission = model.predict(X_test_submission_scaled)

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission = pd.DataFrame({
        'PassengerId': gender_submission['PassengerId'],
        'Survived': y_pred_submission
    })

submission.to_csv('titanic_submission.csv', index=False)

print('It\'s done')



