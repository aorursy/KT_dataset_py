import re

import scipy
import pandas
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
dftrain_full = pandas.read_csv('../input/train.csv')
dftrain_full.info()
dftrain_full.head()
dftrain_full.describe()
dftrain_full.describe(include=['O'])
print('is NA?\n', dftrain_full.isna().sum(), '\n')
print('is NULL?\n', dftrain_full.isnull().sum())
def get_title(name):
    regex_title = re.search(' ([a-zA-Z]+)\.', name)
    if regex_title:
        return regex_title.group(1)
    return ''
def preprocess(df: pandas.DataFrame,
               dfgroup_age: pandas.DataFrame=None,
               dfgroup_fare: pandas.DataFrame=None,
               df_mean: pandas.DataFrame=None,
               df_std: pandas.DataFrame=None,
               use_dummies: bool=True,
               standardized_columns=None):
    """Preprocessing function
    
    if only `df` is supplied, it regards `df` as the training data,
    otherwise as the test data.
    
    This function prefers "inplace" operation,
    so I will input a copy of a dataframe; e.g. `preprocess(df.copy())`.
    """
    df['Title'] = df['Name'].apply(get_title)
    df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr',
                         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                        'Rare',
                        inplace=True)
    df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'},
                        inplace=True)

    df = pandas.concat([df, pandas.get_dummies(df['Sex'],
                                               prefix='Sex')],
                       axis=1)

    df['Embarked'].fillna('S', inplace=True)

    if use_dummies is True:
        df = pandas.concat(
            [df, pandas.get_dummies(df['Title'], prefix='Title')],
            axis=1
        )
        df = pandas.concat(
            [df, pandas.get_dummies(df['Embarked'], prefix='Embarked')],
            axis=1
        )
        dropped_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin',
                        'Sex', 'Sex_male', 'Embarked', 'Embarked_C',
                        'Title', 'Title_Rare']
    else:
        df['Title'].replace({'Master': 1, 'Miss': 2,
                             'Mrs': 3, 'Mr': 4, 'Rare': 5},
                            inplace=True)
        df['Title'].fillna(0, inplace=True)
        df['Embarked'].replace({'S': 0, 'Q': 1, 'C': 2}, inplace=True)
        dropped_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin',
                        'Sex', 'Sex_male']

    if dfgroup_age is None:
        # Only at training
        dfgroup_age = df.groupby('Title')['Age'].mean()
    for title, age in dfgroup_age.items():
        df.loc[(df.Age.isnull())&(df.Title == title), 'Age'] = age

    if dfgroup_fare is None:
        # Only at training
        dfgroup_fare = df.groupby('Title')['Fare'].mean()
    for title, fare in dfgroup_fare.items():
        df.loc[(df.Fare.isnull())&(df.Title == title), 'Fare'] = fare

    df.drop(columns=dropped_cols, inplace=True)
    
    # Generate mean and std, then standardize the specifed columns
    if standardized_columns is not None:
        if df_mean is None:
            df_mean = df[standardized_columns].mean()
        if df_std is None:
            df_std = df[standardized_columns].std()
        df[standardized_columns] -= df_mean
        df[standardized_columns] /= df_std
    return df, dfgroup_age, dfgroup_fare, df_mean, df_std
dftrain, dfval = sklearn.model_selection.train_test_split(
    dftrain_full,
    test_size=0.3,
    random_state=0,
    stratify=dftrain_full['Survived'],
)
tr1Xy, tr1group_age, tr1group_fare, _, _ = preprocess(dftrain.copy())
tr2Xy, tr2group_age, tr2group_fare, _, _ = preprocess(dftrain.copy(),
                                                      use_dummies=False)
tr1X = tr1Xy.drop(columns=['Survived'])
tr1Y = tr1Xy['Survived']
tr2X = tr2Xy.drop(columns=['Survived'])
tr2Y = tr2Xy['Survived']

# tr1Xy.head()
# tr2Xy.head()
# Prepare the validation data
val1X, _, _, _, _ = preprocess(dfval.copy(), tr1group_age, tr1group_fare)
val1Y = val1X['Survived']
val1X.drop(columns=['Survived'], inplace=True)

val2X, _, _, _, _ = preprocess(dfval.copy(), tr2group_age, tr2group_fare,
                               use_dummies=False)
val2Y = val2X['Survived']
val2X.drop(columns=['Survived'], inplace=True)

# val1X.head()
# val2X.head()
# Set a very large value for the regularization parameter
logistic1 = LogisticRegression(penalty='l2', C=1e50)
logistic1.fit(tr1X, tr1Y)
logistic1_pred = logistic1.predict(val1X)
val1_accuracy = (logistic1_pred != val1Y.values).mean()

logistic2 = LogisticRegression(penalty='l2', C=1e50)
logistic2.fit(tr2X, tr2Y)
logistic2_pred = logistic2.predict(val2X)
val2_accuracy = (logistic2_pred != val2Y.values).mean()
print('Validation accuracy (Dummies):', 1-val1_accuracy)
print('Validation accuracy (No dummies):', 1-val2_accuracy)
columns_numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
columns_all = columns_numeric + [
    'Sex_female', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
    'Embarked_Q', 'Embarked_S',
]

(tr3Xy,
 tr3group_age,
 tr3group_fare,
 tr3mean,
 tr3std) = preprocess(dftrain.copy(),
                      standardized_columns=columns_numeric)
(tr4Xy,
 tr4group_age,
 tr4group_fare,
 tr4mean,
 tr4std) = preprocess(dftrain.copy(),
                      standardized_columns=columns_all)

tr3X = tr3Xy.drop(columns=['Survived'])
tr3Y = tr3Xy['Survived']
tr4X = tr4Xy.drop(columns=['Survived'])
tr4Y = tr4Xy['Survived']

# print('Really standardized?\n', tr3Xy.mean(), '\n', tr3Xy.std())
# print('Really standardized?\n', tr4Xy.mean(), '\n', tr4Xy.std())
# Prepare the validation data
val3X, _, _, _, _ = preprocess(dfval.copy(),
                               tr3group_age, tr3group_fare,
                               tr3mean, tr3std,
                               standardized_columns=columns_numeric)
val4X, _, _, _, _ = preprocess(dfval.copy(),
                               tr4group_age, tr4group_fare,
                               tr4mean, tr4std,
                               standardized_columns=columns_all)

val3Y = val3X['Survived']
val3X.drop(columns=['Survived'], inplace=True)
val4Y = val4X['Survived']
val4X.drop(columns=['Survived'], inplace=True)

# val3X.head()
# val4X.head()
logistic3 = LogisticRegression(penalty='l2', C=1e50)
logistic3.fit(tr3X, tr3Y)
logistic3_pred = logistic3.predict(val3X)
val3_accuracy = (logistic3_pred != val3Y.values).mean()

logistic4 = LogisticRegression(penalty='l2', C=1e50)
logistic4.fit(tr4X, tr4Y)
logistic4_pred = logistic4.predict(val4X)
val4_accuracy = (logistic4_pred != val4Y.values).mean()
print('Validation accuracy (Standardized numeric variables):', 1-val3_accuracy)
print('Validation accuracy (All standardized):', 1-val4_accuracy)
tr5Xy, _, _, _, _ = preprocess(dftrain_full.copy())
tr6Xy, _, _, _, _ = preprocess(dftrain_full.copy(), use_dummies=False)

tr5X = tr5Xy.drop(columns=['Survived'])
tr5y = tr5Xy['Survived']
tr6X = tr6Xy.drop(columns=['Survived'])
tr6y = tr6Xy['Survived']

logistic5 = LogisticRegressionCV(penalty='l2', Cs=100, cv=10)
logistic6 = LogisticRegressionCV(penalty='l2', Cs=100, cv=10)

logistic5.fit(tr5X, tr5y)
logistic6.fit(tr6X, tr6y)

val5accuracy = logistic5.scores_[1]
val6accuracy = logistic6.scores_[1]
# Plot
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.semilogx(logistic5.Cs_, val5accuracy.mean(axis=0), label='With dummies')
ax1.semilogx(logistic6.Cs_, val6accuracy.mean(axis=0), label='No dummies')
ax1.set_xlabel('Regularization')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.set_title('LogisticRegressionCV')
# https://stackoverflow.com/questions/20625582
pandas.options.mode.chained_assignment = None


class CustomScaler(sklearn.base.TransformerMixin):
    """This standardizes only specified variables.
    Reference: https://stackoverflow.com/questions/37685412
    """
    def __init__(self, columns):
        self.columns = columns
        self.scaler = sklearn.preprocessing.StandardScaler()
        
    def fit(self, X, y):
        self.scaler.fit(X[self.columns], y)
        return self
    
    def transform(self, X):
        X.loc[:, self.columns] = self.scaler.transform(X[self.columns])
        return X
# Only the numeric standardized
pipe7 = sklearn.pipeline.make_pipeline(
    CustomScaler(columns=['Age', 'SibSp', 'Parch', 'Fare']),
    LogisticRegressionCV(penalty='l2', Cs=100, cv=10)
)
# All standardized
pipe8 = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    LogisticRegressionCV(penalty='l2', Cs=100, cv=10)
)
tr7Xy, _, _, _, _ = preprocess(dftrain_full.copy())
tr7X = tr7Xy.drop(columns=['Survived'])
tr7y = tr7Xy['Survived']

pipe7.fit(tr7X, tr7y)
pipe8.fit(tr7X, tr7y)

# Access LogisticRegressionCV classes from pipelines
logistic7 = pipe7.steps[1][1]
logistic8 = pipe8.steps[1][1]

# Get the raw CV results (cv x Cs ndarrays)
val7accuracy = logistic7.scores_[1]
val8accuracy = logistic8.scores_[1]
# Plot
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.semilogx(logistic5.Cs_, val5accuracy.mean(axis=0), label='None')
ax2.semilogx(logistic7.Cs_, val7accuracy.mean(axis=0), label='Numeric')
ax2.semilogx(logistic8.Cs_, val8accuracy.mean(axis=0), label='All')
ax2.set_xlabel('Regularization')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('LogisticRegressionCV with Dummies')
# Read the test data
dftest = pandas.read_csv('../input/test.csv')

# Generating-csv function for the submission
def submit(dftest: pandas.DataFrame, pred: scipy.ndarray, csv_fname: str)->None:
    """For local-use Only."""
#     dfsubmit = pandas.DataFrame(data={'PassengerId': dftest['PassengerId'],
#                                       'Survived': pred})
#     dfsubmit.to_csv(f'../data/titanic/{csv_fname}', index=False)
    pass
# The first round, plain logistic regression models without CV
te1X, _, _, _, _ = preprocess(dftest.copy(), tr1group_age, tr1group_fare)
pred1 = logistic1.predict(te1X)
submit(dftest, pred1, 'logistic1.csv')

te2X, _, _, _, _ = preprocess(dftest.copy(), tr2group_age, tr2group_fare,
                              use_dummies=False)
pred2 = logistic2.predict(te2X)
submit(dftest, pred2, 'logistic2.csv')

te3X, _, _, _, _ = preprocess(dftest.copy(),
                              tr3group_age, tr3group_fare,
                              tr3mean, tr3std,
                              standardized_columns=columns_numeric)
pred3 = logistic3.predict(te3X)
submit(dftest, pred3, 'logistic3.csv')

te4X, _, _, _, _ = preprocess(dftest.copy(),
                              tr4group_age, tr4group_fare,
                              tr4mean, tr4std,
                              standardized_columns=columns_all)
pred4 = logistic4.predict(te4X)
submit(dftest, pred4, 'logistic4.csv')
# The second round, regularization with CV
te5X, _, _, _, _ = preprocess(dftest.copy())
pred5 = logistic5.predict(te5X)
submit(dftest, pred5, 'logistic5.csv')

te6X, _, _, _, _ = preprocess(dftest.copy(), use_dummies=False)
pred6 = logistic6.predict(te6X)
submit(dftest, pred6, 'logistic6.csv')

pred7 = pipe7.predict(te5X)
submit(dftest, pred7, 'logistic7.csv')

pred8 = pipe8.predict(te5X)
submit(dftest, pred8, 'logistic8.csv')