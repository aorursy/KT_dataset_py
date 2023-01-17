# Libraries

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



# Preprocessing

from sklearn import pipeline

from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn_pandas import DataFrameMapper

from sklearn.base import TransformerMixin



# Optimization

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# Classifier and regressor

from xgboost.sklearn import XGBClassifier, XGBRegressor



# Metrics

from sklearn.metrics import accuracy_score, classification_report



%matplotlib inline
data = pd.read_csv('../input/train.csv')

data_predict = pd.read_csv('../input/test.csv')

labels = data.Survived
data.head(10)
# Unused in learning columns.

unused = ['PassengerId', 'Name', 'Ticket']



# Columns, used for learning.

valuable = list(set(data.columns) - set(unused) - set([labels.name]))

valuable
data.hist('Survived')

plt.title('Class balance')

plt.show()
corr = data[[*valuable, 'Survived']].corr()

# mask = np.zeros_like(corr)

# mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))

ax = sns.heatmap(corr, vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)

plt.title('Cross correlation between numerical features and target label')

plt.show()
sns.factorplot(x='Embarked', hue='Survived', data=data, kind='count')

plt.title('Dependency on embarked port')

plt.show()
from matplotlib.scale import InvertedLog10Transform

g = sns.FacetGrid(data, col='Sex',  hue='Survived', size=5)

g = (g.map(plt.scatter, 'Age', 'Fare', edgecolor="w", s=70, ).add_legend())

plt.yscale('symlog')

plt.ylim((3, 1000))

plt.show()
def missed_values_count(data, verbose=True):

    columns_with_na = {}

    for col in data.columns:

        columns_with_na[col] = data[col].shape[0] - data[col].dropna().shape[0]

        if verbose:

            print('{:<15}{}'.format(col, columns_with_na[col]))
print('Missed values in train data:')

missed_values_count(data[valuable])

print()

print('Missed values in test data:')

missed_values_count(data_predict[valuable])
print('Count of missed Fare values:', 

      np.sum(data['Fare'] == 0) + 

      np.sum(data_predict['Fare'] == 0) + 

      np.sum(data_predict['Fare'].notnull() == False))
# Calculate mean Fare for passenger classes (on train data only)

median_fare_by_class = (data[['Pclass', 'Fare']].groupby(['Pclass']).median())

median_fare_by_class
def fill_missed_fare(data, fill_values):

    rows_with_na = (data['Fare'] == 0) | (data['Fare'].notnull() == False)

    data.loc[rows_with_na, 'Fare'] = (fill_values.loc[data.loc[rows_with_na, 'Pclass']])['Fare'].values



# Fill missed values

fill_missed_fare(data, median_fare_by_class)

fill_missed_fare(data_predict, median_fare_by_class)
# Check

print('Count of missed Fare values:', 

      np.sum(data['Fare'] == 0) + 

      np.sum(data_predict['Fare'] == 0) + 

      np.sum(data_predict['Fare'].notnull() == False))
data[data['Embarked'].isnull()]
sns.boxplot('Embarked', 'Fare', data=data[data['Pclass']==1])



x = plt.gca().axes.get_xlim()

plt.plot(x, len(x) * [80.], sns.xkcd_rgb['red'], label='Fare=80')

plt.yscale('symlog')

plt.ylim((3, 1000))

plt.title('Fare distribution for Plcass=1')

# plt.yticks(np.arange(10))

plt.legend()

plt.show()
data['Embarked'].fillna('C', inplace=True)
def add_family_column(data):

    data['Family'] = data['SibSp'] + data['Parch'] + 1



add_family_column(data)

add_family_column(data_predict)
def add_deck_column(data):

    data['Deck'] = data['Cabin'].apply(lambda x: np.nan if pd.isnull(x) else x[0])



add_deck_column(data)

add_deck_column(data_predict)

data.head()
deck_train = data['Deck'].isnull() == False

# Train classifier on train data



deck_classifier = pipeline.Pipeline(steps = [

        ('data_preprocessing', StandardScaler()),

        ('model_fitting', XGBClassifier()),

    ]

)



deck_classifier.fit(data.loc[deck_train, ['Pclass', 'Fare']], data.loc[deck_train, ['Deck']].values.ravel())
def fill_missing_deck(data, predictor):

    rows_to_fill = data['Deck'].isnull()

    data.loc[rows_to_fill, 'Deck'] = predictor(data.loc[rows_to_fill, ['Pclass', 'Fare']])



fill_missing_deck(data, deck_classifier.predict)

fill_missing_deck(data_predict, deck_classifier.predict)
data[['Pclass', 'Fare', 'Deck']].head()
# Add popular titles as separate column, extracted from the full name, or with 'Other' if none found.

titles = ['Mr', 'Mrs', 'Miss', 'Master']

no_title = 'Other'





def get_title(name):

    for title in titles:

        if name.find(title) != -1:

            return title

    return no_title





def add_title_column(data):

    data['Title'] = data['Name'].apply(get_title)





add_title_column(data)

add_title_column(data_predict)
# Converter for transorming categorical columns to dict.

class DictConverter(TransformerMixin):

    def fit(self, x, y=None):

        return self

    def transform(self, X):

        return X.to_dict(orient='index').values()
columns = ['Sex', 'Title', 'Deck', 'Pclass', 'Fare', 'Family']

    

age_estimator = pipeline.Pipeline(steps = [

        ('data_preprocessing', DataFrameMapper([

                (columns[3:], [

                    # Scale

                    StandardScaler()

                ]),

                (columns[:3], [

                    # Convert to dict

                    DictConverter(),

                    # Vectorize features

                    DictVectorizer(sparse=False)

                ])

            ], input_df=True)),

        ('model_fitting', XGBRegressor())

    ]

)



age_train = data['Age'].isnull() == False

age_estimator.fit(data.loc[age_train, columns], data.loc[age_train, 'Age'].values.ravel())



def fill_missing_age(data, predictor):

    rows_to_fill = data['Age'].isnull()

    data.loc[rows_to_fill, 'Age'] = predictor(data.loc[rows_to_fill, columns])



fill_missing_age(data, age_estimator.predict)

fill_missing_age(data_predict, age_estimator.predict)
data[age_train == False].head()
data.head()
train_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','Deck', 'Family']



# Numeric columns

numeric = [column for column in train_columns if data[column].dtype != 'O']



# Categorical columns

categorical = [column for column in train_columns if data[column].dtype == 'O']



train_columns, numeric, categorical
(data_train, 

 data_test, 

 labels_train, labels_test) = train_test_split(data, labels, test_size=0.25, stratify=labels)
preprocessor = DataFrameMapper([

    (numeric, [

        # Scale

        StandardScaler()

    ]),

    (categorical, [

        DictConverter(),

        # Vectorize features

        DictVectorizer(sparse=False)

    ])

], input_df=True)



estimator = pipeline.Pipeline(steps = [

        ('data_preprocessing', preprocessor),

        ('model_fitting', XGBClassifier())

    ]

)
estimator.fit(data_train, labels_train)

prediction_test = estimator.predict(data_test)

print('{:<20}{:.3f}\n'.format('Test accuracy score: ', accuracy_score(labels_test, prediction_test)))

print('Classification report:\n')

print(classification_report(labels_test, prediction_test, target_names=['Lost', 'Survived'], digits=3))
def print_results(optimizer, X_test, y_test, title=None, tab=30):

    # Calculate score

    prediction = optimizer.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    

    # Print & plot results

    if title:

        print(title)

    print('Optimization results:\n')

    print('Best parameters:')

    for key in optimizer.best_params_.keys():

        print('{key:<{tab}}{value}'.format('', key=key, 

                                           value=optimizer.best_params_[key], 

                                           tab=tab))

    

    print('\nAccuracy scores:')

    print('{:<{tab}}{:.4f}'.format('Best validation score', optimizer.best_score_, tab=tab))

    print('{:<{tab}}{:.4f}'.format('Test score', accuracy, tab=tab))
param_grid = {

    'model_fitting__learning_rate': [0.005, 0.01, 0.025, 0.05, 0.1],

    'model_fitting__n_estimators': [100, 250, 500, 750, 1000],

    'model_fitting__max_depth': [3, 4, 5, 7],

}



cv = 3

optimizer = GridSearchCV(estimator, param_grid=param_grid, cv=cv)
optimizer.fit(data_train, labels_train)

print_results(optimizer, data_test, labels_test)
def export_submission(prediction, filename):

    submission = pd.DataFrame(data_predict.PassengerId)

    submission['Survived'] = prediction

    submission.to_csv(filename, index=False, header=True, sep=',')
optimizer.best_estimator_.fit(data, labels)

prediction = optimizer.best_estimator_.predict(data_predict)

export_submission(prediction, 'XGBoost_best.csv')

# Score 0.80382 on data for prediction