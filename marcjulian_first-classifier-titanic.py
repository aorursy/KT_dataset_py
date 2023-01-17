# Importing all neccessary packages



import numpy as np # linear algebra

import pandas as pd # data processing



# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pandas.plotting import scatter_matrix



# Classifier

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Models

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier





# Data Cleaning

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



# Pipeline

from sklearn.pipeline import Pipeline



print('Import complete')
# Importing training and test data

train_data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('../input/titanic/test.csv')

print('Data loaded')
train_data.head()
train_data.shape
num_data = train_data.select_dtypes(exclude='object')

corr = num_data.corr()

sns.heatmap(corr)

num_data = train_data.select_dtypes(exclude='object')

cat_data = train_data.select_dtypes(include='object')
num_data.isna().sum()
num_data = num_data.drop('Survived', axis=1)
cat_data.head()
cat_data.isna().sum()
len(cat_data.Ticket.unique())
cat_data = cat_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)

#num_data = num_data.drop(['Fare'], axis=1)
data_copy = train_data.copy()



cat_cols_to_drop = ['Cabin', 'Ticket', 'Name']

data_copy = data_copy.drop(cat_cols_to_drop, axis=1)



#num_cols_to_drop = ['Fare']

#data_copy = data_copy.drop(num_cols_to_drop, axis=1)







data_copy.Embarked = data_copy.Embarked.fillna('S')

data_copy.isna().sum()
X = data_copy.drop('Survived', axis=1)

y = data_copy['Survived']



numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = OneHotEncoder()



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, num_data.columns),

    ('cat', categorical_transformer, cat_data.columns)

])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



from sklearn.model_selection import GridSearchCV



rf_classifier = RandomForestClassifier(n_estimators=100)

xgb_classifier = XGBClassifier(nthread=4, n_estimators=100, learning_rate=0.05)







rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_classifier)])

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_classifier)])







xgb_pipeline.fit(X_train, y_train)

print("model score: %.3f" % xgb_pipeline.score(X_test, y_test))
# Applying all data cleaning to the test data



test_X = test_data.copy()



test_X = test_X.drop(cat_cols_to_drop, axis=1)

test_X = test_X.drop('PassengerId', axis=1)
test_preds = xgb_pipeline.predict(test_X)

test_preds
output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                      'Survived': test_preds})

output.to_csv('submission.csv', index=False)

print('Submitted')