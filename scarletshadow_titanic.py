

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

datas = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        datas.append(os.path.join(dirname, filename))

test_data = pd.read_csv(datas[1])
train_data = pd.read_csv(datas[2])

train_data.head()
test_data.head()
y = train_data.Survived
train_data.drop('Survived', axis = 1, inplace = True)
train_data.head()
train_copy = train_data.copy()
test_copy = test_data.copy()

def sex_binary(value):
    if value == 'female':
        return 1
    else:
        return 0

set(train_copy) == set(test_copy) # TRUE

train_copy['Binary_sex'] = train_copy['Sex'].map(sex_binary)
test_copy['Binary_sex'] = test_copy['Sex'].map(sex_binary)

train_copy.drop('Sex', axis = 1, inplace = True)
test_copy.drop('Sex', axis = 1, inplace = True)

train_copy.head()
# I think these features arent useful
train_copy.drop(['Ticket', 'Cabin','PassengerId', 'Name'], axis = 1, inplace=True)
test_copy.drop(['Ticket', 'Cabin','PassengerId', 'Name'], axis = 1, inplace=True)
object_cols = [col for col in train_copy.columns if train_copy[col].dtype == 'object']
n_cols = [col for col in train_copy.columns if train_copy[col].dtype != 'object']

print('Categorical variables: {}\nNumerical variables: {}'.format(object_cols, n_cols))
#Separate high and low cardinality to use Label and One hot encoding
high_cardinality = [col for col in object_cols if train_copy[col].nunique() > 10]
low_cardinality = set(object_cols) - set(high_cardinality)

print('High cardinality: {}\nLow cardinality: {}'.format(high_cardinality, low_cardinality))
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# places missing values with the most frequent value
numerical_transformer = SimpleImputer(strategy = 'constant')

# the same to missing categorical and apply one hot encoding to a low card col
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, n_cols),
    ('cat', categorical_transformer, object_cols)
])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
X_train,X_valid, y_train, y_valid = train_test_split(train_copy, y, train_size = 0.8, test_size = 0.2)
trees = [25,50, 100, 150, 200, 250, 300]

for tree in trees:
    model = RandomForestClassifier(max_leaf_nodes=tree, random_state = 0)
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    my_pipeline.fit(X_train, y_train)
    preds = my_pipeline.predict(X_valid)
    print(np.mean(y_valid == preds))
model = RandomForestClassifier(max_leaf_nodes=100, n_estimators = 5, random_state = 0)
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])
my_pipeline.fit(train_copy, y)
preds = my_pipeline.predict(test_copy)

sub = pd.Series(preds, index=test_data['PassengerId'], name='Survived')
sub.to_csv('titanic_second_model2.csv', index=True)