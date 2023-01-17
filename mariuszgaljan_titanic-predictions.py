import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



passengers_train = pd.read_csv('../input/titanic/train.csv', index_col=0)

passengers_test = pd.read_csv('../input/titanic/test.csv', index_col=0)

# passengers_train.head()

passengers_train.head()
columns_with_missing_values = [col for col in passengers_train.columns if passengers_train[col].isnull().any()]

print("Columns with missing values: ", columns_with_missing_values)



print("Amount of unique values in each column:")

for col in passengers_train.columns:

    print(col, passengers_train[col].nunique())

    

print("Rows with missing cabin value: ", passengers_train['Cabin'].isnull().sum(), '/', passengers_train.Cabin.notnull().count())
# and now we can form our data

# I decided to use this columns

num_features = ['Age', 'Fare', 'Pclass', 'Parch']

category_features = ['Embarked', 'Sex']

features = num_features + category_features



# most of the rows are missing the cabin field, therefore I decided to just drop it. Ticket and name will propably not be neccessary

columns_to_drop = ['Cabin', 'Ticket', 'Name']



X_train = passengers_train.drop(columns_to_drop, axis='columns')

y_train = passengers_train['Survived']

X_train = X_train.drop('Survived', axis='columns')



X_test = passengers_test.drop(columns_to_drop, axis='columns')



y_train.head()
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score
numerical_transformer = SimpleImputer(strategy='most_frequent')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, num_features),

    ('cat', categorical_transformer, category_features)

])



model = XGBClassifier(seed=1, learning_rate=0.01)

# model = RandomForestClassifier(random_state=0)

# model = LogisticRegression(random_state=1)



main_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])
def get_score(params):

    """Return the average MAE over 5 CV folds of the model.

    

    Keyword argument:

    params - a dict of parameters

    """

    model.set_params(**params)

    

    results = cross_val_score(main_pipeline, X_train, y_train, cv=5, scoring='accuracy')

    return results.mean()
results = {i : get_score({'n_estimators' : i}) for i in range(100, 1000, 50)}

results
best_params = {'n_estimators' : 500}



model.set_params(**best_params)



main_pipeline.fit(X_train, y_train)
predictions = main_pipeline.predict(X_test)

# predictions = [round(pred) for pred in predictions]
output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

output.head()