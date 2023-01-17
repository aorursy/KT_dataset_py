import pandas as pd
train_data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
y = train_data['Survived'] # Setting target

train_data.drop(['Survived'], axis=1, inplace=True) # Dropping target from features data
numerical_cols = [col for col in train_data.columns

                    if train_data[col].dtype in ['int64', 'float64']]

print(numerical_cols)
categorical_cols = [col for col in train_data.columns

                        if train_data[col].dtype == 'object'

                        and train_data[col].nunique() < 10]

print(categorical_cols)
# Selecting columns

my_cols = numerical_cols + categorical_cols

X = train_data[my_cols]

X_test = test_data[my_cols]
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
num_transformer = SimpleImputer()
cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, numerical_cols),

        ('cat', cat_transformer, categorical_cols)

    ])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_features=6, n_jobs=4)
my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=4)
print('Mean Accuracy: {:.3f}'.format(cv_scores.mean()))
my_pipeline.fit(X, y)
test_preds = my_pipeline.predict(X_test)
output = pd.DataFrame({

    'PassengerId': X_test.index,

    'Survived': test_preds

})
output.to_csv('submission.csv', index=False)