import numpy as np

import pandas as pd



data_full = pd.read_csv("../input/train.csv", index_col='PassengerId')

data_test = pd.read_csv("../input/test.csv", index_col='PassengerId')
data_full.head()
data_full.describe()
cols_with_missing = [col for col in data_full.columns

                     if data_full[col].isnull().any()]

print(cols_with_missing)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier



# Select our features

feature_names = ["Age", "Sex", "Pclass", "Fare", "SibSp", "Parch"]

X = data_full[feature_names]

y = data_full.Survived



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Setup a transformer to clean up our data.

preprocessor = ColumnTransformer(

    transformers=[

        # We will fill in missing numerical data with their mean.

        ('numerical', SimpleImputer(), ["Age", "Fare"]),

        # We will fill in missing categorical values with the mode value.

        ('categorical', categorical_transformer, ["Sex", "Pclass"]),

        # Let's also create a family size feature by adding the SibSp and Parch counts.

        ('family-size',

            FunctionTransformer(

                lambda v: pd.DataFrame({

                    'FamilySize': np.apply_along_axis(lambda row: row[0] + row[1], 1, v)

                }),

            # Set validate to false to disable sklearn 0.22 deprecation warning.

            validate=False),

            ["SibSp", "Parch"])

    ])



pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', XGBClassifier(random_state=0))

])



# Perform a grid search to identify the best model hyperparameters.

parameters = {

    'model__learning_rate': [0.0375, 0.04, 0.0425],

    'model__n_estimators': [500, 550]

}



CV = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy')

CV.fit(X, y)



print('Best score and parameter combination')



print(CV.best_score_)

print(CV.best_params_)

# Get our test predictions

X_test = data_test[feature_names]



# Create our final model

final_model = XGBClassifier(random_state=0, learning_rate=0.0375, n_estimators=550)



final_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', final_model)

    ])



# Fit our model with all the available data.

final_pipeline.fit(X, y)



# Generate and output our test predictions.

preds_test = final_pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': preds_test})

output.to_csv('submission.csv', index=False)