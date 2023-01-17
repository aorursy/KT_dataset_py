import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import ExtraTreesClassifier as Classifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
target = 'Cover_Type'

cols_to_drop = ["Soil_Type7", "Soil_Type15"]
preprocessor = ColumnTransformer(

    remainder='passthrough',                  # keep all columns

    transformers=[

        ('drop', 'drop', cols_to_drop),       # except these

        # Could possibly use `FunctionTransformer` (as many as needed ) for feature engineering

    ])
model = Classifier(n_jobs=-1, random_state=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
train = pd.read_csv('../input/learn-together/train.csv', index_col='Id')  # 15120 records

X, y = train.drop([target], axis=1), train[target]
cv = StratifiedKFold(n_splits=3, random_state=0)

param_grid = {

    "model__random_state": [0],   # [0, 1, 2, 3, 4],

    "model__n_estimators": [360], # [320, 340, 360, 380, 400],

    "model__max_depth": [32]      # [25, 30, 32, 34, 38, 45]

}

searchCV = GridSearchCV(estimator=pipeline, scoring='accuracy', cv=cv, param_grid=param_grid, verbose=True)



# WARNING: This could take some time to run.

searchCV.fit(X, y)



print('Best index:', searchCV.best_index_)

print('Best score:', searchCV.best_score_)

print('Best params:', searchCV.best_params_)
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')  # 565892 records

test_preds = searchCV.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)