import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y_train = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



categorical_cols = [cname for cname in X_full.columns if X_full[cname].nunique() < 10 and X_full[cname].dtype == "object"]

numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]



#Set aside columns with more than 25% missing values

cat_highNA_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]

categorical_cols = [item for item in categorical_cols if item not in cat_highNA_cols]



#Set aside discrete numerical columns

numcat_cols = ["MSSubClass", "OverallQual", "OverallCond", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", 

                "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "MoSold"]

numerical_cols = [item for item in numerical_cols if item not in numcat_cols]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols + numcat_cols + cat_highNA_cols

X_train = X_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy="mean")

numcat_transformer = SimpleImputer(strategy="most_frequent")



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



cat_highNA_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value = "Missing")),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols),

        ('numcat', numcat_transformer, numcat_cols)

#        ('highcat', cat_highNA_transformer, cat_highNA_cols)

    ])



# Define model

model = RandomForestRegressor()
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



param_grid = {

    'model__n_estimators': [120, 180, 240],

    'model__max_depth': [80, 120, 160],

    'model__min_samples_leaf': [3, 4, 5],

    'model__min_samples_split': [6, 8, 10]

}



search = GridSearchCV(my_pipeline, param_grid, n_jobs=-1, verbose=10, cv=3)

search.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % search.best_score_)

print(search.best_params_)
# Preprocessing of test data, fit model

preds = search.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)