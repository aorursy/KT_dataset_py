import pandas as pd



from sklearn.model_selection import train_test_split
# Load data from file

melb_data = pd.read_csv('../input/melb_data.csv')
# Extract X and Y

y = melb_data.Price

X = melb_data.drop('Price', axis=1)


train_X, test_X, train_Y, test_Y = train_test_split(X, y, random_state=0)



# select categorical data

categorical_cols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object']



# select numerical data

numerical_cols = [cname for cname in train_X.columns if train_X[cname].dtype in ['int64', 'float64']]
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
#Preprocessing for Numerical Data

num_transformer = SimpleImputer(strategy='constant')



#PreProcessing Pipeline for  Category Data

cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



#Bundle preprocessing for Numerical and Categorical data

preprocessor = ColumnTransformer(transformers=[

    ('num', num_transformer, numerical_cols),

    ('cat', cat_transformer, categorical_cols)

])
# Define the model

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=100, random_state=0)
# Bundle preprocessing and modeling in a pipeline

from sklearn.metrics import mean_absolute_error



my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])
my_pipeline.fit(train_X, train_Y)
preds = my_pipeline.predict(test_X)
mean_absolute_error(test_Y, preds)