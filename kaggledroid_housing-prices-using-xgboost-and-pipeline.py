#Importing the necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor
#Loading the training and testing data 

train_data_path = '../input/home-data-for-ml-course/train.csv'

X_full = pd.read_csv(train_data_path, index_col="Id")

test_data_path = '../input/home-data-for-ml-course/test.csv'

X_test_full = pd.read_csv(test_data_path, index_col="Id")
X_full.shape

X_full.describe()
#Separating the target from the features

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



#Breaking the data into training and validation data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)
#Selecting columns

categorical_cols = [col for col in X_train_full.columns

                    if X_train_full[col].nunique() < 20 and 

                    X_train_full[col].dtypes=='object']



numerical_cols = [col for col in X_train_full.columns

                 if X_train_full[col].dtypes in ['int64','float64']]



my_cols = categorical_cols + numerical_cols



X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
#Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')



#Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



#Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



#Defining model

model = XGBRegressor(n_estimators=2000, learning_rate=0.03)



#Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])



clf.fit(X_train, y_train)



preds = clf.predict(X_valid)



print(mean_absolute_error(preds, y_valid))
preds_test = clf.predict(X_test)
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)