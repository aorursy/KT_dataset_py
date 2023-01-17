# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex1 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

X = X_full[my_cols].copy()
X_train.head()
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.linear_model import Lasso



def rf_model(n_estimators=100, criterion='mae', random_state=0):

    return RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, random_state=random_state)



def xgb_model(n_estimators=1000, learning_rate=0.05, n_jobs=4):

    return XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs)



def lasso_model(alpha=0.0005, random_state=5):

    return Lasso(alpha=alpha, random_state=random_state)

## Option 1 : drop cols with missing data



# Number of missing values in each column of training data

#missing_val_count_by_column = (X.isnull().sum())

#missing_val_count_by_column_test = (X_test.isnull().sum())



# Fill in the line below: get names of columns with missing values

#cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].keys()

#cols_with_missing_test = missing_val_count_by_column_test[missing_val_count_by_column_test > 0].keys()

#cols_with_missing = cols_with_missing.union(cols_with_missing_test)

#print(cols_with_missing)



# Fill in the lines below: drop columns in training and validation data

#X_train = X_train.drop(cols_with_missing, axis=1)

#X_valid = X_valid.drop(cols_with_missing, axis=1)

#X_test = X_test.drop(cols_with_missing, axis=1)

##X = X.drop(cols_with_missing, axis=1)



#X_train.head()



# MEA Only is : 18935



## Option 2 : Imputation of missing data



#from sklearn.impute import SimpleImputer



# Imputation

#final_imputer = SimpleImputer(strategy='median')

#X_train = pd.DataFrame(final_imputer.fit_transform(X_train))

#X_valid = pd.DataFrame(final_imputer.transform(X_valid))

#X_test = pd.DataFrame(final_imputer.transform(X_test))

#X = pd.DataFrame(final_imputer.transform(X))



# Imputation removed column names; put them back

#X_train.columns = X_train.columns

#X_valid.columns = X_valid.columns

#X_test.columns = X_test.columns

#X.columns = X.columns



# MEA is : 18093
## Option 3 : Label Encoding



#from sklearn.preprocessing import LabelEncoder



# Get list of categorical variables

#s = (X.dtypes == 'object')

#object_cols = list(s[s].index)



#print("Categorical variables:")

#print(object_cols)



# Columns that can be safely label encoded

#good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

#print("good labels:")

#print(good_label_cols)



# Problematic columns that will be dropped from the dataset

#bad_label_cols = list(set(object_cols)-set(good_label_cols))

#print("bad labels:")

#print(bad_label_cols)



# Drop categorical columns that will not be encoded

#X_train = X_train.drop(bad_label_cols, axis=1)

#X_valid = X_valid.drop(bad_label_cols, axis=1)

#X_test = X_test.drop(bad_label_cols, axis=1)

#X = X.drop(bad_label_cols, axis=1)



# Apply label encoder to each column with categorical data

#label_encoder = LabelEncoder()

#for col in good_label_cols:

#    X_train[col] = label_encoder.fit_transform(X_train[col])

#    X_valid[col] = label_encoder.transform(X_valid[col])

#    X_test[col] = label_encoder.transform(X_test[col])

#    X[col] = label_encoder.transform(X[col])



# MAE is : 18607 (combined with drop missing cols)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Bundle preprocessing and modeling code in a pipeline

def clf(n_estimators=100, criterion='mae', random_state=0):

    return  Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', rf_model(n_estimators=n_estimators, criterion=criterion, random_state=random_state))

                     ])

# MAE : 17772



# Bundle preprocessing and modeling code in a pipeline

def xgb_clf(n_estimators=1000, learning_rate=0.05, n_jobs=4):

    return  Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', xgb_model(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs ))

                     ])



# MAE: 16499



def lasso_clf(alpha=0.0005, random_state=5):

    return  Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', lasso_model(alpha=alpha, random_state=random_state))

                     ])



# MAE: 

from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(mdl, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    mdl.fit(X_t, y_t)

    preds = mdl.predict(X_v)

    return mean_absolute_error(y_v, preds)



mae = score_model(clf(n_estimators=100, criterion='mae', random_state=0))

print("Random forset Regression Model's MAE: %d" % (mae))



mae = score_model(xgb_clf(n_estimators=1000, learning_rate=0.05, n_jobs=4))

print("Gradient boosting Model's MAE: %d" % (mae))



mae = score_model(lasso_clf(alpha=0.00005, random_state=5))

print("Lasso Model's MAE: %d" % (mae))

from sklearn.model_selection import cross_val_score



def get_score_cross_validation(n_estimators):

    test_pipeline = xgb_clf(n_estimators=n_estimators)

    scores = -1 * cross_val_score(test_pipeline, X, y,

                                  cv=3,

                                  scoring='neg_mean_absolute_error')

    return scores.mean()

    pass



results = {}

for i in range(1,20):

    results[50*i] = get_score_cross_validation(50*i)



# results for  rf_model are {50: 17989.90902711937, 100: 17951.248890445127, 150: 17905.961725587727, 200: 17873.74381446413, 250: 17912.836598102094, 300: 17939.665974828862, 350: 18000.571310498595, 400: 17993.435347382427}

# So, we conclude that 200 is the best value of estimators



# results for  xgb_model are {50: 23154.361754126232, 100: 17392.48273273184, 150: 16586.630607658255, 200: 16291.770399811685, 250: 16143.857207045168, 300: 16074.786700873738, 350: 16011.175154920638, 400: 15983.171869889198, 450: 15964.655717794165, 500: 15954.15480163433, 550: 15949.418943565805, 600: 15947.995305519818, 650: 15936.35036284181, 700: 15922.79011287209, 750: 15926.222763224805, 800: 15916.359328436118, 850: 15914.507840491146, 900: 15912.554610106248, 950: 15916.544758285245}

# So, we conclude that 900 is the best value of estimators



import matplotlib.pyplot as plt

%matplotlib inline



print(results)

plt.plot(results.keys(), results.values())

plt.show()

# Fit the model to the training data

#my_pipeline = clf(n_estimators=200)

#my_pipeline = xgb_clf(n_estimators=900)

my_pipeline = lasso_clf(alpha=0.00005, random_state=5)



my_pipeline.fit(X, y)



# Generate test predictions

preds_test = my_pipeline.predict(X_test)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)