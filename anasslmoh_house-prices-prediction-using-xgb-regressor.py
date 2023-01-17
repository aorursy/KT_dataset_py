import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

train_path = '../input/home-data-for-ml-course/train.csv'

test_path = '../input/home-data-for-ml-course/test.csv'

X = pd.read_csv(train_path,index_col = 'Id')

X_test_full = pd.read_csv(test_path,index_col = 'Id')
X.describe()
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)


from xgboost import XGBRegressor



my_model = XGBRegressor(random_state= 0,reg_lambda = 3,max_depth = 5,) 









# Fit the model

my_model.fit(X_train,y_train) 

prediction1 = my_model.predict(X_valid)

MAE1 = mean_absolute_error(prediction1,y_valid)



print(MAE1)

output = pd.DataFrame({'Id': X_valid.index,

                       'SalePrice': prediction1})

output.to_csv('submission.csv', index=False)