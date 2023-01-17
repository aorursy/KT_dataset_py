# Exercise for Iowa data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



iowa_data_file = "../input/house-prices-advanced-regression-techniques/train.csv"
iowa_data = pd.read_csv(iowa_data_file)




iowa_target = iowa_data.SalePrice
iowa_feats = iowa_data.drop(["SalePrice" ], axis=1)
iowa_feats_num = iowa_feats.select_dtypes(exclude=["object"])

X_train, X_test, y_train, y_test = train_test_split(iowa_feats_num, iowa_target,train_size=0.7, test_size=0.3, random_state=0 )

# missing = iowa_test_feats_num.isnull().sum()
# print(missing[missing > 0])

# function to return MAE

def score_MAE(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(preds, y_test)


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
# Solution 1 : Drop cols with missing values


cols_missing_val  = [col for col in X_train.columns
                        if X_train[col].isnull().any()
                    ]

sol1_train_X = X_train.drop(cols_missing_val, axis=1)
sol1_test_X = X_test.drop(cols_missing_val, axis=1)

print("Solution 1 : dropping cols with missing value, MAE ={:.02f}".format(score_MAE(sol1_train_X,sol1_test_X,y_train,y_test)))


# Solution 2 : using Simple Imputer


myimputer = SimpleImputer()
sol2_train_X = myimputer.fit_transform(X_train)
sol2_test_X = myimputer.transform(X_test)

print("Solution 2 : dropping cols with missing value, MAE ={:.02f}".format(score_MAE(sol2_train_X,sol2_test_X,y_train,y_test)))


# Solution 3 : using Simple Imputer with Extension


sol3_train_X_plus = X_train.copy()
sol3_test_X_plus = X_test.copy()

for col in cols_missing_val:
    sol3_train_X_plus[col+ "_was_missing"] = sol3_train_X_plus[col].isnull()
    sol3_test_X_plus[col+ "_was_missing"] = sol3_test_X_plus[col].isnull()
    
# for col in cols_missing_val:
#     sol3_train_X_plus.loc[:,col+ "_was_missing"] = sol3_train_X_plus[col].isnull()
#     sol3_test_X_plus.loc[:,col+ "_was_missing"] = sol3_test_X_plus[col].isnull()

    
sol3_train_X = myimputer.fit_transform(sol3_train_X_plus)
sol3_test_X = myimputer.transform(sol3_test_X_plus)

print("Solution 2 : dropping cols with missing value, MAE ={:.02f}".format(score_MAE(sol3_train_X,sol3_test_X,y_train,y_test)))

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################