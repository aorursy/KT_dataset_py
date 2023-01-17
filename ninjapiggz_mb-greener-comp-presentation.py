import pandas as pd



# Read the data (updated to use data stored on Kaggle rather than your local storage)

raw_train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')

raw_test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv.zip')



raw_train.set_index('ID', inplace=True)

raw_test.set_index('ID', inplace=True)



raw_full = pd.concat([raw_train, raw_test], axis=0)

raw_full['y'].fillna('predict', inplace=True)
raw_full.sort_index(inplace=True)
# number of rows with a missing value in any column

print(len(raw_full.loc[raw_full.isnull().any(axis=1)]))



# good news is that there is no missing data, so don't need to worry about that



# remove y values from training samples

# y = raw_full.y

# raw_full.drop(['y'], axis=1, inplace=True)
# determine which features are categorical, binary, or contain no information (will be needed for feature engineering later)



# note: must use the training set to evaluate which features have no information (e.g. if the feature value is the same for every sample in training, then even if it varies in test we don't know anything about it's effect on y)



constant_features = []

binary_features = []

categorical_features = []

for c_name in raw_train.columns:

    if c_name == 'ID' or c_name == 'y':

        continue

    unique_values_in_column = len(set(raw_train[c_name]))

    if unique_values_in_column > 2:

        categorical_features.append(c_name)

    elif unique_values_in_column == 2:

        binary_features.append(c_name)

    elif unique_values_in_column == 1:

        constant_features.append(c_name)
# we will use pandas get_dummies on all categorical features (look this up and we can chat about how it works)



# applying this to the full dataframe of training and testing saves time because we can do it all in one step (there are some potential downsides to this, we can chat about)



categorical_feature_dfs = []

for cf in categorical_features:

    cf_df = pd.get_dummies(raw_full[cf], prefix=cf)

    categorical_feature_dfs.append(cf_df)
categorical_feature_dfs[0]
categorical_feature_final = pd.concat(categorical_feature_dfs, axis=1)
# binary features don't need any feature engineering, so just concatenate these to the categorical dataframe we just created



final_df = pd.concat([raw_full['y'], categorical_feature_final, raw_full.loc[:, binary_features]], axis=1)
X_train_full = final_df.loc[final_df['y'] != 'predict', final_df.columns != 'y']

y_train_full = final_df.loc[final_df['y'] != 'predict', 'y']

X_test_full = final_df.loc[final_df['y'] == 'predict', final_df.columns != 'y']
from sklearn.model_selection import train_test_split



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, 

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)







# Select categorical columns with relatively low cardinality (convenient but arbitrary)

# categorical_cols = [cname for cname in X_train_full.columns if

#                     X_train_full[cname].nunique() < 80 and 

#                     X_train_full[cname].dtype == "object"]



# Select numerical columns

# numerical_cols = [cname for cname in X_train_full.columns if 

#                 X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

# my_cols = categorical_cols + numerical_cols

# X_train = X_train_full[my_cols].copy()

# X_valid = X_valid_full[my_cols].copy()

# X_test = X_test_full[my_cols].copy()



# X_train.head(10)



##creat a kaggle kernel, figure out what the shakeup of the rankings are, 11th and 3rd place solutions are posted too 
import xgboost as xgb



train_data = xgb.DMatrix(data=X_train, label=y_train)

valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'silent': 1

}



model = xgb.train(xgb_params, train_data, num_boost_round=1000, evals=[(valid_data, 'Valid')], verbose_eval=1, early_stopping_rounds=5)
test_data = xgb.DMatrix(data=X_test_full)

test_preds = model.predict(test_data)
sample_submision = pd.read_csv('../input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')

sample_submision
# create submission file



X_test_full['y'] = test_preds

X_test_full.reset_index(inplace=True)

submission = X_test_full.loc[:, ['ID', 'y']]

submission.to_csv('submission.csv', index=False)
# from sklearn.compose import ColumnTransformer

# from sklearn.pipeline import Pipeline

# from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import OneHotEncoder

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.metrics import mean_absolute_error

# from sklearn.metrics import r2_score

# from sklearn.model_selection import cross_val_score

# import xgboost as xgb





# # # Preprocessing for numerical data

# # numerical_transformer = SimpleImputer(strategy='constant')



# # # Preprocessing for categorical data

# # categorical_transformer = Pipeline(steps=[

# #     ('imputer', SimpleImputer(strategy='most_frequent')),

# #     ('onehot', OneHotEncoder(handle_unknown='ignore'))

# # ])



# # # Bundle preprocessing for numerical and categorical data

# # preprocessor = ColumnTransformer(

# #     transformers=[

# #         ('num', numerical_transformer, numerical_cols),

# #         ('cat', categorical_transformer, categorical_cols)

# #     ])



# xgb_params = {

#     'eta': 0.05,

#     'max_depth': 6,

#     'subsample': 0.7,

#     'colsample_bytree': 0.7,

#     'objective': 'reg:linear',

#     'silent': 1

# }



# model = xgb.XGBRegressor(eta=0.05, max_depth=6,subsample=0.7,colsample_bytree=0.7, random_state=0)



# ##why am i having a hard time modifying my model when I use a pipeline?



# # Bundle preprocessing and modeling code in a pipeline

# clf = Pipeline(steps=[('preprocessor', preprocessor),

#                       ('model', model)

#                      ])



# # Preprocessing of training data, fit model 

# clf.fit(X_train, y_train,

#        )



# # Preprocessing of validation data, get predictions

# preds = clf.predict(X_valid)



# print('MAE:', mean_absolute_error(y_valid, preds))



# #Calculating r^2 value

# r2_score(y_valid,preds)

# #Cross Validation

# scores = -1 * cross_val_score(clf, X_train, y_train,

#                               cv=5,

#                               scoring='neg_mean_absolute_error')



# print("Average MAE score:", scores.mean())




#Cross validation plotting: i am supposed to get the lowest mae model?

#how do i impliment this into my model?

#help get the estimator values down. 



# def get_score(n_estimators):

#     my_pipeline = Pipeline(steps=[

#         ('preprocessor', SimpleImputer()),

#         ('model', RandomForestRegressor(n_estimators, random_state=0))

#     ])

#     scores = -1 * cross_val_score(my_pipeline, X_train[numerical_cols], y_train,

#                                   cv=5,

#                                   scoring='neg_mean_absolute_error')

#     return scores.mean()



# results={}

# for i in range(1,9):

#      results[50*i]=get_score(50*i)

# print (results)