# packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load data

modelling_data_feat = np.load('../input/x_train.npy')

modelling_data_target = np.load('../input/y_train.npy')

submit_data_feat = np.load('../input/x_test.npy')



# convert ndarray to dataframe

df_mod_feat = pd.DataFrame(data=modelling_data_feat)

df_sub_feat = pd.DataFrame(data=submit_data_feat)



rs = 4319
df_mod_feat.describe()
df_sub_feat.describe()
# packages

from sklearn.preprocessing import OneHotEncoder



# drop date, waterfront, lat, and long

df_mod_feat = df_mod_feat.drop(columns=['waterfront','lat','long'])

df_sub_feat = df_sub_feat.drop(columns=['waterfront','lat','long'])



# split feature list into categorical and numerical

df_mod_feat_cat = df_mod_feat[['zipcode']].copy()

df_mod_feat_num = df_mod_feat.drop(columns=['zipcode'])

df_sub_feat_cat = df_sub_feat[['zipcode']].copy()

df_sub_feat_num = df_sub_feat.drop(columns=['zipcode'])



# convert categorical feature(s) into nominal numeric features

# merge train set with test set first to convert to associative values

cat_allfeat = pd.concat([df_mod_feat_cat, df_sub_feat_cat])

enc = OneHotEncoder()

temp = enc.fit_transform(cat_allfeat)

temp = pd.DataFrame(temp.todense())

df_mod_feat_cat_conv = temp.iloc[:df_mod_feat_cat.shape[0]]

df_sub_feat_cat_conv = temp.iloc[df_mod_feat_cat.shape[0]:cat_allfeat.shape[0]]

df_sub_feat_cat_conv = df_sub_feat_cat_conv.reset_index(drop=True)



# if yr_renovated == 0, yr_renovated = yr_built

for i in range (0, df_mod_feat_num.shape[0]):

    df_mod_feat_num.loc[i,'date'] = int(df_mod_feat_num.loc[i,'date'][0:4])

    if df_mod_feat_num.loc[i,'yr_renovated'] == 0:

        df_mod_feat_num.loc[i,'yr_renovated'] = df_mod_feat_num.loc[i,'yr_built']

        

for i in range (0, df_sub_feat_num.shape[0]):

    df_sub_feat_num.loc[i,'date'] = int(df_sub_feat_num.loc[i,'date'][0:4])

    if df_sub_feat_num.loc[i,'yr_renovated'] == 0:

        df_sub_feat_num.loc[i,'yr_renovated'] = df_sub_feat_num.loc[i,'yr_built']



# merge converted categorical and numerical features

df_mod_feat = pd.concat([df_mod_feat_cat_conv, df_mod_feat_num], axis=1)

df_sub_feat = pd.concat([df_sub_feat_cat_conv, df_sub_feat_num], axis=1)



# scaling



# normalize

# packages

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.feature_selection import SelectKBest, RFE, RFECV, SelectFromModel, f_regression

from sklearn.linear_model import LinearRegression, BayesianRidge



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# Split train set and test set for modelling

X_train, X_test, y_train, y_test = train_test_split(df_mod_feat, modelling_data_target, test_size=0.1, random_state=rs)



# Select a regression model

# LinearRegression

model = LinearRegression(fit_intercept=False)



# Feature selection via SelectKBest



# Feature selection via RFE (Recursive Feature Elimination)



# Feature selection via RFECV (Recursive Feature Elmination with Cross Validation)

selector = RFECV(estimator=model, step=1, min_features_to_select=1, cv=StratifiedKFold(n_splits=5, random_state=rs),scoring='r2')

feat_res = selector.fit(X_train, y_train)

print("Optimal number of features : %d" % selector.n_features_)



# Feature selection via SelectFromModel



# The selected features

sel_X_train = feat_res.transform(X_train)

sel_X_test = feat_res.transform(X_test)



y_true, y_pred = y_test, feat_res.estimator_.predict(sel_X_test)

mean_absolute_percentage_error(y_true, y_pred)
submission_features = feat_res.transform(df_sub_feat)

test_predictions = feat_res.estimator_.predict(submission_features)

submission = pd.DataFrame({'Id': range(1, test_predictions.shape[0]+1), 'Price': test_predictions})

submission = submission.reset_index(drop=True)

submission.to_csv('submission.csv', index=False)

submission