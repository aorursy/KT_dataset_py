# import libraries 

import pandas as pd

import numpy as np

import xgboost as xgb



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



import warnings

warnings.filterwarnings("ignore")
df_test = pd.read_csv('../input/test.csv')

df_test.info()
df_test.head(2)
df_test.isna().sum()/len(df_test)
# why is user score stored as string?

df_test.User_Score.unique()
# replace 'tbd'

df_test['User_Score'].replace(to_replace='tbd', value=np.nan, inplace=True)

# convert string values into numeric values

df_test['User_Score'] = pd.to_numeric(df_test['User_Score'])



df_test.info()
df_test.duplicated().sum()
df_train = pd.read_csv('../input/train.csv')

df_train.info()
df_train.head(2)
df_train.isna().sum()/len(df_train)
# replace 'tbd'

df_train['User_Score'].replace(to_replace='tbd', value=np.nan, inplace=True)



# convert string values into numeric values

df_train['User_Score'] = pd.to_numeric(df_train['User_Score'])



df_train.info()
df_train.duplicated().sum()
plt.subplots(figsize=(8, 6))

sns.heatmap(df_train.corr(), cmap="Oranges", linewidths=0.1);
df_train = df_train.drop(['Publisher', 'Developer'], axis=1)

df_test  = df_test.drop(['Publisher', 'Developer'], axis=1)
print('Train Data:')

print('------------------------------------------')

df_train.info()
df_train = df_train.dropna(subset=['Genre'])



df_train.Genre = df_train.Genre.astype('category')



df_train.Genre.value_counts()
print('Test Data:')

print('------------------------------------------')

df_test.info()
df_test.Genre = df_test.Genre.astype('category')

df_test.Genre.value_counts()
df_train.Rating.unique()
df_train.Rating.value_counts()
def value_replacement(col, to_replace, new_value):

    col.replace(to_replace, new_value, inplace=True)
value_replacement(df_train.Rating, to_replace='EC', new_value='E')

value_replacement(df_train.Rating, to_replace='K-A', new_value='E')

value_replacement(df_train.Rating, to_replace='RP', new_value='None')

value_replacement(df_train.Rating, to_replace=np.nan, new_value='None')



df_train.Rating.value_counts()
df_train.Rating = df_train.Rating.astype('category')
df_test.Rating.value_counts()
value_replacement(df_test.Rating, to_replace='EC', new_value='E')

value_replacement(df_test.Rating, to_replace='AO', new_value='M')

value_replacement(df_test.Rating, to_replace='K-A', new_value='E')

value_replacement(df_test.Rating, to_replace='RP', new_value='None')

value_replacement(df_test.Rating, to_replace=np.nan, new_value='None')



df_test.Rating.value_counts()
df_test.Rating = df_test.Rating.astype('category')
df1 = df_train[df_train['Year_of_Release'].isna()]

df1.tail()
df_train.Year_of_Release.max()
df_test.Year_of_Release.max()
df_train.Year_of_Release.hist(bins=20);
# bin the year_of_release into periods_of_release

bins = [1980, 1995, 2000, 2005, 2010, 2015, 2017]

labels = ['Before 1995', '1995-2000', '2000-2005', '2005-2010', '2010-2015', '2015-2020']

df_train['Periods_of_Release'] = pd.cut(df_train['Year_of_Release'], bins=bins, labels=labels)



# create another category for the unknown release date

df_train['Periods_of_Release'].replace(to_replace=np.nan, value='Unknown', inplace=True)
df_train.Periods_of_Release.value_counts()
# visualize the distribution of categories

order = ['Unknown', '2015-2020', '2010-2015', '2005-2010',  '2000-2005','1995-2000', 'Before 1995']

df_train.Periods_of_Release.value_counts().loc[order].plot(kind='barh');
# drop the original year related column

df_train = df_train.drop(['Year_of_Release'], axis=1)



df_train.Periods_of_Release = df_train.Periods_of_Release.astype('category')



df_train.info()
df_test['Periods_of_Release'] = pd.cut(df_test['Year_of_Release'], bins=bins, labels=labels)



# create another category for the unknown release date

df_test['Periods_of_Release'].replace(to_replace=np.nan, value='Unknown', inplace=True)



df_test = df_test.drop(['Year_of_Release'], axis=1)



df_test.Periods_of_Release = df_test.Periods_of_Release.astype('category')



df_test.info()
df_train.info()
df_train = df_train.drop(['Critic_Count', 'User_Score'], axis=1)
df_train.User_Count.describe()
df_train.User_Count.plot(kind='box', vert=False, xlim=(4,500));
# removing outliers

df_train = df_train.drop(df_train[df_train.User_Count > 200].index, axis=0)

df_train.User_Count.describe()
df_train.info()
# filter out sub_df to work with

#sub_df = df[['NA_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']]

sub_df = df_train[['JP_Sales', 'Genre', 'Rating', 'User_Count']]



# split datasets

train_data = sub_df[sub_df['User_Count'].notnull()]

test_data  = sub_df[sub_df['User_Count'].isnull()]



# define X

X_train = train_data.drop('User_Count', axis=1)

X_train = pd.get_dummies(X_train)



X_test  = test_data.drop('User_Count', axis=1)

X_test  = pd.get_dummies(X_test)



# define y

y_train = train_data['User_Count']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.fit_transform(X_test)

X_test_scaled = pd.DataFrame(X_test_scaled)
# import Linear Regression

from sklearn.linear_model import LinearRegression



# instantiate

linreg_user_score = LinearRegression()



# fit model to training data

linreg_user_score.fit(X_train_scaled, y_train)



# making predictions

y_test = linreg_user_score.predict(X_test_scaled)
# preparing y_test

y_test = pd.DataFrame(y_test)

y_test.columns = ['User_Count']

print(y_test.shape)

y_test.head(2)
# preparing X_test

print(X_test.shape)

X_test.head(2)
# make the index of X_test to an own dataframe

prelim_index = pd.DataFrame(X_test.index)

prelim_index.columns = ['prelim']



# ... and concat this dataframe with y_test

y_test = pd.concat([y_test, prelim_index], axis=1)

y_test.set_index(['prelim'], inplace=True)



# finally combine the new test data

test_data = pd.concat([X_test, y_test], axis=1)



# combine train and test data back to a new sub df

sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)



print(sub_df_new.shape)

sub_df_new.head(2)
# drop duplicate columns in dataframe before concatening 

df_train.drop(['User_Count'], axis=1, inplace=True)

sub_df_new = sub_df_new[['User_Count']]



# concatenate back to complete dataframe

df_train_1 = pd.concat([sub_df_new, df_train], axis=1)



#print(df_train.shape)

df_train_1.head(2)
df_train_1.User_Count.isna().sum()
df_train_1.User_Count.describe()
df_test.info()
df_test = df_test.drop(['Critic_Count', 'User_Score'], axis=1)
df_test.User_Count.describe()
# filter out sub_df to work with

sub_df = df_test[['JP_Sales', 'Genre', 'Rating', 'User_Count']]



# split datasets

train_data = sub_df[sub_df['User_Count'].notnull()]

test_data  = sub_df[sub_df['User_Count'].isnull()]



# define X

X_train = train_data.drop('User_Count', axis=1)

X_train = pd.get_dummies(X_train)



X_test  = test_data.drop('User_Count', axis=1)

X_test  = pd.get_dummies(X_test)



# define y

y_train = train_data['User_Count']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.fit_transform(X_test)

X_test_scaled = pd.DataFrame(X_test_scaled)
# import Linear Regression

from sklearn.linear_model import LinearRegression



# instantiate

linreg_user_score = LinearRegression()



# fit model to training data

linreg_user_score.fit(X_train_scaled, y_train)



# making predictions

y_test = linreg_user_score.predict(X_test_scaled)
# preparing y_test

y_test = pd.DataFrame(y_test)

y_test.columns = ['User_Count']

print(y_test.shape)

y_test.head(2)
# preparing X_test

print(X_test.shape)

X_test.head(2)
# make the index of X_test to an own dataframe

prelim_index = pd.DataFrame(X_test.index)

prelim_index.columns = ['prelim']



# ... and concat this dataframe with y_test

y_test = pd.concat([y_test, prelim_index], axis=1)

y_test.set_index(['prelim'], inplace=True)



# finally combine the new test data

test_data = pd.concat([X_test, y_test], axis=1)



# combine train and test data back to a new sub df

sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)



print(sub_df_new.shape)

sub_df_new.head(2)
# drop duplicate columns in dataframe before concatening 

df_test.drop(['User_Count'], axis=1, inplace=True)

sub_df_new = sub_df_new[['User_Count']]



# concatenate back to complete dataframe

df_test_1 = pd.concat([sub_df_new, df_test], axis=1)



#print(df_train.shape)

df_test_1.head(2)
df_test_1.User_Count.isna().sum()
df_test_1.User_Count.describe()
df_test_1.info()
df_train_1.info()
df_train_1.Critic_Score.describe()
df_train_1.Critic_Score.plot(kind='box', vert=False, xlim=(0,100));
# removing outliers

df_train_1 = df_train_1.drop(df_train_1[df_train_1.Critic_Score < 30].index, axis=0)

df_train_1.Critic_Score.describe()
df_train_1.info()
# filter out sub_df to work with

#sub_df = df[['NA_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']]

sub_df = df_train_1[['JP_Sales', 'Genre', 'Rating', 'Critic_Score']]



# split datasets

train_data = sub_df[sub_df['Critic_Score'].notnull()]

test_data  = sub_df[sub_df['Critic_Score'].isnull()]



# define X

X_train = train_data.drop('Critic_Score', axis=1)

X_train = pd.get_dummies(X_train)



X_test  = test_data.drop('Critic_Score', axis=1)

X_test  = pd.get_dummies(X_test)



# define y

y_train = train_data['Critic_Score']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.fit_transform(X_test)

X_test_scaled = pd.DataFrame(X_test_scaled)
# import Linear Regression

from sklearn.linear_model import LinearRegression



# instantiate

linreg_user_score = LinearRegression()



# fit model to training data

linreg_user_score.fit(X_train_scaled, y_train)



# making predictions

y_test = linreg_user_score.predict(X_test_scaled)
# preparing y_test

y_test = pd.DataFrame(y_test)

y_test.columns = ['Critic_Score']

print(y_test.shape)

y_test.head(2)
# preparing X_test

print(X_test.shape)

X_test.head(2)
# make the index of X_test to an own dataframe

prelim_index = pd.DataFrame(X_test.index)

prelim_index.columns = ['prelim']



# ... and concat this dataframe with y_test

y_test = pd.concat([y_test, prelim_index], axis=1)

y_test.set_index(['prelim'], inplace=True)



# finally combine the new test data

test_data = pd.concat([X_test, y_test], axis=1)



# combine train and test data back to a new sub df

sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)



print(sub_df_new.shape)

sub_df_new.head(2)
# drop duplicate columns in dataframe before concatening 

df_train_1.drop(['Critic_Score'], axis=1, inplace=True)

sub_df_new = sub_df_new[['Critic_Score']]



# concatenate back to complete dataframe

df_train_2 = pd.concat([sub_df_new, df_train_1], axis=1)



#print(df_train.shape)

df_train_2.head(2)
df_train_2.Critic_Score.isna().sum()
df_train_2.Critic_Score.describe()
df_train_2.info()
df_test_1.info()
df_test_1.Critic_Score.describe()
# filter out sub_df to work with

#sub_df = df[['NA_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']]

sub_df = df_test_1[['JP_Sales', 'Genre', 'Rating', 'Critic_Score']]



# split datasets

train_data = sub_df[sub_df['Critic_Score'].notnull()]

test_data  = sub_df[sub_df['Critic_Score'].isnull()]



# define X

X_train = train_data.drop('Critic_Score', axis=1)

X_train = pd.get_dummies(X_train)



X_test  = test_data.drop('Critic_Score', axis=1)

X_test  = pd.get_dummies(X_test)



# define y

y_train = train_data['Critic_Score']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.fit_transform(X_test)

X_test_scaled = pd.DataFrame(X_test_scaled)
# import Linear Regression

from sklearn.linear_model import LinearRegression



# instantiate

linreg_user_score = LinearRegression()



# fit model to training data

linreg_user_score.fit(X_train_scaled, y_train)



# making predictions

y_test = linreg_user_score.predict(X_test_scaled)
# preparing y_test

y_test = pd.DataFrame(y_test)

y_test.columns = ['Critic_Score']

print(y_test.shape)

y_test.head(2)
# preparing X_test

print(X_test.shape)

X_test.head(2)
# make the index of X_test to an own dataframe

prelim_index = pd.DataFrame(X_test.index)

prelim_index.columns = ['prelim']



# ... and concat this dataframe with y_test

y_test = pd.concat([y_test, prelim_index], axis=1)

y_test.set_index(['prelim'], inplace=True)



# finally combine the new test data

test_data = pd.concat([X_test, y_test], axis=1)



# combine train and test data back to a new sub df

sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)



print(sub_df_new.shape)

sub_df_new.head(2)
# drop duplicate columns in dataframe before concatening 

df_test_1.drop(['Critic_Score'], axis=1, inplace=True)

sub_df_new = sub_df_new[['Critic_Score']]



# concatenate back to complete dataframe

df_test_2 = pd.concat([sub_df_new, df_test_1], axis=1)



#print(df_train.shape)

df_test_2.head(2)
df_test_2.Critic_Score.isna().sum()
df_test_2.Critic_Score.describe()
df_test_2.info()
#df_train_2.Platform.value_counts()
#df_test_2.Platform.value_counts()
#df_train_final = df_train_final.groupby('Platform').filter(lambda x: len(x) > 100)
df_train_2 = df_train_2.drop(['Platform'], axis=1)
df_test_2 = df_test_2.drop(['Platform'], axis=1)
df_training = df_train_2.drop(['Id'], axis=1)

df_training.info()
# define our features 

features = df_training.drop(['NA_Sales'], axis=1)



# define our target

target = df_training[['NA_Sales']]
# create dummy variables of all categorical features

features = pd.get_dummies(features)
# import train_test_split function

from sklearn.model_selection import train_test_split



# split our data

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=40)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train) 

X_test = scaler.transform(X_test)
# create a baseline

booster = xgb.XGBRegressor()
from sklearn.model_selection import GridSearchCV



# create Grid

param_grid = {'n_estimators': [100, 150, 200],

              'learning_rate': [0.01, 0.05, 0.1], 

              'max_depth': [3, 4, 5, 6, 7],

              'colsample_bytree': [0.6, 0.7, 1],

              'gamma': [0.0, 0.1, 0.2]}



# instantiate the tuned random forest

booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)



# train the tuned random forest

booster_grid_search.fit(X_train, y_train)



# print best estimator parameters found during the grid search

print(booster_grid_search.best_params_)
# instantiate xgboost with best parameters

booster = xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.2, learning_rate=0.05, 

                             max_depth=6, n_estimators=100, random_state=4)



# train

booster.fit(X_train, y_train)



# predict

y_pred_train = booster.predict(X_train)

y_pred_test  = booster.predict(X_test)
# import metrics

from sklearn.metrics import mean_squared_error, r2_score



RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"RMSE: {round(RMSE, 4)}")



r2 = r2_score(y_test, y_pred_test)

print(f"r2: {round(r2, 4)}")
booster.score(X_test, y_test)
# plot the important features

feat_importances = pd.Series(booster.feature_importances_, index=features.columns)

feat_importances.nlargest(15).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))

plt.xlabel('Relative Feature Importance with XGBoost');
df_testing = df_test_2.drop(['Id'], axis=1)

#df_testing = df_test_2

df_testing.info()
df_testing.head()
# create dummy variables of all categorical features

test_features = pd.get_dummies(df_testing)

test_features.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(test_features)

test_features_scaled = scaler.transform(test_features) 
#booster.fit(test_features_scaled)

predictions = booster.predict(test_features_scaled)

predictions = pd.DataFrame(predictions)



predictions.columns = ['Prediction']

predictions.head()
df_submission = pd.merge(df_test_2, predictions, left_index=True, right_index=True)

df_submission = df_submission[['Id', 'Prediction']]

df_submission.head()
df_submission.to_csv('df_submission.csv', index=False)