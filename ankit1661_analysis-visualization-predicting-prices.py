import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

import re
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None 

df_initial = pd.read_csv('../input/milan-airbnb-dataset-2020/listings.csv')

# checking shape
print("The dataset has {} rows and {} columns.".format(*df_initial.shape))

# ... and duplicates
print("It contains {} duplicates.".format(df_initial.duplicated().sum()))
df_initial.head(2)
# check the columns we currently have
df_initial.columns
pd.options.display.max_rows = None
df_initial.isna().sum()
# define the columns we want to keep
columns_to_keep = ['id', 'description','host_has_profile_pic',
                   'neighbourhood_cleansed', 'latitude','longitude', 'property_type', 'room_type', 'accommodates',
        'beds', 'amenities', 'price',
       'minimum_nights', 'has_availability',
       'number_of_reviews',
        'review_scores_rating', 'instant_bookable'
       ]
    

df_raw = df_initial[columns_to_keep].set_index('id')
print("The dataset has {} rows and {} columns - after dropping irrelevant columns.".format(*df_raw.shape))


df_raw.room_type.value_counts(normalize=True)
df_raw.property_type.value_counts(normalize=True)
df_raw['price'].head()
# checking Nan's in "price" column
df_raw.price.isna().sum()
# clean up the columns (by method chaining)
df_raw.price = df_raw.price.str.replace('$', '').str.replace(',', '').astype(float)
df_raw['price'].describe()
red_square = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df_raw['price'].plot(kind='box', xlim=(0, 800), vert=False, flierprops=red_square, figsize=(16,2));
df_raw.drop(df_raw[ (df_raw.price > 300) | (df_raw.price == 0) ].index, axis=0, inplace=True)
df_raw['price'].describe()
print("The dataset has {} rows and {} columns - after being price-wise preprocessed.".format(*df_raw.shape))
pd.options.display.max_rows = None
df_raw.isna().sum()
# drop columns with too many Nan's
#df_raw.drop(columns=['neighbourhood', 'bathrooms','neighbourhood_overview' ], inplace=True)
# drop rows with NaN's in bathrooms, host pic, discription, besds, and bedrooms
df_raw.dropna(subset=[  'host_has_profile_pic' ], inplace=True)
#df_raw.host_has_profile_pic.unique()
# replace host_has_profile_pic Nan's with no
#df_raw.host_has_profile_pic.fillna(value='f', inplace=True)
#df_raw.host_has_profile_pic.unique()
pd.options.display.max_rows = None
df_raw.isna().sum()
print("The dataset has {} rows and {} columns - after having dealt with missing values.".format(*df_raw.shape))
df_raw.info()
# filter out sub_df to work with
sub_df = df_raw[['accommodates', 'minimum_nights', 'beds',  'price']]
# split datasets
train_data = sub_df[sub_df['beds'].notnull()]
test_data  = sub_df[sub_df['beds'].isnull()]

# define X
X_train = train_data.drop('beds', axis=1)
X_test  = test_data.drop('beds', axis=1)

# define y
y_train = train_data['beds']
print("Shape of Training Data:", train_data.shape)
print("Shape of Test Data:    ",test_data.shape)
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("\nShape of y_train:", y_train.shape)
# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit model to training data
linreg.fit(X_train, y_train)
# making predictions
y_test = linreg.predict(X_test)
y_test = pd.DataFrame(y_test)
y_test.columns = ['beds']
#y_test = round(y_test.columns)
print(y_test.shape)
y_test.head()
print(X_test.shape)
X_test.head()
# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)
y_test.head()
new_test_data = pd.concat([X_test, y_test], axis=1)
print(new_test_data.shape)
new_test_data.head()
new_test_data['beds'].isna().sum()
# combine train and test data back to a new sub df
sub_df_new = pd.concat([new_test_data, train_data], axis=0)

print(sub_df_new.shape)
sub_df_new.head()
sub_df_new['beds'].isna().sum()
# prepare the multiple columns before concatening
df_raw.drop(['accommodates', 'minimum_nights', 'beds',  'price'], 
            axis=1, inplace=True)
# concate back to complete dataframe
sub_df_new=round(sub_df_new)
df = pd.concat([sub_df_new, df_raw], axis=1)

print(df.shape)
df.head(2)
df['beds'].isna().sum()
df['beds'].describe()
df.drop(df[ (df['beds'] == 0.) | (df['beds'] > 8.) ].index, axis=0, inplace=True)
print("The dataset has {} rows and {} columns - after being engineered.".format(*df.shape))
from geopy.distance import great_circle
def distance_to_mid(lat, lon):
    milan_centre = ( 45.4641, 9.1919)
    accommodation = (lat, lon)
    return great_circle(milan_centre, accommodation).km
df['distance'] = df.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)
df.head(2)
#list(df.description[:10])
df.description.isna().sum()
# extract numbers 
df['size'] = df['description'].str.extract('(\d{2,3}\s?[smSM][q2Q])', expand=True)
df['size'] = df['size'].str.replace("\D", "")

# change datatype of size into float
df['size'] = df['size'].astype(float)

print('NaNs in size_column absolute:     ', df['size'].isna().sum())
print('NaNs in size_column in percentage:', round(df['size'].isna().sum()/len(df),3), '%')
df[['description', 'size']].head(10)
# drop description column
df.drop(['description'], axis=1, inplace=True)
df.info()
# filter out sub_df to work with
sub_df = df[['accommodates', 'beds',  'price', 'minimum_nights','distance', 'size']]
# split datasets
train_data = sub_df[sub_df['size'].notnull()]
test_data  = sub_df[sub_df['size'].isnull()]

# define X
X_train = train_data.drop('size', axis=1)
X_test  = test_data.drop('size', axis=1)

# define y
y_train = train_data['size']
print("Shape of Training Data:", train_data.shape)
print("Shape of Test Data:    ",test_data.shape)
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("\nShape of y_train:", y_train.shape)
# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit model to training data
linreg.fit(X_train, y_train)
# making predictions
y_test = linreg.predict(X_test)
y_test = pd.DataFrame(y_test)
y_test.columns = ['size']
print(y_test.shape)
y_test.head()
print(X_test.shape)
X_test.head()
# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)
y_test.head()
new_test_data = pd.concat([X_test, y_test], axis=1)
new_test_data['size'].isna().sum()
# combine train and test data back to a new sub df
sub_df_new = pd.concat([new_test_data, train_data], axis=0)

print(sub_df_new.shape)
sub_df_new.head()
sub_df_new['size'].isna().sum()
# prepare the multiple columns before concatening
df.drop(['accommodates', 'beds',  'price', 'minimum_nights','distance', 'size'], 
            axis=1, inplace=True)

# concate back to complete dataframe
sub_df_new=round(sub_df_new)
df_new = pd.concat([sub_df_new, df], axis=1)

print(df_new.shape)
df_new.head(2)
df_new['size'].isna().sum()
df_new['size'].describe()
df_new.drop(df_new[ (df_new['size'] == 0.) | (df_new['size'] > 300.) ].index, axis=0, inplace=True)
print("The dataset has {} rows and {} columns - after being engineered.".format(*df_new.shape))
from collections import Counter
results = Counter()
df_new['amenities'].str.strip('{}')\
               .str.replace('"', '')\
               .str.lstrip('\"')\
               .str.rstrip('\"')\
               .str.split(',')\
               .apply(results.update)

results.most_common(30)
# create a new dataframe
sub_df = pd.DataFrame(results.most_common(30), columns=['amenity', 'count'])
# plot the Top 20
sub_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',  
                                                      figsize=(10,7), legend=False, color='darkgrey',
                                                      title='Amenities')
plt.xlabel('Count');
df_new['Heating'] = df_new['amenities'].str.contains('Heating')
df_new['TV'] = df_new['amenities'].str.contains('TV')
df_new['Wifi'] = df_new['amenities'].str.contains('Wifi')
df_new['Kitchen'] = df_new['amenities'].str.contains('Kitchen')
df_new['Essentials'] = df_new['amenities'].str.contains('Essentials')
df_new['Hair dryer'] = df_new['amenities'].str.contains('Hair dryer')
df_new.drop(['amenities'], axis=1, inplace=True)
df_new.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7), 
        c="price", cmap="gist_heat_r", colorbar=True, sharex=False);
df_new['neighbourhood_cleansed'].value_counts().sort_values().plot(kind='barh',figsize=(10,30), color='darkgrey')
plt.title('Number of Accommodations per District');
# group_by neighbourhood groups, take the median price and store new values in sub_df 
df_new_grouped = pd.DataFrame(df_new.groupby(['neighbourhood_cleansed'])['price'].agg(np.median))
df_new_grouped.reset_index(inplace=True)

# plot this 
df_new_grouped.sort_values(by=['price'], ascending=True)\
          .plot(kind='barh', x='neighbourhood_cleansed', y='price', 
                figsize=(10,30), legend=False, color='salmon')

plt.xlabel('\nMedian Price', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nMedian Prices by Neighbourhood\n', fontsize=14, fontweight='bold');
red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')

df_new.boxplot(column='price', by='neighbourhood_cleansed', 
           flierprops=red_square, vert=False, figsize=(10,30))

plt.xlabel('\nMedian Price', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nBoxplot: Prices by Neighbourhood\n', fontsize=14, fontweight='bold')

# get rid of automatic boxplot title
plt.suptitle('');
df_new.plot.scatter(x="distance", y="price", figsize=(9,6), c='dimgrey')
plt.title('\nRelation between Distance & Median Price\n', fontsize=14, fontweight='bold');
sns.jointplot(x=df_new["distance"], y=df_new["price"], kind='hex')
plt.title('\nRelation between Distance & Median Price\n', fontsize=14, fontweight='bold');
plt.figure(figsize=(10,30))
sns.heatmap(df_new.groupby(['neighbourhood_cleansed', 'beds']).price.median().unstack(), 
            cmap='Reds', annot=True, fmt=".0f")

plt.xlabel('\nBeds', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nHeatmap: Median Prices by Neighbourhood and Number of Beds\n\n', fontsize=14, fontweight='bold');
df_new.columns
df_new.info()
df_new.drop([ 'neighbourhood_cleansed','beds','host_has_profile_pic',
     'property_type', 
      'has_availability', 'review_scores_rating',
      'instant_bookable',  'number_of_reviews', 'TV', 'Wifi', 'Kitchen', 'accommodates',
      'Essentials', 'Hair dryer'], axis=1, inplace=True)
for col in [ 'room_type', 'Heating', 'price']:
    df_new[col] = df_new[col].astype('category')
# define our target
target = df_new[["price"]].astype('float64')

# define our features 
features = df_new.drop(["price"], axis=1)
target.info()
num_feats = features.select_dtypes(include=['float64', 'int64', 'bool']).copy()

# one-hot encoding of categorical features
cat_feats = features.select_dtypes(include=['category']).copy()
cat_feats = pd.get_dummies(cat_feats)
features_recoded = pd.concat([num_feats, cat_feats], axis=1)
print(features_recoded.shape)

features_recoded.head(20)
features_recoded.info()
# import train_test_split function
from sklearn.model_selection import train_test_split
# import metrics
from sklearn.metrics import mean_squared_error, r2_score

# split our data
X_train, X_test, y_train, y_test = train_test_split(features_recoded, target, test_size=0.2)
# scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
print(y_train.columns)
y_train.info()

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
booster = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.1, learning_rate=0.05, 
                           max_depth=7, n_estimators=100, random_state=4)

# train
booster.fit(X_train, y_train)

# predict
y_pred_train = booster.predict(X_train)
y_pred_test = booster.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE: {round(RMSE, 4)}")
r2 = r2_score(y_test, y_pred_test)
r2
print(f"r2: {round(r2, 4)}")
xg_train = xgb.DMatrix(data=X_train, label=y_train)
params = {'colsample_bytree':0.6, 'gamma':0.2, 'learning_rate':0.05, 'max_depth':6}

cv_results = xgb.cv(dtrain=xg_train, params=params, nfold=3,
                    num_boost_round=200, early_stopping_rounds=10, 
                    metrics="rmse", as_pandas=True)
cv_results.head()
cv_results.tail()
# plot the important features
feat_importances = pd.Series(booster.feature_importances_, index=features_recoded.columns)
feat_importances.nlargest(15).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))
plt.xlabel('Relative Feature Importance with XGBoost');
