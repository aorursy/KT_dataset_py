import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
dataset = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
dataset.head()
dataset.describe(include='all')
dataset.isnull().sum()
data_1 = dataset.drop(['host_id', 'id', 'name', 'host_name', 'last_review'], axis =1)
data_1.head()
#Replacing Nan values in the reviews per month column with 0 because there was no review as there was also no value 
#recorded for the corresponding number of reviews. If there was a value in the number of reviews, then there would also 
#be a value for reviews_per_month
data_1['reviews_per_month'].fillna(0, inplace = True)
data_1.isnull().sum()
data_1.columns
#Rearranging the column so that price, the dependent variable would be at the last column
data_1 = data_1[['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',
       'room_type', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365', 'price']]
data_1.describe(include='all')
data_1['price'].skew()
sns.boxplot(data_1['price'])
data_1['minimum_nights'].skew()
sns.boxplot(data_1['minimum_nights'])
data_1['number_of_reviews'].skew()
data_1['reviews_per_month'].skew()
data_1['calculated_host_listings_count'].skew()
sns.boxplot(data_1['calculated_host_listings_count'])
data_1['availability_365'].skew()
data_1['latitude'].skew()
data_1['longitude'].skew()
data_2 = data_1.copy()
data_2['log_price'] = np.log(data_1['price']+1)
data_2.head()
data_2['log_price'].skew()
sns.distplot(data_2['log_price'])
#removing the extreme upper values
q = data_2['minimum_nights'].quantile(0.99)
data_3 = data_2[data_2['minimum_nights']<q]
#removing the extreme upper values
q = data_3['calculated_host_listings_count'].quantile(0.95)
data_4 = data_3[data_3['calculated_host_listings_count']<q]
data_4['minimum_nights'].skew()
data_4['calculated_host_listings_count'].skew()
data_4.describe(include='all')
neighbourhood_group = data_4.groupby(['neighbourhood_group']).agg({'neighbourhood_group':'count'})
#renaming the aggregated column
neighbourhood_group.columns = ['total_airbnb_listings']
neighbourhood_group.sort_values(by=['total_airbnb_listings'], inplace = True, ascending = False)
neighbourhood_group.reset_index(inplace=True)
neighbourhood_group
curr_palette = sns.light_palette('navy',reverse=True)
fig = plt.figure(figsize=(8,6))
sns.barplot(x = 'neighbourhood_group', y = 'total_airbnb_listings', data = neighbourhood_group, palette = curr_palette)
plt.xlabel('Neighbourhood Group', fontsize = 14)
plt.ylabel('Total AirBnB Listings', fontsize = 14)
plt.title('AirBnB Listings by Neighbourhood Group', fontsize = 18)
plt.show()
neighbourhood_top_10 = data_4[['neighbourhood_group','neighbourhood']].groupby(['neighbourhood_group',
                                                                                'neighbourhood']).agg({'neighbourhood':'count'})
neighbourhood_top_10.columns = ['total_airbnb_listings']
neighbourhood_top_10.sort_values(by=['total_airbnb_listings'], inplace = True, ascending = False)
#Getting the top 10 highest listings by neighbourhood
neighbourhood_top_10 = neighbourhood_top_10.nlargest(10,'total_airbnb_listings')
neighbourhood_top_10.reset_index(inplace=True)
neighbourhood_top_10
fig = px.bar(neighbourhood_top_10, x= 'neighbourhood', y='total_airbnb_listings', 
             color='neighbourhood_group',
             title='AirBnB Listings by Neighbourhood',
             labels = {'neighbourhood':'Neighbourhood', 'total_airbnb_listings':'Total AirBnB Listings'},
             hover_name='neighbourhood',
             hover_data=['neighbourhood','neighbourhood_group','total_airbnb_listings'],
             template='plotly_dark',
             width = 800,
             height = 400)

fig.update_xaxes(categoryorder='total descending')
#aligning the title position to center
fig.update(layout = dict(title = dict(x = 0.5)))

fig.show()
airbnb_room_type = data_4[['neighbourhood_group','room_type']].groupby(['neighbourhood_group','room_type']
                                                                             ).agg({'room_type':'count'})
airbnb_room_type.columns = ['count_of_room_type']
airbnb_room_type.reset_index(inplace=True)
airbnb_room_type
fig = px.bar(airbnb_room_type, x= 'neighbourhood_group', y='count_of_room_type', 
             color='room_type',
             title='AirBnB Room Type Distribution by Neighbourhood Groups',
             labels = {'neighbourhood_group':'Neighbourhood Groups', 'count_of_room_type':'Count of Room Type'},
             hover_name='neighbourhood_group',
             hover_data=['neighbourhood_group','room_type','count_of_room_type'],
             barmode='group',
             template='plotly_dark',
             width = 800,
             height = 400)

#fig.update_xaxes(categoryorder='total descending')
#aligning the title position to center
fig.update(layout = dict(title = dict(x = 0.5)))

fig.show()
#airbnb_avg_price = data_4[['neighbourhood_group','room_type']].groupby(['neighbourhood_group','room_type'])
airbnb_avg_price = data_4[['neighbourhood_group','room_type','price']].groupby(['neighbourhood_group','room_type']
                                                                              ).agg({'price':'mean'})
airbnb_avg_price.columns = ['average_price']
airbnb_avg_price.reset_index(inplace=True)
airbnb_avg_price
fig = px.bar(airbnb_avg_price, x= 'neighbourhood_group', y='average_price', 
             color='room_type',
             title='AirBnB Average Prices by Room Type Distribution',
             labels = {'neighbourhood_group':'Neighbourhood Groups', 'average_price':'Average Price'},
             hover_name='neighbourhood_group',
             hover_data=['neighbourhood_group','room_type','average_price'],
             barmode='group',
             template='plotly_dark',
             width = 800,
             height = 400)

#fig.update_xaxes(categoryorder='total descending')
#aligning the title position to center
fig.update(layout = dict(title = dict(x = 0.5)))

fig.show()
airbnb_data = data_4.copy()
airbnb_data.drop(['neighbourhood','price'], axis=1, inplace=True)
airbnb_data.head()
airbnb_data = pd.get_dummies(airbnb_data, drop_first = True)
airbnb_data.head()
airbnb_data.columns
airbnb_data = airbnb_data[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365', 'neighbourhood_group_Brooklyn',
       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',
       'neighbourhood_group_Staten Island', 'room_type_Private room',
       'room_type_Shared room', 'log_price']]
X= airbnb_data.drop(['log_price'], axis =1)
y = airbnb_data['log_price']
lab_enc = LabelEncoder()
y_enc = lab_enc.fit_transform(y)

extra_tree_forest = ExtraTreesClassifier(n_estimators = 20)
extra_tree_forest.fit(X,y_enc)
feature_importance = extra_tree_forest.feature_importances_

plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Labels')
plt.ylabel('Feature Importances')
plt.title('Comparison of different features')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train.shape
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, y_train)
lin_pred = lin_regressor.predict(X_test)

n = X_train.shape[0]
p = X_train.shape[1]
r2 = r2_score(y_test,lin_pred)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,lin_pred)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,lin_pred,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train, y_train)
svr_pred = svr_regressor.predict(X_test)

n = X_train.shape[0]
p = X_train.shape[1]
r2 = r2_score(y_test,svr_pred)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,svr_pred)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,svr_pred,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
dt_regressor = DecisionTreeRegressor(random_state = 0)
dt_regressor.fit(X_train, y_train)
dt_pred = dt_regressor.predict(X_test)

n = X_train.shape[0]
p = X_train.shape[1]
r2 = r2_score(y_test,dt_pred)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,dt_pred)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,dt_pred,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)

n = X_train.shape[0]
p = X_train.shape[1]
r2 = r2_score(y_test,rf_pred)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,rf_pred)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,rf_pred,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

n = X_train.shape[0]
p = X_train.shape[1]
r2 = r2_score(y_test,xgb_pred)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,xgb_pred)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test,xgb_pred,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
airbnb_data2 = airbnb_data.copy()
to_drop = airbnb_data[['neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens', 
                       'neighbourhood_group_Staten Island']]
airbnb_data2.drop(to_drop, axis = 1, inplace=True)
airbnb_data2.head()
X2= airbnb_data2.drop(['log_price'], axis =1)
y2 = airbnb_data2['log_price']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.3, random_state = 42)
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)
X_train2.shape
rf_regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf_regressor2.fit(X_train2, y_train2)
rf_pred2 = rf_regressor2.predict(X_test2)

n = X_train2.shape[0]
p = X_train2.shape[1]
r2 = r2_score(y_test2,rf_pred2)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test2,rf_pred2)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test2,rf_pred2,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
xgb2 = XGBRegressor()
xgb2.fit(X_train2, y_train2)
xgb_pred2 = xgb2.predict(X_test2)

n = X_train2.shape[0]
p = X_train2.shape[1]
r2 = r2_score(y_test2,xgb_pred2)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test2,xgb_pred2)))
print('Root Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test2,xgb_pred2,squared=False)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
score_metric_1 = cross_val_score(estimator = xgb2, X = X_train2, y = y_train2, cv = 10,
                               scoring =  'neg_mean_squared_error')

print("Mean Squared Error: {:.6f}".format(np.abs(score_metric_1.mean()))) #taking the absolute of mse because  
#cross_val_score returns anegative value of mse rather than a positive value.
score_metric_2 = cross_val_score(estimator = xgb2, X = X_train2, y = y_train2, cv = 10,
                               scoring =  'r2')

print("R-Squared: {:.6f}".format(np.abs(score_metric_2.mean())))
#xgboost regressor parameters to be tuned
parameters = {'n_estimators': range(50, 1000, 50),
              'learning_rate': [0.01, 0.05, 0.1, 0.2],
              'max_depth': range(3, 10, 3),
              'min_child_weight': range(1, 10, 2),
              'subsample': [0.6, 0.8, 1]}

random_search = RandomizedSearchCV(estimator = xgb2,
                           param_distributions = parameters,
                           scoring = 'neg_mean_squared_error', #metric we want to use to measure the regression model
                           cv = 10, #evaluated with 10 k cross fold
                           random_state = 42,
                           n_jobs = 1) 

random_search_ = random_search.fit(X_train2, y_train2)
best_mse = random_search_.best_score_
best_parameters = random_search_.best_params_

print("Best MSE: {:.6f}".format(np.absolute(best_mse)))
print("Best Parameters:", best_parameters)
xgb3 = XGBRegressor(subsample = 0.8, n_estimators = 700, min_child_weight = 5, max_depth = 6, learning_rate = 0.05)
xgb3.fit(X_train2, y_train2)
xgb_pred3 = xgb3.predict(X_test2)

n = X_train2.shape[0]
p = X_train2.shape[1]
r2 = r2_score(y_test2,xgb_pred3)
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
print('Mean Squared Error: {:.6f}'.format(mean_squared_error(y_test2,xgb_pred3)))
print('R-Squared: {:.6f}'.format(r2))
print('Adjusted R-Squared: {:.6f}'.format(adjusted_r2))
predicted_values = pd.DataFrame(np.exp(xgb_pred3).round() , columns=['Prediction']) #taking np.exp because price was transformed to 
#log+1 and i have to transform it back to the normal values.
predicted_values.head()
y_test2 = y_test2.reset_index(drop=True)
predicted_values['Actual'] = np.exp(y_test2) #taking np.exp because price was transformed to 
#log+1 and i have to transform it back to the normal values.
predicted_values['Difference (Prediction - Actual)'] = (predicted_values['Prediction'] - predicted_values['Actual']).round()
pd.set_option('display.max_rows', None)
predicted_values