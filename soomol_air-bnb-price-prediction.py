import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from matplotlib import pyplot as plt

data=pd.read_csv('C:\\Users\\soory\\Desktop\\machinelearning\\train.csv')
data.head()
data.shape
data.info()
# Putting missing columns in one variable
missing_data = data.isnull()
missing_data.head()
# Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 
data=data.fillna(0)
data['host_response_rate'].head()
# Descriptive analytics
data.describe()
# Grouping the data with respect property_type so we get to know that Timeshare is the expensive one of all.
data[['property_type', 'log_price']].groupby(['property_type'], as_index=False).mean().sort_values(by='log_price',ascending=False).head(n=10)
# Grouping the data with respect to city and by this we got to know that Sanfransico is the most expensive one.
data[['city', 'log_price']].groupby(['city'], as_index=False).mean().sort_values(by='log_price',ascending=False)
plt.scatter(data['review_scores_rating'],data['log_price'])
plt.ylabel('log_price')
plt.xlabel('review_scores_rating')
plt.show()
#printing the correlations
correlations = data.corr()['log_price'].drop('log_price')
print(correlations*100)
# Printing correlation with respect to two variables
print (np.corrcoef(data['bedrooms'], data['accommodates']))
# Printing correlation with respect to two variables
print (np.corrcoef(data['cleaning_fee'], data['accommodates']))
# Printing correlation with respect to two variables
print (np.corrcoef(data['bathrooms'], data['accommodates']))
# Printing correlation with respect to two variables
print (np.corrcoef(data['beds'], data['accommodates']))
# By this we can say accomadates are more related to bed and bedrooms while booking an airbnb because they have the highest poistive correlation among them.
# selecting the lat_center and long_center with respect to center of city
def lat_center(row):
    if (row['city']=='LA'):
        return 34.04
    
def long_center(row):
    if (row['city']=='LA'):
        return -118.26

data['lat_center']=data.apply(lambda row: lat_center(row), axis=1)
data['long_center']=data.apply(lambda row: long_center(row), axis=1)
# calculatiing distance with  the center value
data['distance to center']=np.sqrt((data['lat_center']-data['latitude'])**2+(data['long_center']-data['longitude'])**2)
pd.options.mode.chained_assignment = None 
la=data[data['city']=='LA']
la.head(5)

#coordinates of downtown
lat_la=34.04
long_la=-118.26
la['distance to center']=np.sqrt((lat_la-la['latitude'])**2+(long_la-la['longitude'])**2)
# We plot a graph distance to center with respect to log_price
plt.scatter(la['distance to center'],la['log_price'])
plt.ylabel('log_price')
plt.xlabel('distance to center')
plt.show()
print (np.corrcoef(la['distance to center'], la['log_price']))
# Correlation with respect to price and review_scores_rating
print (np.corrcoef(la['review_scores_rating'], la['log_price']))
# Used one hot encoding to this categorical columns 
categorical=['property_type','room_type','bed_type','cancellation_policy']
la_model=pd.get_dummies(la, columns=categorical)
la_model.head(5)
la_model.info()
# Select only numeric data and impute missing values as 0
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
x=la_model.select_dtypes(include=numerics).drop('log_price',axis=1).fillna(0).values
y=la_model['log_price'].values
print(x)
print(y)
# Splitting of data into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=40)
# Running models random forest for building this Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 80)
rf.fit(x_train,y_train)
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
training_reg = rf.predict(x_train)
val_preds_reg = rf.predict(x_test)
# Printing RMSE and r-square
print("\nTraining MSE:", round(mean_squared_error(y_train, training_reg),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_preds_reg),4))
print("\nTraining r2:", round(r2_score(y_train, training_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_reg),4))
# Running model a XGB Regressor
xgb_reg = xgb.XGBRegressor()

# Fit the model on training data
xgb_reg.fit(x_train, y_train)

# Predict
xgb_reg_1 = xgb_reg.predict(x_train)

# Validate
val_xgb_reg = xgb_reg.predict(x_test)

print("\nTraining MSE:", round(mean_squared_error(y_train, xgb_reg_1),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_xgb_reg),4))
print("\nTraining r2:", round(r2_score(y_train, xgb_reg_1),4))
print("Validation r2:", round(r2_score(y_test, val_xgb_reg),4))
# Same codes goes for other 2 cities and we computed various result on basis of this models
def lat_center(row):
    if (row['city']=='NYC'):
        return 40.74
    
def long_center(row):
    if (row['city']=='NYC'):
        return -73.98

data['lat_center']=data.apply(lambda row: lat_center(row), axis=1)
data['long_center']=data.apply(lambda row: long_center(row), axis=1)
data['distance to center']=np.sqrt((data['lat_center']-data['latitude'])**2+(data['long_center']-data['longitude'])**2)
pd.options.mode.chained_assignment = None 
nyc=data[data['city']=='NYC']

#coordinates of Midtown
lat_nyc=40.74
long_nyc=-73.98
nyc['distance to center']=np.sqrt((lat_nyc-nyc['latitude'])**2+(long_nyc-nyc['longitude'])**2)

plt.scatter(nyc['distance to center'],nyc['log_price'])
plt.ylabel('log_price')
plt.xlabel('distance to center')
plt.show()
print (np.corrcoef(nyc['distance to center'], nyc['log_price']))
plt.scatter(nyc['review_scores_rating'],nyc['log_price'])
plt.ylabel('log_price')
plt.xlabel('review_scores_rating')
plt.show()
print (np.corrcoef(nyc['review_scores_rating'], nyc['log_price']))
print (np.corrcoef(nyc['review_scores_rating'], nyc['log_price']))
print (np.corrcoef(nyc['beds'], nyc['accommodates']))
categorical=['property_type','room_type','bed_type','cancellation_policy']
nyc_model=pd.get_dummies(nyc, columns=categorical)
nyc_model.head(5)
nyc_model.info()
# Select only numeric data and impute missing values as 0
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
x=nyc_model.select_dtypes(include=numerics).drop('log_price',axis=1).fillna(0).values
y=nyc_model['log_price'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=40)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 80)
rf.fit(x_train,y_train)
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
training_reg = rf.predict(x_train)
val_preds_reg = rf.predict(x_test)
print("\nTraining MSE:", round(mean_squared_error(y_train, training_reg),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_preds_reg),4))
print("\nTraining r2:", round(r2_score(y_train, training_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_reg),4))
xgb_reg = xgb.XGBRegressor()

# Fit the model on training data
xgb_reg.fit(x_train, y_train)

# Predict
xgb_reg_1 = xgb_reg.predict(x_train)

# Validate
val_xgb_reg = xgb_reg.predict(x_test)

print("\nTraining MSE:", round(mean_squared_error(y_train, xgb_reg_1),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_xgb_reg),4))
print("\nTraining r2:", round(r2_score(y_train, xgb_reg_1),4))
print("Validation r2:", round(r2_score(y_test, val_xgb_reg),4))
def lat_center(row):
    if (row['city']=='DC'):
        return 38.89
    
def long_center(row):
    if (row['city']=='DC'):
        return -76.989

data['lat_center']=data.apply(lambda row: lat_center(row), axis=1)
data['long_center']=data.apply(lambda row: long_center(row), axis=1)
data['distance to center']=np.sqrt((data['lat_center']-data['latitude'])**2+(data['long_center']-data['longitude'])**2)
pd.options.mode.chained_assignment = None 
dc=data[data['city']=='DC']

#coordinates of Capitol hill
lat_dc=38.89
long_dc=-76.98
dc['distance to center']=np.sqrt((lat_dc-dc['latitude'])**2+(long_dc-dc['longitude'])**2)
plt.scatter(dc['distance to center'],dc['log_price'])
plt.ylabel('log_price')
plt.xlabel('distance to center')
plt.show()
print (np.corrcoef(dc['distance to center'], dc['log_price']))
plt.scatter(dc['review_scores_rating'],dc['log_price'])
plt.ylabel('log_price')
plt.xlabel('review_scores_rating')
plt.show()
print (np.corrcoef(dc['review_scores_rating'], dc['log_price']))
print (np.corrcoef(dc['beds'], dc['accommodates']))
categorical=['property_type','room_type','bed_type','cancellation_policy']
dc_model=pd.get_dummies(dc, columns=categorical)
dc_model.head(5)
dc_model.info()
# Select only numeric data and impute missing values as 0
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
x=dc_model.select_dtypes(include=numerics).drop('log_price',axis=1).fillna(0).values
y=dc_model['log_price'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=40)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 80)
rf.fit(x_train,y_train)
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
training_reg = rf.predict(x_train)
val_preds_reg = rf.predict(x_test)
print("\nTraining MSE:", round(mean_squared_error(y_train, training_reg),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_preds_reg),4))
print("\nTraining r2:", round(r2_score(y_train, training_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_reg),4))
xgb_reg = xgb.XGBRegressor()

# Fit the model on training data
xgb_reg.fit(x_train, y_train)

# Predict
xgb_reg_1 = xgb_reg.predict(x_train)

# Validate
val_xgb_reg = xgb_reg.predict(x_test)

print("\nTraining MSE:", round(mean_squared_error(y_train, xgb_reg_1),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_xgb_reg),4))
print("\nTraining r2:", round(r2_score(y_train, xgb_reg_1),4))
print("Validation r2:", round(r2_score(y_test, val_xgb_reg),4))
