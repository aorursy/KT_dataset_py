import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
bangalore = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Bangalore.csv')
bangalore.head()
bangalore.info()
len(bangalore)
bangalore['PricePSF'] = bangalore['Price'] / bangalore['Area']
def dist_plot (feature, color, position=121):
    plt.figure(figsize=(15,4))
    plt.subplot(position)
    sns.boxplot(x=feature, data=bangalore, color=color)
    plt.subplot(position+1)
    sns.distplot(bangalore[feature], color=color)
dist_plot('Price', 'red')
dist_plot('Area', 'grey')
dist_plot('PricePSF', 'teal')
# define outliers
price_psf_outliers = np.percentile(bangalore.PricePSF, [5,90])
price_outliers = np.percentile(bangalore.Price, [5,90])
area_outliers = np.percentile(bangalore.Area, [5,90])
# create filters based on outliers
price_psf_filter = (bangalore.PricePSF > price_psf_outliers[0]) & (bangalore.PricePSF < price_psf_outliers[1])
price_filter = (bangalore.Price > price_outliers[0]) & (bangalore.Price < price_outliers[1])
area_filter = (bangalore.Area > area_outliers[0]) & (bangalore.Area < area_outliers[1])
# apply filters
bangalore = bangalore[(price_psf_filter) & (price_filter) & (area_filter)]
dist_plot('Price', 'red')
dist_plot('Area', 'grey')
dist_plot('PricePSF', 'teal')
bangalore.columns
bangalore.rename(columns={'No. of Bedrooms':'Bedrooms', "Children'splayarea":'PlayArea'}, inplace=True)
# assign weights to features
feature_dict = {'MaintenanceStaff':2, 'Gymnasium':4, 'SwimmingPool':4,'LandscapedGardens':3, 'JoggingTrack':3, 'RainWaterHarvesting':2,'IndoorGames':3, 'ShoppingMall':2, 'Intercom':2, 'SportsFacility':3, 'ATM':2, 'ClubHouse':2, 'School':2, '24X7Security':1, 'PowerBackup':4, 'CarParking':3, 'StaffQuarter':0, 'Cafeteria':0, 'MultipurposeRoom':2, 'Hospital':3, 'WashingMachine':0, 'Gasconnection':2, 'AC':0, 'Wifi':0, 'PlayArea':3, 'LiftAvailable':0, 'BED':0, 'VaastuCompliant':0, 'Microwave':0, 'GolfCourse':0, 'TV':0, 'DiningTable':0, 'Sofa':0, 'Wardrobe':0, 'Refrigerator':0}
# convert to Dataframe
features = pd.DataFrame(feature_dict.items(), columns=['Features', 'Weight'])
features.head()
# The features matrix has 35 features
features.shape
features_matrix = bangalore[['MaintenanceStaff', 'Gymnasium', 'SwimmingPool',
       'LandscapedGardens', 'JoggingTrack', 'RainWaterHarvesting',
       'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility', 'ATM',
       'ClubHouse', 'School', '24X7Security', 'PowerBackup', 'CarParking',
       'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital',
       'WashingMachine', 'Gasconnection', 'AC', 'Wifi', 'PlayArea',
       'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave',
       'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe',
       'Refrigerator']]
features_matrix.tail()
features_matrix.replace(9, 0, inplace=True)
features_matrix.shape
# apply dot product to compute feature score for every row
feature_scores_df = np.dot(features_matrix, features['Weight'])
# join feature scores column with bangalore dataframe
bangalore['FeatureScore'] = feature_scores_df
bangalore.head(3)
corr_df = bangalore[['Area', 'Resale', 'Bedrooms', 'FeatureScore', 'Price']]
sns.heatmap(corr_df.corr(method='pearson'), cmap='RdYlGn_r', linewidths=2)
# Create a pivot table of Locations with PricePSF as the value
location_pivot = pd.pivot_table(data=bangalore, index='Location', aggfunc='mean', values='PricePSF')
location_pivot
location_pivot['LocationPremium'] = location_pivot['PricePSF'] / location_pivot['PricePSF'].min()
location_pivot.sort_values('LocationPremium', ascending=False)
bangalore = pd.merge(bangalore, location_pivot['LocationPremium'], on='Location')
bangalore['LogPremium'] = np.log(bangalore['LocationPremium'])
# check if all required columns are present
bangalore.head(3)
x = bangalore[['Area', 'FeatureScore', 'Resale', 'LogPremium', 'Bedrooms']]
y = bangalore['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)
model = sm.OLS(y_train, x_train).fit()
model.summary()
predictions = model.predict(x_test)
plt.scatter(x=y_test, y=predictions)
plt.xlabel('Actual Test Prices')
plt.ylabel('Predicted Test Prices')
residuals = y_test - predictions
sns.distplot(residuals)
plt.xlabel('Residuals')
sns.regplot(x=predictions, y=residuals)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
mean_absolute_error(y_test,predictions)
predictions.mean()
x2 = bangalore[['Area', 'FeatureScore', 'Resale', 'LogPremium']]
y2 = bangalore['Price']
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2, random_state=19)
model2 = sm.OLS(y_train2, x_train2).fit()
model2.summary()
predictions2 = model2.predict(x_test2)
mean_absolute_error(y_test2,predictions2)
predictions2.mean()
