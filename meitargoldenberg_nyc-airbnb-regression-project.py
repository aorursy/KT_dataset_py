# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# General tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 

# For transformations and predictions
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO

# For scoring
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse


# For validation
from sklearn.model_selection import train_test_split as split


%matplotlib inline
import sys

if 'google.colab' in sys.modules:
    from google.colab import files
    uploaded = files.upload()
nyc= pd.read_csv('nyc.csv')
nyc.head(3)
# Removing zero or negative prices
nyc = nyc[nyc.log_price > 0]
# Removing bad data
def drop_zeros(nyc):
    return nyc.loc[nyc.beds * nyc.accommodates * nyc.bathrooms * nyc.bedrooms != 0]
zeros_dropper = FunctionTransformer(drop_zeros, validate=False)

nyc = zeros_dropper.fit_transform(nyc) 
# Changing True\False field into 1\0 values
cleaning_fee_dict = {True:1,False:0}
host_identity_verified_dict={'t':1,'f':0}
instant_bookable_dict={'t':1,'f':0}

nyc.cleaning_fee.replace(cleaning_fee_dict,inplace=True)
nyc.host_identity_verified.replace(host_identity_verified_dict,inplace=True)
nyc.instant_bookable.replace(instant_bookable_dict,inplace=True)
#nyc.head()
# Creating Date\Year and delta dates between first review and last review

nyc['date_host'] = pd.to_datetime(nyc.host_since)
nyc['year_host'] = nyc['date_host'].dt.year

nyc['date_end'] = pd.to_datetime(nyc.last_review)
nyc['year_end'] = nyc['date_end'].dt.year

nyc['date_start'] = pd.to_datetime(nyc.first_review)
nyc['year_start'] = nyc['date_start'].dt.year

nyc['delta_dates'] = (nyc['date_end']-nyc['date_start'])

nyc['delta_dates']=nyc['delta_dates'].dt.days

nyc.drop(['date_host','date_end','date_start','date_host'],axis=1,inplace=True)

#nyc.head()

# Dropping irelevant fields
nyc.drop(['id','description','host_has_profile_pic','name','thumbnail_url','first_review','host_since','last_review','city','zipcode','host_response_rate'],axis=1,inplace=True)
# Converting Bed Type field to Real bed or Other
print(nyc.groupby('bed_type')['log_price'].count())

nyc.groupby('bed_type')['log_price'].mean().plot.bar()

def bed_group_func(row):
  if row.loc['bed_type'] == 'Real Bed':
    return 1
  else:
    return 0

nyc['real_bed'] = nyc.apply(bed_group_func, axis=1)

nyc.drop('bed_type',axis=1,inplace=True)
# Cancellation Policy field (Removing values with fiew data)
print(nyc.cancellation_policy.value_counts())

nyc = nyc[nyc.cancellation_policy != ('super_strict_30')]
nyc = nyc[nyc.cancellation_policy != ('super_strict_60')]
nyc['cancellation_policy'].unique()
# Changing Property type field to property_group (after creating high level values using dictionary)
print(nyc.property_type.value_counts())

property_type_dict1 = {'Apartment':['Condominium','Loft','Serviced apartment','Guest suite'],
         'House':['Vacation home','Villa','Townhouse','In-law','Casa particular'],
         'Hotel1':['Dorm','Hostel','Guesthouse'],
         'Hotel2':['Boutique hotel','Bed & Breakfast'],
         'Timeshare':['Timeshare'],
         'Other':['Island','Castle','Yurt','Hut','Chalet','Treehouse',
                  'Earth House','Tipi','Cave','Train','Parking Space','Lighthouse',
                 'Tent','Boat','Cabin','Camper/RV','Bungalow']
        }

property_type_dict2 = {i : k for k, v in property_type_dict1.items() for i in v}

nyc['property_group'] = nyc['property_type'].replace(property_type_dict2)

nyc.drop('property_type',axis=1,inplace=True)

print('---------------------------------------')
print(nyc['property_group'].unique())
# Adding price per room field (For neighnourhood price level)
nyc['price_per_room'] = nyc['log_price'] / nyc['bedrooms']

nyc.neighbourhood.value_counts().head(30).plot.bar(color=(.0, 0.4, 0.9, 1))

neighbourhood_avg_price = nyc[['neighbourhood','price_per_room']].groupby('neighbourhood')['price_per_room'].mean().sort_values()
neighbourhood_avg_price.replace(np.inf, np.nan,inplace=True)
neighbourhood_avg_price.fillna(neighbourhood_avg_price.mean(),inplace=True)

print(neighbourhood_avg_price.sort_values(ascending=False))
print('---------------------------------------')
print(neighbourhood_avg_price.describe())
neighbourhood_class_df = neighbourhood_avg_price.to_frame()
type(neighbourhood_class_df)
# Converting neighbourhoods to Levels
def neigbourhood_class(row):
  if row['price_per_room'] >=0 and row['price_per_room'] <= 3.683610:
    return 1
  elif row['price_per_room'] > 3.6836100 and row['price_per_room'] <= 3.868928:
    return 2
  elif row['price_per_room'] >3.868928 and row['price_per_room'] <= 4.194452: 
    return 3
  else:
    return 4
  
neighbourhood_class_df['neigbourhood_level'] = neighbourhood_class_df.apply(neigbourhood_class,axis=1)
neighbourhood_class_df.sort_values(by='neigbourhood_level',ascending=False)
neighbourhood_class_df.drop('price_per_room',axis=1,inplace=True)
# Joining between the Main Data Frame and the  neighbourhood_class data frame to get neighbourhood class
nyc = nyc.join(neighbourhood_class_df,on='neighbourhood')
# Using Longtitude and Latitude fields to create 0-1 scale (North to South / West to East)

plt.scatter(nyc.longitude,nyc.latitude,c = nyc.log_price)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title('log_price on Map', x=0.5, y=1.05, ha='center', fontsize='xx-large')
nyc.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(15,10),
    c="log_price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
# The Formulas are:
# 1. (latitude - min_latitude) / (max_latitude - min_latitude)
# 2. (longitude - min_longitude) / (max_longitude - min_longitude)

nyc['latitude_north'] = (nyc.latitude - nyc.latitude.min()) / (nyc.latitude.max() - nyc.latitude.min())
nyc['longitude_east'] = (nyc.longitude - nyc.longitude.min()) / (nyc.longitude.max() - nyc.longitude.min())
# Distance from time Square
nyc['distance_to_t_squre'] = np.sqrt((40.758896-nyc['latitude'])**2+(-73.985130-nyc['longitude'])**2)

nyc_num_2 = nyc[['distance_to_t_squre','log_price']].select_dtypes(include=np.number)
sns.pairplot(nyc_num_2, height=3)
# Distance from central train
nyc['distance_to_c_train'] = np.sqrt((40.752655-nyc['latitude'])**2+(-73.977295-nyc['longitude'])**2)

nyc_num_2 = nyc[['distance_to_c_train','log_price']].select_dtypes(include=np.number)
sns.pairplot(nyc_num_2, height=3)
# Distance from Wall Street
nyc['distance_to_w_street']=np.sqrt((40.706005-nyc['latitude'])**2+(-74.008827-nyc['longitude'])**2)

nyc_num_2 = nyc[['distance_to_w_street','log_price']].select_dtypes(include=np.number)
sns.pairplot(nyc_num_2, height=3)
nyc.drop(['latitude','longitude'],axis=1,inplace=True)
# Checking null values

nyc.info()

nyc.dropna(subset=['bathrooms','host_identity_verified','bedrooms','beds'],inplace=True)
print('--------------------------------------------------')
nyc.info()
# Changing amenities field to booleans seperated fields

l=list(nyc['amenities'])
l=[[word.strip('[" ]') for word in row[1:-1].split(',')] for row in list(nyc['amenities'])]
cols = set(word for row in l  for word in row)
amenities_df=pd.DataFrame(columns=cols)
print(cols)
amenities_df = pd.DataFrame(columns=cols)
for row_idx in range(len(l)):
    for col in cols:
        amenities_df.loc[row_idx,col]=int(col in l[row_idx])
amenities_df.head()
# Building a new field aggregating fields from amenities_df
# The new fields will be: kitchen, accesibility, Electricity_and_Technology, facilities, kids_friendly, security, services

amenities_group_df = pd.DataFrame()
#--------------------------------------
amenities_group_df['kitchen'] = amenities_df['Kitchen']+amenities_df['Breakfast']+amenities_df['Cooking basics']+amenities_df['Cooking basics']+amenities_df['BBQ grill']+amenities_df['Oven']+amenities_df['Coffee maker']+amenities_df['Microwave']+amenities_df['Refrigerator']+amenities_df['Dishwasher']
amenities_group_df['accesibility'] = amenities_df['Free parking on premises']+amenities_df['Wide clearance to bed']+amenities_df['smooth pathway to front door']+amenities_df['Ground floor access']+amenities_df['Lake access']+amenities_df['Wheelchair accessible']+amenities_df['Wide clearance to shower & toilet']+amenities_df['Wide hallway clearance']+amenities_df['Wide doorway']+amenities_df['Accessible-height toilet']+amenities_df['Step-free access']+amenities_df['Well-lit path to entrance']+amenities_df['Waterfront']+amenities_df['Free parking on street']+amenities_df['Disabled parking spot']+amenities_df['Accessible-height bed']+amenities_df['Private entrance']+amenities_df['Elevator']
amenities_group_df['Elect_Tech'] = amenities_df['Wide entryway']+amenities_df['Air conditioning']+amenities_df['Ethernet connection']+amenities_df['Cable TV']+amenities_df['Internet']+amenities_df['EV charger']+amenities_df['Baby monitor']+amenities_df['TV']+amenities_df['Wireless Internet']+amenities_df['Pocket wifi']+amenities_df['Washer']+amenities_df['Dryer']+amenities_df['Keypad']+amenities_df['Game console']+amenities_df['Washer / Dryer']+amenities_df['Hair dryer']
amenities_group_df['facilities'] = amenities_df['Private living room']+amenities_df['Air purifier']+amenities_df['Handheld shower head']+amenities_df['Hot water kettle']+amenities_df['Extra pillows and blankets']+amenities_df['Hot tub']+amenities_df['Pets live on this property']+amenities_df['Heating']+amenities_df['Dishes and silverware']+amenities_df['Patio or balcony']+amenities_df['Bed linens']+amenities_df['First aid kit']+amenities_df['Crib']+amenities_df['Flat']+amenities_df['Laptop friendly workspace']+amenities_df['Buzzer/wireless intercom']+amenities_df['Firm mattress']+amenities_df['Iron']+amenities_df['Changing table']+amenities_df['Hangers']+amenities_df['Roll-in shower with chair']+amenities_df['Gym']+amenities_df['Outlet covers']+amenities_df['Essentials']+amenities_df['Private bathroom']+amenities_df['Baby bath']+amenities_df['Bathtub']+amenities_df['Shampoo']+amenities_df['Beachfront']+amenities_df['Single level home']+amenities_df['Hot water']+amenities_df['High chair']+amenities_df['Bathtub with shower chair']+amenities_df['Pool']+amenities_df['Fixed grab bars for shower & toilet']+amenities_df['Room-darkening shades']+amenities_df['Beach essentials']+amenities_df['Garden or backyard']
amenities_group_df['kids_friendly'] = amenities_df['Babysitter recommendations']+amenities_df['Family/kid friendly']+amenities_df['Children’s books and toys']+amenities_df['Children’s dinnerware']
amenities_group_df['security'] = amenities_df['Window guards']+amenities_df['Stair gates']+amenities_df['Fireplace guards']+amenities_df['Doorman']+amenities_df['Carbon monoxide detector']+amenities_df['Smoke detector']+amenities_df['Table corner guards']+amenities_df['Fire extinguisher']+amenities_df['Lock on bedroom door']+amenities_df['Smart lock']+amenities_df['Lockbox']
amenities_group_df['services'] = amenities_df['Ski in/Ski out']+amenities_df['Cleaning before checkout']+amenities_df['Long term stays allowed']+amenities_df['Other pet(s)']+amenities_df['Cat(s)']+amenities_df['Self Check-In']+amenities_df['24-hour check-in']+amenities_df['Host greets you']+amenities_df['Luggage dropoff allowed']+amenities_df['Pack ’n Play/travel crib']+amenities_df['Pets allowed']+amenities_df['Suitable for events']+amenities_df['Safety card']+amenities_df['Indoor fireplace']+amenities_df['Dog(s)']+amenities_df['Smoking allowed']
amenities_group_df.head()
amenities_group_df.describe()
# This field will use to join between nyc data frame and amenities_group data frame
nyc['join_key'] = range(0,len(nyc))
nyc.index = nyc['join_key']
# Joining between the main data set and the data set based on amenities field
nyc_j = nyc.join(amenities_group_df)
nyc_j.head(3)
# Keeping Data Frame before Dummies
nyc_before_dummies_df = nyc_j
nyc_before_dummies_2_df = nyc_j
# Creating Dummy Variables to string fields

room_dummies = pd.get_dummies(nyc_j.room_type,prefix='room').iloc[:,1:]
nyc_j = pd.concat([nyc_j,room_dummies],axis=1)

cancellation_dummies = pd.get_dummies(nyc_j.cancellation_policy,prefix='cancellation').iloc[:,1:]
nyc_j = pd.concat([nyc_j,cancellation_dummies],axis=1)

property_dummies = pd.get_dummies(nyc_j.property_group,prefix='property').iloc[:,1:]
nyc_j = pd.concat([nyc_j,property_dummies],axis=1)

bedrooms_dummies = pd.get_dummies(nyc_j.bedrooms,prefix='bedrooms').iloc[:,1:]
nyc_j = pd.concat([nyc_j,bedrooms_dummies],axis=1)
nyc_j.head()
# Dropping original fields after dummies
nyc_j.drop(['bedrooms','amenities','room_type','cancellation_policy','neighbourhood','property_group','join_key','price_per_room'],axis=1,inplace=True)
# New Data Frame without losing the first dummy variable (For decision tree)
room_dummies = pd.get_dummies(nyc_before_dummies_df.room_type,prefix='room').iloc[:,:]
nyc_before_dummies_df = pd.concat([nyc_before_dummies_df,room_dummies],axis=1)

cancellation_dummies = pd.get_dummies(nyc_before_dummies_df.cancellation_policy,prefix='cancellation').iloc[:,:]
nyc_before_dummies_df = pd.concat([nyc_before_dummies_df,cancellation_dummies],axis=1)

property_dummies = pd.get_dummies(nyc_before_dummies_df.property_group,prefix='property').iloc[:,:]
nyc_before_dummies_df = pd.concat([nyc_before_dummies_df,property_dummies],axis=1)

nyc_before_dummies_df.drop(['amenities','room_type','cancellation_policy','neighbourhood','property_group','join_key','price_per_room'],axis=1,inplace=True)
nyc_j.head(3)
# Histogram of nyc data frame
nyc_j.hist(figsize=(10, 10),color=("c"))
# Pair plot of nyc data frame
nyc_num = nyc_j.select_dtypes(include=np.number)
sns.pairplot(nyc_num, height=3)
# Log price Correlations matrix
corr_matrix = nyc_j.corr()
corr_matrix["log_price"].sort_values(ascending=False)
# Using correlation graph
fig, ax = plt.subplots(figsize =(20, 20)) 
corr = nyc_j.corr()
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(nyc_j.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(nyc_j.columns)
ax.set_yticklabels(nyc_j.columns)
plt.show()
# Log price by accommodates pair plot
nyc_num_accommodates = nyc[['log_price','accommodates']]#.select_dtypes(include=np.number)
sns.pairplot(nyc_num_accommodates, height=3)
# Log price by beds pair plot
nyc_num_beds = nyc[['log_price','beds']]#.select_dtypes(include=np.number)
sns.pairplot(nyc_num_beds, height=3)
nyc_j.dropna(inplace=True)
nyc_before_dummies_df.dropna(inplace=True)
# Splitting the data to X and y
X = nyc_j.drop('log_price', axis=1)
y = nyc_j.log_price

# Splitting data to train and test
X_train, X_test, y_train, y_test = split(X,y,train_size=0.7,random_state=12345)
# Using Scikitlearn Linear regression & fit func
linear_model_1 = LinearRegression()
linear_model_1.fit(X_train, y_train)
# list of linear regression coefficient
list(zip(X_train.columns, linear_model_1.coef_))
linear_model_1.coef_[0]
# y pred based on training set
y_train_pred = linear_model_1.predict(X_train)
# Comparing between y_train and y_train_pred
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
# Checking the train set RMSE
mse(y_train, y_train_pred)**0.5
# Checking results on the test set
y_test_pred = linear_model_1.predict(X_test)
# Comparing between y_test and y_test_pred
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
# Checking test set RMSE
mse(y_test, y_test_pred)**0.5
# Removing anomalous dots
cols=['accommodates',	'bathrooms'	,'cleaning_fee',	'number_of_reviews',	'review_scores_rating',	'beds'	,'year_host']

for col in cols:
    if nyc_j[col].dtype == 'float64':
        std = nyc_j[col].std()
        ave = nyc_j[col].mean()
        nyc_j = nyc_j.loc[nyc_j[col].between(ave-3.6*std, ave+3.6*std)]
        print(f'processing {col:10} --> {nyc_j.shape[0]:5} nyc_j remain')
# Splitting the data
X = nyc_j.drop('log_price', axis=1)
y = nyc_j.log_price

X_train, X_test, y_train, y_test = split(X,y,train_size=0.7,random_state=12345)
linear_model_2 = LinearRegression().fit(X_train, y_train)
list(zip(X_train.columns, linear_model_2.coef_))
# y train prediction
y_train_pred = linear_model_2.predict(X_train)
# y train vs y train pred
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
# Training RMSE
mse(y_train, y_train_pred)**0.5
# y test prediction
y_test_pred = linear_model_2.predict(X_test)
# y test vs y test pred
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
# Testing RMSE
mse(y_test, y_test_pred)**0.5
# stepwise
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X, y)

print('resulting features:')
print(result)
X = X[['kids_friendly', 'room_Private room', 'distance_to_t_squre', 'room_Shared room', 'accommodates', 'neigbourhood_level', 'Elect_Tech', 'distance_to_w_street', 'bathrooms', 'distance_to_c_train', 'review_scores_rating', 'bedrooms_3.0', 'bedrooms_2.0', 'bedrooms_4.0', 'longitude_east', 'year_end', 'property_Hotel2', 'property_House', 'number_of_reviews', 'year_start', 'property_Timeshare', 'latitude_north', 'bedrooms_5.0', 'beds', 'accesibility', 'bedrooms_7.0', 'bedrooms_6.0', 'property_Other', 'year_host', 'real_bed']]
y = nyc_j.log_price
X_train, X_test, y_train, y_test = split(X,y,train_size=0.7,random_state=12345)
linear_model_3 = LinearRegression().fit(X_train, y_train)
list(zip(X_train.columns, linear_model_2.coef_))
# y train prediction
y_train_pred = linear_model_3.predict(X_train)
# y train vs y train pred
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
# Training RMSE
mse(y_train, y_train_pred)**0.5
# y test prediction
y_test_pred = linear_model_3.predict(X_test)
# y test vs y test pred
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
# Testing RMSE
mse(y_test, y_test_pred)**0.5
# Splitting data to X and y
X = nyc_before_dummies_df.drop('log_price', axis=1)
y = nyc_before_dummies_df.log_price
# Splitting data to train and test
X_train, X_test, y_train, y_test = split(X,y,train_size=0.7,random_state=12345)
# The descision tree model
dt_model_1 = DecisionTreeRegressor(max_depth=5)
dt_model_1.fit(X_train, y_train)
# Visualization Tree
def visualize_tree(model, md=3):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=1800) 
visualize_tree(dt_model_1, 4)
# Train Prediction
y_train_pred = dt_model_1.predict(X_train)
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
#Training RMSE
mse(y_train, y_train_pred)**0.5
# Test Prediction
y_test_pred = dt_model_1.predict(X_test)
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
# Test RMSE
mse(y_test, y_test_pred)**0.5
sub_model = nyc_before_dummies_df.copy()
# Removing anomalous dots
cols=['accommodates',	'bathrooms'	,'cleaning_fee',	'number_of_reviews',	'review_scores_rating',	'beds'	,'year_host']

for col in cols:
    if sub_model[col].dtype == 'float64':
        std = nyc_j[col].std()
        ave = nyc_j[col].mean()
        sub_model = sub_model.loc[sub_model[col].between(ave-3.6*std, ave+3.6*std)]
        print(f'processing {col:10} --> {sub_model.shape[0]:5} sub_model remain')
sub_X = sub_model.drop('log_price', axis=1)
sub_y = sub_model.log_price

sub_X_train, sub_X_test, sub_y_train, sub_y_test = split(sub_X, sub_y, train_size=0.7,random_state=12345)
# max_depth check
complexity = range (2, 50 , 1)
scores = pd.DataFrame(index=complexity, columns=['train', 'test'])

for leafs in complexity:
    model = DecisionTreeRegressor(max_leaf_nodes=leafs).fit(sub_X_train, sub_y_train)
    
    sub_y_train_pred = model.predict(sub_X_train)
    scores.loc[leafs, 'train'] = mse(sub_y_train_pred, sub_y_train) ** 0.5
    
    sub_y_test_pred = model.predict(sub_X_test)
    scores.loc[leafs, 'test'] = mse(sub_y_test_pred, sub_y_test) ** 0.5

scores.plot()
# min_samples_leaf check
complexity = range (1, 50 , 5)
scores = pd.DataFrame(index=complexity, columns=['train', 'test'])

for comp in complexity:
    model = DecisionTreeRegressor(min_samples_leaf=comp).fit(sub_X_train, sub_y_train)
    
    sub_y_train_pred = model.predict(sub_X_train)
    scores.loc[comp, 'train'] = mse(sub_y_train_pred, sub_y_train) ** 0.5
    
    sub_y_test_pred = model.predict(sub_X_test)
    scores.loc[comp, 'test'] = mse(sub_y_test_pred, sub_y_test) ** 0.5

scores.plot()
# max_leaf_nodes check
complexity = range (2, 70 , 1)
scores = pd.DataFrame(index=complexity, columns=['train', 'test'])

for comp in complexity:
    model = DecisionTreeRegressor(max_leaf_nodes=comp).fit(sub_X_train, sub_y_train)
    
    sub_y_train_pred = model.predict(sub_X_train)
    scores.loc[comp, 'train'] = mse(sub_y_train_pred, sub_y_train) ** 0.5
    
    sub_y_test_pred = model.predict(sub_X_test)
    scores.loc[comp, 'test'] = mse(sub_y_test_pred, sub_y_test) ** 0.5

scores.plot()
# Improving Hyper parameters
l_max_depth = [2,3,4,5,6,7,8,9,10]
l_min_samples_leaf = [20,25,30,35,40,45,50]
l_min_impurity_decrease = [0.001,0.002,0.003,0.004,0.005]
max_leaf_nodes= [40,45,50,55,60]

from itertools import product

scores = []
for i, (md, msl, mid, mln) in enumerate(product(l_max_depth, l_min_samples_leaf, l_min_impurity_decrease,max_leaf_nodes)):
    sub_model = DecisionTreeRegressor(max_depth=md, min_samples_leaf=msl, min_impurity_decrease=mid, max_leaf_nodes=mln)
    sub_model.fit(sub_X_train, sub_y_train)
    y_train_pred = sub_model.predict(sub_X_train)
    train_score = mse(sub_y_train, sub_y_train_pred)**0.5
    y_test_pred = sub_model.predict(sub_X_test)
    test_score = mse(sub_y_test, sub_y_test_pred)**0.5
    scores.append((md, msl, mid, mln,train_score, test_score))
sorted(scores, key=lambda x: x[3], reverse=False) 
dt_model_2 = DecisionTreeRegressor(max_leaf_nodes=12, max_depth=8, min_samples_leaf=150, min_impurity_decrease=0.002).fit(sub_X_train, sub_y_train)
def visualize_tree(model, md=5):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=1200) 

visualize_tree(dt_model_2, 7)
sub_y_train_pred = dt_model_2.predict(sub_X_train)
ax = sns.scatterplot(x=sub_y_train, y=sub_y_train_pred)
ax.plot(y_train, y_train, 'r')
RMSE = mse(sub_y_train, sub_y_train_pred)**0.5
RMSE
sub_y_test_pred = dt_model_2.predict(sub_X_test)
ax = sns.scatterplot(x=sub_y_test, y=sub_y_test_pred)
ax.plot(y_test, y_test, 'r')
RMSE = mse(sub_y_test, sub_y_test_pred)**0.5
RMSE
# Splitting data to X and y
X = nyc_j.drop('log_price', axis=1)
y = nyc_j.log_price
# Splitting data to train and test
X_train, X_test, y_train, y_test = split(X, y, train_size=0.7, random_state=314159)
# Using feature scaling
my_scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(my_scaler.transform(X_train), columns=X_train.columns)
# Building the model
knn_model_1 = KNeighborsRegressor().fit(X_train_scaled, y_train)
# Train Prediction
y_train_pred = knn_model_1.predict(X_train_scaled)
# y_train vs y_train_pred
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
# train RMSE
RMSE = mse(y_train, y_train_pred)**0.5
RMSE
# Scaling X test
X_test_scaled = my_scaler.transform(X_test)
# y Test prediction
y_test_pred = knn_model_1.predict(X_test_scaled)
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
RMSE = mse(y_test, y_test_pred)**0.5
RMSE
# Removing anomalous dots
cols=['accommodates',	'bathrooms'	,'cleaning_fee',	'number_of_reviews',	'review_scores_rating',	'beds'	,'year_host']

for col in cols:
    if nyc_j[col].dtype == 'float64':
        std = nyc_j[col].std()
        ave = nyc_j[col].mean()
        nyc_j = nyc_j.loc[nyc_j[col].between(ave-3.6*std, ave+3.6*std)]
        print(f'processing {col:10} --> {nyc_j.shape[0]:5} nyc_j remain')
# Choosing the best KNN
RMSE = []
for num in range(1, 12):
    knn_model_1 = KNeighborsRegressor(n_neighbors=num).fit(X_train, y_train)
    y_test_pred = knn_model_1.predict(X_test)
    RMSE1 = mse(y_test, y_test_pred)**0.5
    RMSE.append(RMSE1)

    print(num,'-',RMSE1)
# Splitting data to X and y
X = nyc_j.drop('log_price', axis=1)
y = nyc_j.log_price
# Splitting data to train and test
X_train, X_test, y_train, y_test = split(X, y, train_size=0.7, random_state=314159)
# Using feature scaling
my_scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(my_scaler.transform(X_train), columns=X_train.columns)
# Building the model
knn_model_2 = KNeighborsRegressor(n_neighbors=10).fit(X_train_scaled, y_train)
# Train Prediction
y_train_pred = knn_model_2.predict(X_train_scaled)
# y_train vs y_train_pred
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
# train RMSE
RMSE = mse(y_train, y_train_pred)**0.5
RMSE
# Scaling X test
X_test_scaled = my_scaler.transform(X_test)
# y Test prediction
y_test_pred = knn_model_2.predict(X_test_scaled)
#y_test vs y_test_pred
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
# Test RMSE
RMSE = mse(y_test, y_test_pred)**0.5
RMSE
# Splitting data to X and y
X = nyc_j.drop('log_price', axis=1)
y = nyc_j.log_price
# Splitting data to train and test
X_train, X_test, y_train, y_test = split(X, y, train_size=0.7, random_state=314159)
# Using feature scaling
my_scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(my_scaler.transform(X_train), columns=X_train.columns)
# Building the model
knn_model_3 = KNeighborsRegressor(n_neighbors=10,weights='distance').fit(X_train_scaled, y_train)
# Train Prediction
y_train_pred = knn_model_3.predict(X_train_scaled)
# Scaling X test
X_test_scaled = my_scaler.transform(X_test)
# y Test prediction
y_test_pred = knn_model_3.predict(X_test_scaled)
#y_test vs y_test_pred
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
# Test RMSE
RMSE = mse(y_test, y_test_pred)**0.5
RMSE