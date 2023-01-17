import pandas as pd

import numpy as np

import seaborn as sns



raw_data = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

raw_data.head()
raw_data.columns
raw_data.info()
# I think We don't need ID. So we should remove it from our data

data = raw_data.drop('id', axis = 1)



# I try to drop zipcode because We have had Latitude and Longitude data

data = data.drop('zipcode', axis = 1)



# and actually I don't understand about sqft_living15 and sqft_lot15, so I decided to remove it

data = data.drop(['sqft_living15','sqft_lot15'], axis = 1)
data = data[data['sqft_lot'] >= data['sqft_living']]

data = data[data['sqft_lot'] >= data['sqft_above']]

data = data[data['sqft_lot'] >= data['sqft_basement']]


data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)



data['date'] = pd.to_datetime(data['date'], format = '%Y/%m/%d')



# this code change value format into datetime format
data['day'] = data['date'].dt.day

data['month'] = data['date'].dt.month



#I think We don't need year data, because it contains two values only (2014 and 2015)
# we drop date columns, because We have split the data into day, month, and year

data = data.drop('date', axis=1)



# reorder the columns

data = data[['price', 'day', 'month','bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',

       'lat', 'long']]
#make a checkpoint

data_price  = data.copy()
sns.distplot(data_price['price'])
sns.boxplot(data_price['price'])
# normal distribution check

price_kurtosis = data_price['price'].kurt()

price_skewness = data_price['price'].skew()



print ('Kurtosis and Skewness of Price \n')

print ('Kurtosis: ' + str(price_kurtosis))

print ('Skewness: ' + str(price_skewness))
#Based on our boxplot, let's try to remove the outlier with threshold 6000000

data_price = data_price[data_price['price'] <= 6000000]



#let's check kurtosis and skewness again

price_kurtosis = data_price['price'].kurt()

price_skewness = data_price['price'].skew()

print ('Kurtosis and Skewness of Price \n')

print ('Kurtosis: ' + str(price_kurtosis))

print ('Skewness: ' + str(price_skewness))
data_price  = data.copy()
data_price['log_price'] = np.log(data_price['price'])
sns.distplot(data_price['log_price'])
sns.boxplot(data_price['log_price'])
#recheck Our kurtosis and skewness score again

price_kurtosis = data_price['log_price'].kurt()

price_skewness = data_price['log_price'].skew()



print ('Final Our Kurtosis and Skewness \n')

print ('Kurtosis: ' + str(price_kurtosis))

print ('Skewness: ' + str(price_skewness))
# before move on, I think We should drop price columns because We have log_price now

data_price = data_price.drop('price', axis = 1)
#make a checkpoint

data_bedroom = data_price.copy()
data_bedroom['bedrooms'].unique()
sns.distplot(data_bedroom['bedrooms'])
sns.boxplot(data_bedroom['bedrooms'])
# normal distribution check

bedroom_kurtosis = data_bedroom['bedrooms'].kurt()

bedroom_skewness = data_bedroom['bedrooms'].skew()



print ('Kurtosis and Skewness of Bedroom \n')

print ('Kurtosis: ' + str(bedroom_kurtosis))

print ('Skewness: ' + str(bedroom_skewness))
#based on Our boxplot let's try to remove the outlier with threshold is 15

data_bedroom = data_bedroom[data_bedroom['bedrooms'] <= 15]



#let's check kurtosis and skewness again

bedroom_kurtosis = data_bedroom['bedrooms'].kurt()

bedroom_skewness = data_bedroom['bedrooms'].skew()

print ('Final Our Kurtosis and Skewness \n')

print ('Kurtosis: ' + str(bedroom_kurtosis))

print ('Skewness: ' + str(bedroom_skewness))
#checkpoint

data_bathroom = data_bedroom.copy()
sns.distplot(data_bathroom['bathrooms'])
sns.boxplot(data_bathroom['bathrooms'])
# normal distribution check

bathroom_kurtosis = data_bathroom['bathrooms'].kurt()

bathroom_skewness = data_bathroom['bathrooms'].skew()



print ('Kurtosis and Skewness of Bathrooms \n')

print ('Kurtosis: ' + str(bathroom_kurtosis))

print ('Skewness: ' + str(bathroom_skewness))
#checkpoint

data_sqft_living = data_bathroom.copy()
sns.distplot(data_sqft_living['sqft_living'])
sns.boxplot(data_sqft_living['sqft_living'])
# normal distribution check



sqft_living_kurtosis = data_sqft_living['sqft_living'].kurt()

sqft_living_skewness = data_sqft_living['sqft_living'].skew()



print ('Kurtosis and Skewness of Sqft Living \n')

print ('Kurtosis: ' + str(sqft_living_kurtosis))

print ('Skewness: ' + str(sqft_living_skewness))
# checkpoint

data_sqft_lot = data_sqft_living.copy()
sns.distplot(data_sqft_lot['sqft_lot'])
sns.boxplot(data_sqft_lot['sqft_lot'])
# normal distribution check

sqft_lot_kurtosis = data_sqft_lot['sqft_lot'].kurt()

sqft_lot_skewness = data_sqft_lot['sqft_lot'].skew()



print ('Kurtosis and Skewness of Sqft Lot \n')

print ('Kurtosis: ' + str(sqft_lot_kurtosis))

print ('Skewness: ' + str(sqft_lot_skewness))
#Based on our boxplot, let's try to remove the outlier with threshold 1250000

data_sqft_lot = data_sqft_lot[data_sqft_lot['sqft_lot'] <= 1250000]



#let's check kurtosis and skewness again

sqft_lot_kurtosis = data_sqft_lot['sqft_lot'].kurt()

sqft_lot_skewness = data_sqft_lot['sqft_lot'].skew()



print ('Kurtosis and Skewness of Sqft Lot \n')

print ('Kurtosis: ' + str(sqft_lot_kurtosis))

print ('Skewness: ' + str(sqft_lot_skewness))
data_sqft_lot = data_sqft_living.copy()
data_sqft_lot['log_sqft_lot'] = np.log(data_sqft_lot['sqft_lot'])
sns.distplot(data_sqft_lot['log_sqft_lot'])
sns.boxplot(data_sqft_lot['log_sqft_lot'])
#let's check kurtosis and skewness again

sqft_lot_kurtosis = data_sqft_lot['log_sqft_lot'].kurt()

sqft_lot_skewness = data_sqft_lot['log_sqft_lot'].skew()



print ('Final Our Kurtosis and Skewness \n')

print ('Kurtosis: ' + str(sqft_lot_kurtosis))

print ('Skewness: ' + str(sqft_lot_skewness))
# before move on, I think We should drop sqft_lot columns because We have log_sqft_lot now

data_sqft_lot= data_sqft_lot.drop('sqft_lot', axis = 1)
data_floor = data_sqft_lot.copy()
sns.distplot(data_floor['floors'])
sns.boxplot(data_floor['floors'])
# normal distribution check

floor_kurtosis = data_floor['floors'].kurt()

floor_skewness = data_floor['floors'].skew()



print ('Kurtosis and Skewness of Floors \n')

print ('Kurtosis: ' + str(floor_kurtosis))

print ('Skewness: ' + str(floor_skewness))
data['waterfront'].unique()
# checkpoint

data_view = data_floor.copy()
data_view['view'].unique()
# one-hot encoding

data_view = pd.get_dummies(data_view, columns=['view'])
data_view.info()
data['condition'].unique()
# a checkpoint

data_condition = data_view.copy()
# one-hot encoding

data_condition = pd.get_dummies(data_condition, columns=['condition'])
data_condition.info()
# checkpoint

data_grade = data_condition.copy()
sns.distplot(data_grade['grade'])
sns.boxplot(data_grade['grade'])
# normal distribution check

grade_kurtosis = data_grade['grade'].kurt()

grade_skewness = data_grade['grade'].skew()



print ('Kurtosis and Skewness of Grade \n')

print ('Kurtosis: ' + str(grade_kurtosis))

print ('Skewness: ' + str(grade_skewness))
# checkpoint

data_above = data_grade.copy()
sns.distplot(data_above['sqft_above'])
sns.boxplot(data_above['sqft_above'])
# normal distribution check

above_kurtosis = data_above['sqft_above'].kurt()

above_skewness = data_above['sqft_above'].skew()



print ('Kurtosis and Skewness of Sqft_Above \n')

print ('Kurtosis: ' + str(above_kurtosis))

print ('Skewness: ' + str(above_skewness))
# checkpoint

data_basement = data_above.copy()
sns.distplot(data_basement['sqft_basement'])
# transform the data more than 0 into 1 which mean it has basement

data_basement.loc[data_basement.sqft_basement > 0, 'sqft_basement'] = 1
#rename the columns

data_basement = data_basement.rename({'sqft_basement': 'basement'}, axis=1)
data_built = data_basement.copy()
sns.distplot(data_built['yr_built'])
sns.boxplot(data_built['yr_built'])
# normal distribution check

built_kurtosis = data_built['yr_built'].kurt()

built_skewness = data_built['yr_built'].skew()



print ('Kurtosis and Skewness of Year Built \n')

print ('Kurtosis: ' + str(built_kurtosis))

print ('Skewness: ' + str(built_skewness))
data_renovated = data_built.copy()
data_renovated['yr_renovated'].value_counts()
# transform the data more than 0 into 1 which mean it has been renovated

data_renovated.loc[data_renovated.yr_renovated > 0, 'yr_renovated'] = 1
#rename the columns

data_renovated = data_renovated.rename({'yr_renovated': 'renovated'}, axis=1)
# checkpoint

data_lat = data_renovated.copy()
sns.distplot(data_lat['lat'])
sns.boxplot(data_lat['lat'])
# normal distribution check

lat_kurtosis = data_lat['lat'].kurt()

lat_skewness = data_lat['lat'].skew()



print ('Kurtosis and Skewness of Latitude \n')

print ('Kurtosis: ' + str(lat_kurtosis))

print ('Skewness: ' + str(lat_skewness))
# checkpoint

data_long = data_lat.copy()
sns.distplot(data_lat['long'])
sns.boxplot(data_lat['long'])
# normal distribution check

lat_kurtosis = data_lat['lat'].kurt()

lat_skewness = data_lat['lat'].skew()



print ('Kurtosis and Skewness of Longitude \n')

print ('Kurtosis: ' + str(lat_kurtosis))

print ('Skewness: ' + str(lat_skewness))
#Let's make a checkpoint and reset the index

data_cleaned = data_long.reset_index()
data_cleaned.head(100)
data_cleaned.columns
# First, We must split the data which one has quantitative data and which one has qualitative data



quantitative_data = data_cleaned[['day', 'month','bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade', 'sqft_above', 'yr_built','lat', 'long', 'log_sqft_lot']]



# I move the log_price to this, because It doesn't need to be standardize

qualitative_data = data_cleaned[['waterfront', 'basement','renovated','view_0','view_1','view_2', 'view_3', 'view_4', 'condition_1','condition_2', 'condition_3','condition_4','condition_5', 'log_price',]]
# Warmup the engine

from sklearn.preprocessing import StandardScaler



data_scaler = StandardScaler()
# standardize the data

scaled = data_scaler.fit_transform(quantitative_data)
scaled
# turn into pandas

scaled_quan = pd.DataFrame(scaled, columns=quantitative_data.columns)
# combine with categorical data

scaled_data = pd.concat([scaled_quan,qualitative_data], axis=1)
scaled_data.head(100)
scaled_data.info()
# First, We must declare which one is dependent variable, and which one is independet variables

y = scaled_data['log_price']

x = scaled_data.drop('log_price', axis=1)
# Import library for VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor



def calc_vif(X):



    # Calculating VIF

    vif = pd.DataFrame()

    vif["variables"] = X.columns

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



    return(vif)
calc_vif(x)
pre_regression = scaled_data.drop(['sqft_above', 'view_0', 'condition_5'], axis = 1)
# Big Checkpoint

pre_regression.to_csv('pre_regression.csv', index=False)
import statsmodels.api as sm
# First, We must declare which one is dependent variable, and which one is independet variables

y = pre_regression['log_price']

x1 = pre_regression.drop('log_price', axis=1)
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()
results.summary()
# import the module

from sklearn.model_selection import train_test_split
# define input and target

indepedent_vr = pre_regression['log_price']

dependent_vr = pre_regression.drop('log_price', axis=1)
# split the data into 80% train and 20% test, as a default It will shuffle the data before the split 

x_train, x_test, y_train, y_test = train_test_split(dependent_vr, indepedent_vr, test_size=0.2)
# regression

from sklearn import linear_model

reg = linear_model.LinearRegression()
# create the model based on training data

reg.fit(x_train, y_train)
# test the accuracy with test data

reg.score(x_test, y_test)
# We must shuffle the data first, because tensorflow doesn't have shuffle function

data_tf = pre_regression.sample(frac=1).reset_index(drop=True)
#split the data into 80% train, 10% validation, and 10 %test data

row_count = data_tf.shape[0]



train_data_count = int(0.8*row_count)

validation_data_count = int(0.1*row_count)

test_data_count = row_count - train_data_count - validation_data_count
#divide the data into targets and inputs

targets = data_tf['log_price']

inputs = data_tf.drop('log_price', axis=1)
train_inputs = inputs[:train_data_count]

train_targets = targets[:train_data_count]



validation_inputs = inputs[train_data_count:train_data_count+validation_data_count]

validation_targets = targets[train_data_count:train_data_count+validation_data_count]



test_inputs = inputs[train_data_count+validation_data_count:]

test_targets = targets[train_data_count+validation_data_count:]
# Import our module

import tensorflow as tf
# The number of inputs

inputs_size = len(inputs.columns)



# The number of targets

targets_size = 1



# The number of hidden layers

hidden_layer_size = 64
model = tf.keras.Sequential([

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(targets_size)

    ])



model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])



batch_size = 100

max_epochs = 100



# to prevent overfitting

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



# start the engine

model.fit(train_inputs, 

          train_targets, 

          batch_size = batch_size, 

          epochs = max_epochs, 

          callbacks = [early_stopping],

          validation_data = (validation_inputs, validation_targets), 

          verbose=2)
# Evaluate Our model with testing data

loss, mae, mse= model.evaluate(test_inputs, test_targets, verbose=2)
print("Testing set Mean Abs Error: " + str(mae))

print("Testing set Mean Sq Error: " + str(mse))