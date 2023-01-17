# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=random_state)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#Load cropland data into home_data
cropland_csv_path = '../input/Combined_Clean.csv'
land_data = pd.read_csv(cropland_csv_path)

state_data = land_data[land_data['Region or State']=='State']
state_data = state_data.drop(columns=['Region or State'])
state_data = state_data.drop(columns=['Region'])
region_data = land_data[land_data['Region or State']=='Region']
region_data = region_data.drop(columns=['Region or State'])
region_data = region_data.drop(columns=['Region']) #DROP Region because it is duplicated into state and it is easier later on to always look for geographic region in state.

region_real_estate_data = region_data[region_data['LandCategory']=='Farm Real Estate']
region_real_estate_data = region_real_estate_data.drop(columns=['LandCategory'])
state_real_estate_data = state_data[state_data['LandCategory']=='Farm Real Estate']
state_real_estate_data = state_real_estate_data.drop(columns=['LandCategory'])

region_pasture_data = region_data[region_data['LandCategory']=='Pasture']
region_pasture_data = region_pasture_data.drop(columns=['LandCategory'])
state_pasture_data = state_data[state_data['LandCategory']=='Pasture']
state_pasture_data = state_pasture_data.drop(columns=['LandCategory'])

region_cropland_data = region_data[region_data['LandCategory']=='Cropland']
region_cropland_data = region_cropland_data.drop(columns=['LandCategory'])
state_cropland_data = state_data[state_data['LandCategory']=='Cropland']
state_cropland_data = state_cropland_data.drop(columns=['LandCategory'])

#############################################
# Toggle data between all, state and region #
#############################################
#### Note always using state because the region name is replicated into state when it is a region row.
#################
#### Default ####
#feature_names=['State', 'LandCategory', 'Year', 'Region or State']

#land_data = state_data
#feature_names=['State', 'LandCategory', 'Year']

#land_data = region_data
#feature_names=['State', 'LandCategory', 'Year']

#land_data = state_real_estate_data
#land_data = state_pasture_data
land_data = state_cropland_data
#land_data = region_real_estate_data
#land_data = region_pasture_data
#land_data = region_cropland_data
feature_names=['State', 'Year']

land_data.reset_index(drop=True, inplace=True)

land_data.head()
#Interpolate missing sales data. - Potential bug since this could cross the boundary between States/Land Categories, etc
#land_data = land_data.sort_values(['State', 'LandCategory', 'Year'], ascending=[True, True, True])
land_data = land_data.sort_values(['State', 'Year'], ascending=[True, True])
land_data = land_data.interpolate(method='linear', axis=0).ffill().bfill()
land_data.reset_index(drop=True, inplace=True)
land_data
land_data.describe()
%matplotlib inline
pd.plotting.scatter_matrix(land_data, alpha = 0.3, figsize = (12, 12))
plt.show()
###
### Abandoned attempt at building a line chart similar to https://www.kaggle.com/safavieh/public-kernels-effect-analyzing-public-lb/notebook
###
#%matplotlib inline
#plot_data = land_data
##plot_data = land_data.groupby(['State'])
#States = defaultdict(lambda :[])
#for year in range(1997,2017):
#    States['State'] = land_data['State']
#    States['Year'] = land_data['Year']
#    States['Acre Value'] = land_data['Acre Value']
##    for state in land_data:
##       States[state].append(T_temp)
#pl.figure(figsize=(10,10))
#cNorm = colors.Normalize(vmin=0, vmax=500)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='plasma')
#for i, state in enumerate(States['State']):
#    print(state)
##for i, state in enumerate(States):
##    pl.plot(States[state], color=scalarMap.to_rgba(i), alpha=.6)
##
##pl.tight_layout()
##pd.plotting.scatter_matrix(plot_data, alpha = 0.3, figsize = (12, 12), c=land_data['State'])
##plt.show()
#This can only run w. one type of land category (ie Only Cropland or Real Estate or Pastureland )
pivoted_land_data = land_data.pivot(index='Year',columns='State', values='Acre Value')
pivoted_land_data.head()
#Narrow data to features
X = land_data[feature_names]
y = land_data[['Acre Value']]

X_oneHot = pd.get_dummies(X[['State']])
X_oneHot.head()
X = X.drop(columns=['State'])
X = pd.concat([X, X_oneHot], axis=1, sort=False)
X.head()
y.head()
from sklearn.preprocessing import MinMaxScaler

def scale(matrix):
    scaler = MinMaxScaler(feature_range=(-.8,.8))
    scaler.fit(matrix.values)
    scaled_columns = scaler.transform(matrix.values)
    return scaler, scaled_columns

scaler, scaled_data = scale(y)

scaled_data
scaler.inverse_transform(scaled_data)
#X['scaled_y'] = scaled_data.toarray().tolist()
#X.head()
scaled_df = pd.DataFrame(data=scaled_data)
scaled_df
#split out training and validation data
random_state = 0
train_X, val_X, train_y, val_y = train_test_split(X, scaled_df, random_state=random_state)

final_train, final_test = train_X.align(val_X, join='left', axis=1)

#Check for missing column data
missing_val_count_by_column = (land_data.isnull().sum())
# print(missing_val_count_by_column[missing_val_count_by_column > 0])
#print("Missing column data %s" %(missing_val_count_by_column))

train_X.head()
train_y.head()
val_X.head()
val_y.head()
scaler.inverse_transform(train_y)
##
## Big Graph - Slow
## 
#pd.plotting.scatter_matrix(pivoted_land_data, alpha = 0.3, figsize = (12, 12))
from matplotlib import pyplot as plt
import seaborn as sns
corr = pd.get_dummies(land_data).corr()
a4_dims=(11.7, 8.27)
fix, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax)
#
# Decision Tree Regressor Model
# 

#Fit to Decision Tree Regressor model
land_model = DecisionTreeRegressor(random_state=random_state)
land_model.fit(final_train, train_y)

#Do predictions and check accuracy of Model
val_predictions = land_model.predict(final_test)
val_mae = mean_absolute_error(val_y, val_predictions)
print("**** Auto MAE %d" %(val_mae))

#Find optimal fit (avoid overfitting or underfitting the model)
#for max_leaf_nodes in [2, 3, 4, 5, 6, 7, 8 , 9 , 10, 100, 1000, 2000, 3000, 4000, 5000]:
#for max_leaf_nodes in range(2, 3000):
val_mae = get_mae(1575, final_train, final_test, train_y, val_y)
print("Max leaf nodes: 1575 \t MAE: %d" %(val_mae))
   
#Now that we have identified 1575 as good fit - note this didn't actually work well, the values just kept decreasing on this dataset.
fit_model = DecisionTreeRegressor(max_leaf_nodes=1575, random_state=random_state)
all_data = pd.get_dummies(X)
fit_model.fit(all_data, y)

#
# Random Forests - defaults to 10 trees.
# 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(final_train, train_y)
land_forest_preds = forest_model.predict(final_test)
print(mean_absolute_error(val_y, land_forest_preds))
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.constraints import maxnorm

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000001)

batch_size = 128
input_size = train_X.shape[1]

print("input_size = " + str(input_size))

model = Sequential()
#model.add(LSTM(1000, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dense(256,  input_shape=(input_size,), activation='tanh', kernel_constraint=maxnorm(3)))
#model.add(Dense(1024,  activation='tanh', kernel_constraint=maxnorm(3)))
model.add(Dense(1024,  activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=opt)
print(model.summary())
import keras.backend as K
epochs = 10
K.set_value(model.optimizer.lr, 0.000001)
model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, shuffle=True)
output_x = pd.DataFrame(columns=X.columns, data=val_X)
output_x.head()
y_predict_scaled = model.predict(val_X)
y_predict = scaler.inverse_transform(y_predict_scaled)
y_predict = y_predict.reshape(1,-1)[0]

y_test_unscaled = scaler.inverse_transform(val_y)
y_test_unscaled = y_test_unscaled.reshape(1,-1)[0]

l = list(zip(y_predict, y_test_unscaled))
val_X.shape
val_y.shape
y_test_unscaled.shape
val_y.head()
output_y = pd.DataFrame(columns=["predicted_acre_value"], data=y_predict, index=val_X.index.values)
output_y.shape
output_y.head()
actual_y = pd.DataFrame(columns=["test_acre_value"], data=y_test_unscaled, index=val_y.index.values)
actual_y.shape
actual_y.head()
output_df = pd.concat([output_x, output_y, actual_y], axis=1)
output_df.head()
model.evaluate(val_X, val_y)
pred_list = output_y["predicted_acre_value"]
actual_list = actual_y["test_acre_value"]
delta_frame = pd.DataFrame(columns=["predicted_acre_value"], data=pred_list)
delta_frame["test_acre_value"] = pd.DataFrame(columns=["test_acre_value"], data=actual_list)
delta_frame.head()
#page=1
#page_size=288
#start=page*page_size
#pred_list, actual_list = zip(*l[start:start+page_size])
#delta_list = [p - a for p, a in l]

pred_list, actual_list = zip(*l[0:])
delta_list = [p - a for p, a in l]
plt.figure(figsize = (20,16))
pred_plt = plt.plot(pred_list, label="predicted")
actual_plt = plt.plot(actual_list, label="actual")
plt.legend()
#plt.ylim((-1,1))
ax = plt.axes()        
ax.yaxis.grid(True) # horizontal lines
ax.xaxis.grid(True) # vertical lines

plt.figure(figsize = (20,16))
plt.hist(delta_list, density=True, bins=100)
plt.legend()
plt.figure(figsize = (20,16))
plt.plot(output_df['Acre Value'][0:], label="VAL")
plt.show()