from glob import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

from folium.plugins import FastMarkerCluster



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



import math



files = []

import os

for dirname, _, filenames in os.walk('/kaggle/input/london-met-police-crime-data-20192020'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))       
df1 = pd.read_csv(files[0])

df2 = pd.read_csv(files[1])
dataframes = [pd.read_csv(file).assign(crime_file=os.path.basename(file).strip(".csv")) for file in files]
data_full = pd.concat(dataframes, ignore_index=True)

data_full.shape[0]
data_full.head(8)
data_full.tail(8)
data_full.sample(8)
missing_values_count = data_full.isnull().sum()

print(missing_values_count)
print("Percentage of missing latitude and longitude values:(%.1f%%)" %(missing_values_count['Latitude']/data_full.shape[0]*100))
data_full[pd.isnull(data_full['LSOA name'])].shape

location_nan_clean = data_full.dropna(axis=0, subset=['LSOA name'])
location_nan_clean.sample(70)

data = location_nan_clean.reset_index()
data.tail()
plt.figure(figsize=(14,10))

plt.title('Number of crimes 2019-2020')

plt.ylabel('Crime Type')

plt.xlabel('Number of Crimes')



data.groupby([data['Crime type']]).size().sort_values(ascending=True).plot(kind='barh')



plt.show()
data['month'] = pd.DatetimeIndex(data['Month']).month
data.head()
data.tail()
# Get list of categorical variables

s = (data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if data[col].nunique() < 15]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
data.tail()
cat_crime = pd.get_dummies(data['Crime type'])



cat_crime.Drugs.value_counts()
cat_crime.head()
merged = pd.concat([data, cat_crime], axis='columns')



merged.head()
for col in merged.columns: 

    print(col)
final = merged.drop(['month','Crime type', 'LSOA code','LSOA name', 'crime_file', 'Crime ID', 'Reported by', 'Falls within', 'Last outcome category', 'Context','Location'], axis='columns')



final.reset_index



final.sample(8)
# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Burglary and drugs 2019-2020")











sns.lineplot(x=final['Month'],y=final['Drugs'],label='Drugs')



sns.lineplot(x=final['Month'],y=final['Violence and sexual offences'], label="Violence and sexual offenses")







# Add label for horizontal axis

plt.xlabel("Month")
m_1 = folium.Map(location=[51.4203,0.0705], tiles='cartodbpositron', zoom_start=10)



# Add a heatmap to the base map

HeatMap(data=final[['Latitude', 'Longitude','Violence and sexual offences']], radius=12).add_to(m_1)



# Display the map

m_1
# Create the map

m_2 = folium.Map(location=[51.4203,0.0705], tiles='cartodbpositron', zoom_start=12)



#

final_sample = final.sample(n=25000)

# Add points to the map

mc = MarkerCluster()







for idx, row in final_sample.iterrows():

    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

        mc.add_child(Marker([row['Latitude'], row['Longitude']],popup=(row == 1).idxmax(axis=1)))



m_2.add_child(mc)









# Display the map

m_2
plt.figure(figsize = (15,8))

sns.scatterplot(final.Latitude, final.Longitude)
final.sample(8)
data.isnull().sum()

data.shape
learn_data = data.dropna(axis=1)
learn_data.head()
final_learn = learn_data.drop(['Reported by', 'Falls within', 'Location', 'LSOA name', 'LSOA code', 'crime_file','index','Month'], axis=1 )
final_learn.head(8)
# Getting list of categorical variables

s = (final_learn.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
final = pd.get_dummies(final_learn,prefix=[''],drop_first=True)

final.head(10)
sns.pairplot(final[['Latitude','Longitude','_Violence and sexual offences','_Drugs','_Burglary','_Robbery','month']],height=2)
final["Latitude"] = np.radians(final["Latitude"])

final["Longitude"] = np.radians(final["Longitude"])

final.head()
y = final['month']

crime_features = final.columns[3:-1]

print(crime_features)

X = final[crime_features]
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)



# Define model. Specify a number for random_state to ensure same results each run

crime_model = DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,

                      max_features=None, max_leaf_nodes=25,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, presort='deprecated',

                      random_state=0, splitter='best')



# Fit model

crime_model.fit(train_X, train_y)





predicted_crimes = crime_model.predict(val_X)



mean_absolute_error(val_y, predicted_crimes)



import numba

print(np.round(predicted_crimes[:100]))
# from numba import jit



# @jit(nopython=True)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

mae_dict = {}

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf_nodes in candidate_max_leaf_nodes:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))    

    mae_dict[max_leaf_nodes] = my_mae

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = min(mae_dict, key=mae_dict.get)
# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=0)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)
from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(random_state=1)



# fit your model

rf_model.fit(train_X, train_y)



# Calculate the mean absolute error of your Random Forest model on the validation data

rf_preds = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(val_y, rf_preds)



print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

print("Model predictions: {}" .format(np.round(rf_preds)[0:20]))                   