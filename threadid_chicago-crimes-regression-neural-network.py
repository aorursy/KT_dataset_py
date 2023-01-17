# local Operating System
import os

# Match and Science
import numpy as np
np.random.seed(123) # for reproducability
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import sklearn

# Machine Learning
import tensorflow as tf
import keras

# Visualisation
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Dataframe
import pandas as pd

# SQL - PostgreSQL
import bq_helper

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model

from numpy import array
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize
from shapely.geometry import Point, Polygon
from bq_helper import BigQueryHelper
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="chicago_crime") 
city_sql_commarea = """
WITH cr AS
(
    SELECT DISTINCT
    community_area
    , COUNT(*) AS crimes 
    FROM  `bigquery-public-data.chicago_crime.crime`
    WHERE year = 2012
    GROUP BY community_area
)
SELECT 
cr.crimes
, cr.community_area
FROM cr
ORDER BY cr.crimes DESC, cr.community_area
"""
chicri_commarea = chicago_crime.query_to_pandas(city_sql_commarea)
chipop_df = pd.read_csv('../input/Chicago_Census_SociaEcon_CommArea_2008_2012.csv')
chipop_columns = [
'community_area',
'community_area_name', 
'pct_housing_crowded', 
'pct_households_below_poverty', 
'pct_age16_unemployed',
'pct_age25_nohighschool',
'pct_not_working_age',
'per_capita_income',
'hardship_index']
chipop_df.columns = chipop_columns

chicri_commarea = pd.merge(chicri_commarea, chipop_df, on='community_area', how='inner')
chicri_commarea = chicri_commarea.dropna(inplace=False) 
chicri_commarea_df = chicri_commarea[['community_area_name',
                                'pct_housing_crowded',
                                'pct_households_below_poverty',
                                'pct_age16_unemployed',
                                'pct_age25_nohighschool',
                                'pct_not_working_age',
                                'per_capita_income',
                                'hardship_index',
                                'crimes']]
sns.set(style='ticks', color_codes=True)
sns.pairplot(chicri_commarea_df, diag_kind='kde', kind='reg')
plt.figure(figsize=(15,30))
sns.barplot(x='crimes', y='community_area_name', data=chicri_commarea_df, ci=None)

city_sql_primtype = """
WITH cr AS
(
    SELECT DISTINCT
    primary_type
    , COUNT(*) AS crimes 
    FROM  `bigquery-public-data.chicago_crime.crime`
    WHERE year = 2012
    GROUP BY primary_type
)
SELECT 
cr.crimes
, cr.primary_type
FROM cr
ORDER BY  cr.crimes DESC, cr.primary_type
"""
chicri_primtype_df = chicago_crime.query_to_pandas(city_sql_primtype)
plt.figure(figsize=(10,15))
sns.barplot(x='crimes', y='primary_type', data=chicri_primtype_df, ci=None)

city_commarea_month = """
WITH cr AS
(
    SELECT DISTINCT
    community_area
    , EXTRACT(MONTH FROM CAST(date AS DATE)) AS crime_month
    , primary_type
    , COUNT(*) AS crimes 
    FROM  `bigquery-public-data.chicago_crime.crime`
    WHERE year = 2012
    GROUP BY primary_type, community_area, crime_month
)
SELECT 
cr.crimes
, CAST(cr.crime_month AS STRING) AS crime_month
, cr.primary_type
, cr.community_area
FROM cr
ORDER BY cr.community_area, cr.crime_month, cr.primary_type
"""

city_data = chicago_crime.query_to_pandas(city_commarea_month)
city_data = pd.merge(city_data, chipop_df, on='community_area', how='inner')
city_data = city_data.dropna(inplace=False)  
city_data = city_data.drop('community_area_name', 1)
city_data['community_area'] = city_data['community_area'].astype(str)
city_data_with_dummies = pd.get_dummies(city_data)
city_data_with_dummies.head(1)
city_crime_data = city_data_with_dummies.values
X = city_crime_data[:,1:].astype(float) 
Y = city_crime_data[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train[:,1] = X_train[:,1] / X_train[:,1].max()
X_train[:,2] = X_train[:,2] / X_train[:,2].max()
X_train[:,3] = X_train[:,3] / X_train[:,3].max()
X_train[:,4] = X_train[:,4] / X_train[:,4].max()
X_train[:,5] = X_train[:,5] / X_train[:,5].max()
X_train[:,6] = X_train[:,6] / X_train[:,6].max()
X_train[:,7] = X_train[:,7] / X_train[:,7].max()

X_test[:,1] = X_test[:,1] / X_test[:,1].max()
X_test[:,2] = X_test[:,2] / X_test[:,2].max()
X_test[:,3] = X_test[:,3] / X_test[:,3].max()
X_test[:,4] = X_test[:,4] / X_test[:,4].max()
X_test[:,5] = X_test[:,5] / X_test[:,5].max()
X_test[:,6] = X_test[:,6] / X_test[:,6].max()
X_test[:,7] = X_test[:,7] / X_test[:,7].max()

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='glorot_uniform', activation='relu'))  
model.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))  
model.add(Dense(1, kernel_initializer='glorot_uniform'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

history = model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=0, validation_split=.2)
plt.plot(history.history['val_loss'])
plt.title('Value Loss')
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.plot(history.history['val_mean_squared_error'])
plt.title('Value Mean Squared Error')
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.plot(history.history['loss'])
plt.title('Loss')
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.plot(history.history['mean_squared_error'])
plt.title('Mean Squared Error')
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
pred_crimes = model.predict(X_test)
mse_pred_score = metrics.mean_squared_error(pred_crimes, Y_test)
print('mse_pred_score {}'.format(mse_pred_score))
rmse_pred_score = np.sqrt(mse_pred_score)
print('rmse_pred_score {}'.format(rmse_pred_score))
r2_pred_score = r2_score(Y_test, pred_crimes, multioutput='uniform_average')  
print('r2_pred_score - Coefficient of Determination {}'.format(r2_pred_score))

matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
fig, ax = plt.subplots(figsize=(50, 40))
plt.style.use('ggplot')
plt.plot(pred_crimes, Y_test, 'ro')
plt.xlabel('Predicted Crime', fontsize = 30)
plt.ylabel('Actual Crime', fontsize = 30)
plt.title('Predicted Y (Crimes) to the Actual Y (Crimes)', fontsize = 30)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
plt.show()