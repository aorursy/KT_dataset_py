import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
import folium
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 

from sklearn.linear_model import Lasso, LinearRegression, Ridge, RANSACRegressor, SGDRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.svm import SVR
df = pd.read_csv('../input/housing.csv')

df.head()
df.shape
df.info()
df.describe()
df.ocean_proximity.value_counts()
df.duplicated().sum()
df.isnull().sum()
print(f'percentage of missing values: {df.total_bedrooms.isnull().sum() / df.shape[0] * 100 :.2f}%')
df = df.fillna(df.median())

df.isnull().sum()
sns.scatterplot(df.longitude, df.latitude)
sns.relplot(x="longitude", y="latitude", hue="median_house_value", size="population", alpha=.5,\

            sizes=(50, 700), data=df, height=8)

plt.show()
# Create a map with folium centered at the mean latitude and longitude

cali_map = folium.Map(location=[35.6, -117], zoom_start=6)



# Display the map

display(cali_map)
# Add markers for each rows

for i in range(df.shape[0]):

    folium.Marker((float(df.iloc[i, 1]), float(df.iloc[i, 0]))).add_to(cali_map) 

    

# Display the map

display(cali_map)
plt.figure(figsize=(10, 4))

sns.distplot(df.median_house_value)

plt.show()
df.ocean_proximity.unique()
plt.figure(figsize=(10, 4))

for prox in df.ocean_proximity.unique():

    sns.kdeplot(data=df[df.ocean_proximity == prox].median_house_value)

    plt.legend(prox)

plt.show()
sns.pairplot(df)

plt.show()
df.hist(figsize=(8, 8))

plt.show()
corr = df.corr()

corr
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 6))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
df = pd.get_dummies(data=df, columns=['ocean_proximity'], drop_first=False)

df.head()
feat_removed = ['median_house_value']



# removed 

#['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income',

#'median_house_value', 'ocean_proximity']
y = df.median_house_value

X = df.drop(columns=feat_removed)

X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def calculate_rmse(model, model_name):

    model.fit(X_train, y_train)

    y_pred, y_pred_train = model.predict(X_test), model.predict(X_train)

    rmse_test, rmse_train = np.sqrt(mean_squared_error(y_test, y_pred)), np.sqrt(mean_squared_error(y_train, y_pred_train))

    print(model_name, f' RMSE on train: {rmse_train:.0f}, on test: {rmse_test:.0f}')

    return rmse_test
lr = LinearRegression()

lr_err = calculate_rmse(lr, 'Linear Reg')
ra = RANSACRegressor()

ra_err = calculate_rmse(ra, 'RANSAC Reg')
la = Lasso()

la_err = calculate_rmse(la, 'Lasso Reg')
sg = SGDRegressor()

sg_err = calculate_rmse(sg, 'SGD Reg')
ri = SGDRegressor()

ri_err = calculate_rmse(ri, 'Ridge')
ad = AdaBoostRegressor()

ad_err = calculate_rmse(ad, 'AdaBoostRegressor')
sv = SVR()

sv_err = calculate_rmse(sv, 'SVR')
df_score = pd.DataFrame({'Model':['Linear Reg', 'RANSAC Reg', 'Lasso Reg', 'AdaBoost', 'SVR'], 

                         'RMSE':[lr_err, ra_err, la_err, ad_err, sv_err]})

ax = df_score.plot.barh(y='RMSE', x='Model')