import pandas as pd

import numpy as np

import folium

import matplotlib.pyplot as plt

import seaborn as sns

#

from sklearn.preprocessing import LabelEncoder

#

from sklearn.model_selection import train_test_split

#

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

#

import warnings

warnings.filterwarnings('ignore')

#

%matplotlib inline
df_ca = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

df_ca.head(10)
latitude = 36.778259

longitude = -119.417931

CA_map= folium.Map(location=[latitude, longitude], zoom_start=6)





houses = folium.map.FeatureGroup()

for lat, lng, in zip(df_ca.latitude, df_ca.longitude):

    houses.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5,

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

            )

    )

CA_map.add_child(houses)    
df_ca.shape
missing_data = df_ca.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
df_ca.dropna(inplace=True)
plt.figure(figsize=(12,8))

df_ca['median_house_value'].plot(kind='hist', bins=100)

plt.title('Median House Value')

plt.xlabel('Median Value')

plt.ylabel('Number of Houses')

plt.show()
df_ca.hist(figsize=(30,30), bins=100)

plt.show()
cor = df_ca.corr()

cor.style.background_gradient()
df_ca.describe()
df_ca['ocean_proximity'].value_counts()
df_ca = pd.get_dummies(df_ca,drop_first = True)

df_ca = df_ca.drop('ocean_proximity_ISLAND', 1)

df_ca.head()
X = df_ca.drop('median_house_value', axis=1)

Y = df_ca['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=26)
print('train set shape: ',x_train.shape)

print('test set shape: ', x_test.shape)
print('train set shape: ', y_train.shape)

print('test set shaoe: ', y_test.shape)
models = []

models.append(('Linear Regression', LinearRegression()))

models.append(('Decision Trees', DecisionTreeRegressor()))

models.append(('K Nearest Neighbor',KNeighborsRegressor()))

models.append(('Random Forest Regressor', RandomForestRegressor()))

results = []

names = []

for i, j in models: 

    k = KFold(n_splits=10 , random_state=42)

    result = cross_val_score(j, x_train,y_train, cv=k, scoring='r2')

    results.append(result)

    names.append(i)

    print('Model: ', i,'Score: %.2f' % result.mean(), "Model's Standard Deviation: %.2f" % result.std())
f,ax = plt.subplots(figsize = (14,10))

sns.boxplot(x=names, y=results,palette='BuGn_r')

plt.title("Comparison between models' accuracy", fontsize=18, color='green')

plt.ylabel('Accuracy', fontsize=14,color='blue')

plt.xlabel('Model',fontsize=14,color='blue')
RF = RandomForestRegressor()

RF.fit(x_train,y_train)



yhat = RF.predict(x_test)

print('Accuracy of the model: %.2f' % r2_score(y_test, yhat))
df = pd.DataFrame({'avg_price': y_test, 'pred': yhat})

df1 = pd.DataFrame(x_test)

df_pred = df1.join(df)
df_pred['error'] = df_pred['avg_price']- df_pred['pred']

df_co = df_pred[ (df_pred['error'] < 10000) & (df_pred['error']>-10000)]
latitude = 36.778259

longitude = -119.417931

CA_map= folium.Map(location=[latitude, longitude], zoom_start=6)





houses = folium.map.FeatureGroup()

for lat, lng, in zip(df_co.latitude, df_co.longitude):

    houses.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5,

            color='yellow',

            fill=True,

            fill_color='green',

            fill_opacity=0.6

            )

    )

CA_map.add_child(houses)    
