import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
houses = pd.read_csv('../input/kc-house-data/kc_house_data.csv')
houses.head()
houses.info()
houses.isnull().sum()
houses[houses['sqft_above'].isnull()]
plt.figure(figsize=(10, 7))

sns.boxplot(x='grade', y='sqft_above', data=houses)
houses['sqft_above'] = houses[['sqft_above', 'grade']].apply(

    lambda sqft_grade:

    houses.groupby('grade').mean()['sqft_above'].loc[sqft_grade[1]] if pd.isnull(sqft_grade[0]) else sqft_grade[0],

    axis=1

)
plt.figure(figsize=(20, 10))

sns.heatmap(houses.drop('id', axis=1).corr(), annot=True, cmap='viridis_r')
plt.figure(figsize=(10, 7))

houses.corr().sort_values('price').drop('price')['price'].plot(kind='bar', title='Correlation with house prices in King County')
plt.figure(figsize=(10, 7))

sns.countplot(x='bedrooms', data=houses).set(ylabel='Count', title='Number of different houses depending on bedrooms', xlabel='Number of bedrooms')
plt.figure(figsize=(10, 7))

sns.boxplot(x='bedrooms', y='price', data=houses).set(xlabel='Number of Bedrooms', ylabel='Price', title='Comparison of House price and number of bedrooms')
print('Correlation value between Number of Bedrooms and Price: ', houses.corr().loc['price', 'bedrooms'])
plt.figure(figsize=(10, 7))

sns.scatterplot(x='sqft_living', y='price', data=houses).set(xlabel='Sqft Living Space', ylabel='Price of the house')
plt.figure(figsize=(10, 7))

sns.boxplot(x='waterfront', y='price', data=houses).set(xlabel='', ylabel='Price', title='Comparison of prices for houses having a waterfront or not',

                                                        xticklabels=['Do not have waterfront', 'Have waterfront'])
plt.figure(figsize=(10, 7))

sns.countplot(x='waterfront', data=houses).set(xlabel='', ylabel='Price', title='Number of waterfront and non waterfront houses',

                                             xticklabels=['Do not have waterfront', 'Have waterfront'])
plt.figure(figsize=(10, 7))

sns.lineplot(x='yr_built', y='price', data=houses)
houses['century_old'] = houses['yr_built'].apply(lambda year: 1 if year <= 1915 else 0)
plt.figure(figsize=(10, 7))

sns.boxplot(x='century_old', y='price', data=houses).set(xlabel='', ylabel='Price', title='Comparison of prices depending on the year the house was built',

                                                        xticklabels=['Less than 100 years old', 'Over a 100 years Old'])
plt.figure(figsize=(10, 7))

sns.countplot(x='century_old', data=houses).set(xlabel='', ylabel='Price', title='Number of Houses according to thier age',

                                                xticklabels=['Less than 100 years old', 'Over a 100 years Old'])
houses.drop('yr_built', axis=1, inplace=True)
houses['yr_renovated'].value_counts()
plt.figure(figsize=(10, 7))

sns.lineplot(x='yr_renovated', y='price', data=houses[houses['yr_renovated'] != 0]).set(xlabel='Year of Renotation', ylabel='Price',

                                                                                        title='Relation between house price and year of renovation')
plt.figure(figsize=(10, 8))

sns.scatterplot(x='long', y='lat', data=houses, hue='price', palette='magma_r', alpha=0.15)
print('Percentage of houses priced below 3 million USD: ', len(houses[houses['price'] < 3000000]) / len(houses) * 100)

print('Percentage of houses priced below 2.5 million USD: ', len(houses[houses['price'] < 2500000]) / len(houses) * 100)

print('Percentage of houses priced below 2 million USD: ', len(houses[houses['price'] < 2000000]) / len(houses) * 100)

print('Percentage of houses priced below 1.5 million USD: ', len(houses[houses['price'] < 1500000]) / len(houses) * 100)
plt.figure(figsize=(10, 8))

sns.scatterplot(x='long', y='lat', data=houses[houses['price'] < 2000000], hue='price', palette='magma_r', alpha=0.15)
houses.drop('zipcode', axis=1, inplace=True)
lat_mid = houses['lat'].min() + ((houses['lat'].max() - houses['lat'].min()) / 2)

long_mid = houses['long'].min() + ((houses['long'].max() - houses['long'].min()) / 2)
houses['zone'] = houses[['lat', 'long']].apply(

    lambda lat_long:

    1 if ((lat_long[0] < lat_mid) and (lat_long[1] < long_mid)) else (

        2 if ((lat_long[0] >= lat_mid) and (lat_long[1] < long_mid)) else (

            3 if ((lat_long[0] < lat_mid) and (lat_long[1] >= long_mid)) else 4

        )

    ),

    axis=1

)
houses = pd.concat([houses.drop(['lat', 'long', 'zone'], axis=1), pd.get_dummies(houses['zone'], drop_first=True)], axis =1)
houses.head()
houses['date'] = pd.to_datetime(houses['date'])
houses['year_sold'] = houses['date'].apply(lambda date: date.year)

houses['month_sold'] = houses['date'].apply(lambda date: date.month)
plt.figure(figsize=(10, 7))

sns.boxplot(x='year_sold', y='price', data=houses)
plt.figure(figsize=(10, 7))

sns.boxplot(x='month_sold', y='price', data=houses)
print('min: ', houses['date'].min(), '\nmax: ',houses['date'].max())
houses.drop(['date', 'year_sold', 'month_sold'], axis=1, inplace=True)
houses.drop('id', axis=1, inplace=True)
houses.info()
X = houses.drop('price', axis=1)

y = houses['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
random_grid = {

    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)],

    'max_features': ['auto', 'sqrt'],

    'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],

    'min_samples_split': [2, 5, 10, 15, 100],

    'min_samples_leaf': [1, 2, 5, 10]

}
rf = RandomForestRegressor()
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=1, random_state=11)
rf_random_search.fit(X_train, y_train)
rf_random_search.best_params_
rf_predictions = rf_random_search.predict(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
ann = Sequential()
ann.add(Dense(18, activation='relu'))

ann.add(Dense(18, activation='relu'))

ann.add(Dense(18, activation='relu'))

ann.add(Dense(18, activation='relu'))

ann.add(Dense(1))
ann.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)
ann.fit(x=X_train, y=y_train.values, verbose=1, batch_size=32, epochs=10000, validation_data=(X_test, y_test.values), callbacks=[early_stop])
ann_predictions = ann.predict(X_test)
predictions_df = pd.DataFrame(y_test)

predictions_df['Linear Regression'] = lr_predictions

predictions_df['Random Forrest Regressor'] = rf_predictions

predictions_df['Artifical Neural Network'] = ann_predictions
predictions_df.head()
sns.pairplot(predictions_df, x_vars=['Linear Regression', 'Random Forrest Regressor', 'Artifical Neural Network'], y_vars=['price'], height=7)
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
print('Linear Regression:')

print('Mean Absolute Error:', mean_absolute_error(predictions_df['price'], predictions_df['Linear Regression']))

print('Mean Squared Error:', mean_squared_error(predictions_df['price'], predictions_df['Linear Regression']))

print('Explained Variance Score:', explained_variance_score(predictions_df['price'], predictions_df['Linear Regression']))

print('R2 Score:', r2_score(predictions_df['price'], predictions_df['Linear Regression']))
print('Random Forrest Regressor:')

print('Mean Absolute Error:', mean_absolute_error(predictions_df['price'], predictions_df['Random Forrest Regressor']))

print('Mean Squared Error:', mean_squared_error(predictions_df['price'], predictions_df['Random Forrest Regressor']))

print('Explained Variance Score:', explained_variance_score(predictions_df['price'], predictions_df['Random Forrest Regressor']))

print('R2 Score:', r2_score(predictions_df['price'], predictions_df['Random Forrest Regressor']))
print('Artifical Neural Network:')

print('Mean Absolute Error:', mean_absolute_error(predictions_df['price'], predictions_df['Artifical Neural Network']))

print('Mean Squared Error:', mean_squared_error(predictions_df['price'], predictions_df['Artifical Neural Network']))

print('Explained Variance Score:', explained_variance_score(predictions_df['price'], predictions_df['Artifical Neural Network']))

print('R2 Score:', r2_score(predictions_df['price'], predictions_df['Artifical Neural Network']))