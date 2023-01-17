import pandas as pd

from sklearn.datasets import load_boston



data = load_boston()

df = pd.DataFrame(data.data, columns = data.feature_names)

df['target'] = data.target

df.head()
print(data.DESCR)
from sklearn.model_selection import train_test_split



X = data.data

y = data.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
import numpy as np

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)



print(

    np.round(lin_reg.coef_,1),

    '\n',

    np.round(lin_reg.intercept_, 1)

)
from sklearn.metrics import mean_squared_error

y_pred = lin_reg.predict(X_test)



print('R^2 Score:', lin_reg.score(X_test, y_test))

print('Mean squared error: ', mean_squared_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor



rf_reg = RandomForestRegressor(max_depth = 4)

rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)

print(rf_reg.feature_importances_)

print('\nR^2 Score:', rf_reg.score(X_test, y_test))

print('Mean squared error: ', mean_squared_error(y_test, y_pred_rf))
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)



rf_reg = RandomForestRegressor()

rf_params = {'n_estimators': [10 ,15],

             'criterion':['mse', 'mae'],

             'max_leaf_nodes': [10, 15],

             'max_depth': [4, 5], 

             'min_samples_split': [4, 5],

             'max_features': [5, 6]}



grid_search = GridSearchCV(rf_reg, rf_params)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print(best_rf)

best_rf.score(X_test, y_test)
from sklearn.datasets import load_wine



wine = load_wine()

wine_df = pd.DataFrame(wine.data, columns = wine.feature_names)

wine_df['target'] = wine.target

wine_df.head()
print(wine.DESCR)
X = wine.data

y = wine.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)



print(

    np.round(log_reg.coef_, 2), 

    '\n',

    np.round(log_reg.intercept_)

)
from sklearn.metrics import confusion_matrix, classification_report



print('Accuracy: ', log_reg.score(X_test, y_test))

y_pred = log_reg.predict(X_test)



conf_matrix = confusion_matrix(y_test, y_pred)

print('\nConfusion Matrix:\n', 

      pd.DataFrame(

          confusion_matrix(y_test, y_pred), 

          columns = ['pred: 0', 'pred: 1', 'pred: 2'],

          index = ['true: 0', 'true: 1', 'true: 2']), 

      '\n\nClassification Report:\n', 

      classification_report(y_test, y_pred))
# load data

city_attributes = pd.read_csv('../input/city_attributes.csv')

humidity = pd.read_csv('../input/humidity.csv')

pressure = pd.read_csv('../input/pressure.csv')

temperature = pd.read_csv('../input/temperature.csv')

weather_description = pd.read_csv('../input/weather_description.csv')

wind_direction = pd.read_csv('../input/wind_direction.csv')

wind_speed = pd.read_csv('../input/wind_speed.csv')



# besides the first dataframe, the data look a lot like this:

humidity.head()
# we can reshape these using pd.melt

humidity = pd.melt(humidity, id_vars = ['datetime'], value_name = 'humidity', var_name = 'City')

pressure = pd.melt(pressure, id_vars = ['datetime'], value_name = 'pressure', var_name = 'City')

temperature = pd.melt(temperature, id_vars = ['datetime'], value_name = 'temperature', var_name = 'City')

weather_description = pd.melt(weather_description, id_vars = ['datetime'], value_name = 'weather_description', var_name = 'City')

wind_direction = pd.melt(wind_direction, id_vars = ['datetime'], value_name = 'wind_direction', var_name = 'City')

wind_speed = pd.melt(wind_speed, id_vars = ['datetime'], value_name = 'wind_speed', var_name = 'City')



humidity.head()
# combine all of the dataframes created above 

weather = pd.concat([humidity, pressure, temperature, weather_description, wind_direction, wind_speed], axis = 1)

weather = weather.loc[:,~weather.columns.duplicated()] # indexing: every row, only the columns that aren't duplicates

weather.head()
# now we can merge this with the city attributes

weather = pd.merge(weather, city_attributes, on = 'City')

weather.head()
# create a variable for binary classification 

weather['weather_binary'] = np.where(weather['weather_description'].isin(["sky is clear", "broken clouds", "few clouds", 

                                                  "scattered clouds", "overcast clouds"]), 'good', 'bad')



# create a variable for multi-classification

conditions = [

    (weather['weather_description'].isin(["drizzle", "freezing_rain", "heavy intensity drizzle", 

                                          "heavy intensity rain", "heavy intensity shower rain", 

                                          "light intensity drizzle", "light intensity drizzle rain", 

                                          "light intensity shower rain", "light rain", "light shower rain", 

                                          "moderate rain", "proximity moderate rain", "ragged shower rain", 

                                          "shower drizzle", "very heavy rain", "proximity shower rain"])),

    (weather['weather_description'].isin(["broken clouds", "overcast clouds", "scattered clouds", "few clouds"])),

    (weather['weather_description'].isin(["heavy snow", "light rain and snow", "light shower sleet", "light snow", 

                                          "rain and snow", "shower snow", "sleet", "snow", "heavy shower snow"])), 

    (weather['weather_description'].isin(["thunderstorm with drizzle", "thunderstorm with heavy drizzle", 

                                          "thunderstorm with light drizzle", "thunderstorm with rain", 

                                          "thunderstorm with light rain", "heavy thunderstorm", 

                                          "proximity thunderstorm", "proximity thunderstorm with drizzle", 

                                          "proximity thunderstorm with rain", "proximity thunderstorm", 

                                          "thunderstorm", "ragged thunderstorm"])),

    (weather['weather_description'].isin(["sky is clear"]))]

     

choices = ['rain', 'cloudy', 'snow', 'thunder', 'clear']

weather['weather_broad'] = np.select(conditions, choices, default='other')



# sklearn models won't work with NaN values. There are a whole suite of imputation techniques used to replace empty 

# values with the most appropriate estimate, but for the sake of these challenges, we'll just remove these cases.



weather = weather.dropna()

weather.head()