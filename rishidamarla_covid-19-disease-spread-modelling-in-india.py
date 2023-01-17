import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings 
warnings.filterwarnings('ignore')
df = pd.read_json("../input/indian-covid19-cases/covid19-INDIA.txt")
df.head()
# Dropping all unnecessary features.
df = df.drop(columns=['CountryCode', 'Province', 'City', 'CityCode', 'Lat', 'Lon', 'Status'])
df.info()
df.describe()
df.dtypes
# Adding up all the number of cases in a particular day across all states.
df2 = df.groupby("Date", as_index=False).Cases.sum()
df2.head()
# Inserting a new index like column.
df2.insert(0, 'Days Since First Confirmed Case', range(0, 0 + len(df2)))
df2.head()
# Visualizing the data.
plt.figure(figsize=(10,10))
plt.plot(df2["Days Since First Confirmed Case"], df2["Cases"], 'ro')
plt.title("Number of confirmed cases since the day of the first reported case in India")
plt.xlabel("Days Since First Confirmed Case")
plt.ylabel('Confirmed Cases')
plt.show()
days_since_1_26 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
x_data = np.array(x_data).reshape(-1,1)
y_data = np.array(y_data).reshape(-1,1)
days_in_future = 20
future_forecast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-20]
import datetime
start = '1/26/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(days_since_1_26, y_data, test_size=0.2, shuffle=False) 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)
poly_future_forecast = poly.fit_transform(future_forecast)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train, y_train)
test_linear_pred = linear_model.predict(poly_X_test)
linear_pred = linear_model.predict(poly_future_forecast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test) )
print('MSE:', mean_squared_error(test_linear_pred, y_test))
def plot_predictions_linear(x, y, pred, algorithm, color):
    plt.figure(figsize=(10,10))
    plt.plot(x,y)
    plt.plot(future_forecast, linear_pred, color = color)
    plt.title("Number of confirmed cases in India")
    plt.xlabel("Days Since First Confirmed Case")
    plt.ylabel('Confirmed Cases')
    plt.legend(['Actual', 'Model'])
    plt.show()
plot_predictions_linear(adjusted_dates, df2["Cases"], linear_pred, 'Polynomial Regression Predictions', 'orange' )