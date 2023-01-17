# importing dataset

%matplotlib inline

import numpy as np

import pandas as pd

weather_df = pd.read_csv('../input/szeged-weather/weatherHistory.csv', encoding='utf-8')

weather_df.head(3)
datetime = pd.to_datetime(weather_df["Formatted Date"])

datetime = datetime.apply(lambda x: x+pd.Timedelta(hours=2)) #Correcting +2 GMT

weather_df["Month"] = datetime.apply(lambda x: x.month)

weather_df["WoY"] = datetime.apply(lambda x: x.week)

weather_df["Hour"] = datetime.apply(lambda x: x.hour)

weather_df["T"] = weather_df.index

weather_df[["Formatted Date","Month","WoY","Hour","T"]].head()
import seaborn as sns

sns.pairplot(weather_df[["Precip Type","Temperature (C)","Apparent Temperature (C)","Humidity","Hour","T"]],

             hue="Precip Type",

             palette="YlGnBu");
corr = weather_df.drop('Loud Cover', axis=1).corr() # dropping Loud Cover because it never change

sns.heatmap(corr,  cmap="YlGnBu", square=True);
sns.violinplot(x="Precip Type", y="Temperature (C)", data=weather_df, palette="YlGnBu");
sns.violinplot(x="Precip Type", y="Humidity", data=weather_df, palette="YlGnBu");
sns.jointplot("Humidity", "Temperature (C)", data=weather_df.where(weather_df['Precip Type']=='null'), kind="hex");
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score



ls = linear_model.LinearRegression()



# Our model will only predict temperature for non-raining/snowing configuration.

# I would recommend you to change 'null' for 'rain' or 'snow' and verify

# quality metrics, and see how efficient the models would be, only by filtering.

data = weather_df.where(weather_df['Precip Type']=='null')

data.dropna(inplace=True)



X = data["Humidity"].values.reshape(-1,1)

y = data["Temperature (C)"].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size=0.33, 

                                                    shuffle=True, random_state=0)

print("Linear Regression")

ls.fit(X_train, y_train)

print("alpha = ",ls.coef_[0])

print("beta = ",ls.intercept_)

print("\n\nCalculating some regression quality metrics, which we'll discuss further on next notebooks")

y_pred = ls.predict(X_test)

print("MSE = ",mean_squared_error(y_test, y_pred))

print("R2 = ",r2_score(y_test, y_pred))

hypothetical_humidity = 0.7

temperature_output = ls.predict(hypothetical_humidity)[0][0]

print("For such {} humidity, Linear Regression predict a temperature of {}C".format(hypothetical_humidity, round(temperature_output,1)))