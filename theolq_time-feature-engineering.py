import numpy as np

import pandas as pd



import os

os.chdir('/kaggle/input/SolarEnergy')



df = pd.read_csv("SolarPrediction.csv")

df.head()
Wind_Direction = df["WindDirection(Degrees)"]

df = df.drop(["UNIXTime", "WindDirection(Degrees)"], axis = 1)

df["Wind_Direction"] = Wind_Direction

df.head()
import re

import datetime

from datetime import date



day = [] ; month = [] ; year = []

for string in df["Data"]:

    match = re.search(r'\d+/\d+/\d+', string)

    date = datetime.datetime.strptime(match.group(), '%m/%d/%Y').date()

    day.append(date.day)

    month.append(date.month)

    year.append(date.year)



df["Day"] = day ; df["Month"] = month ; df["Year"] = year

df = df.drop("Data", axis = 1)

df.head()
print("Year : ", df.Year.value_counts())

print("Month : ", df.Month.value_counts())
from datetime import time



def time_conversion_to_second(time_to_convert):

    time_converted = datetime.datetime.strptime(time_to_convert, '%H:%M:%S').time()

    time_in_second = time_converted.hour * 3600 + time_converted.minute * 60 + time_converted.second

    return time_in_second

    

def sun_is_up(current_time, rising_time, set_time):

    current_time_second = time_conversion_to_second(current_time)

    rising_time_second = time_conversion_to_second(rising_time)

    set_time_second = time_conversion_to_second(set_time)

    

    return (rising_time_second < current_time_second) and (current_time_second < set_time_second)



Sun_is_up = [sun_is_up(df["Time"][index], df["TimeSunRise"][index], df["TimeSunSet"][index]) for index in range(df.shape[0])]

Sun_is_up = np.array(Sun_is_up, dtype = int)



df["Sun_is_up"] = Sun_is_up
proportion = round(sum(df["Sun_is_up"]/df.shape[0]*100), 2)

print("Proportion of record with the sun up : {0}%".format(proportion))
def soustract_time(time1, time2):

    hour = time1.hour - time2.hour

    if time1.minute > time2.minute:

        minute = (time1.minute - time2.minute) / 60

    else:

        hour -= 1

        minute = (time1.minute + 60 - time2.minute) / 60

    return hour + minute

    



def sun_time_count(current_time, rising_time):

    current_time_converted = datetime.datetime.strptime(current_time, '%H:%M:%S').time()

    rising_time_converted = datetime.datetime.strptime(rising_time, '%H:%M:%S').time()

    

    return soustract_time(current_time_converted, rising_time_converted)



Sun_hour_count = [df["Sun_is_up"][index] * sun_time_count(df["Time"][index], df["TimeSunRise"][index]) for index in range(df.shape[0])]

df["Sun_hour_count"] = Sun_hour_count
import matplotlib.pyplot as plt

import seaborn as sns ; sns.set()



data = df[df["Sun_is_up"] == 1]



fig = plt.figure(figsize=(20, 6))



plt.scatter(data["Sun_hour_count"], data["Radiation"])

plt.show()
def extract_time(current_time):

    current_time_converted = datetime.datetime.strptime(current_time, '%H:%M:%S').time()

    return current_time_converted.hour + current_time_converted.minute / 60



Hour = [extract_time(df["Time"][index]) for index in range(df.shape[0])]



df["Hour"] = Hour

fig = plt.figure(figsize=(20, 6))

plt.scatter(Hour, df["Radiation"])

plt.show()
def plot_radiation_profil(day, month):

    df_day = df[(df["Day"] == day) & (df["Month"] == month)]

    label = "{0}/{1}/2016".format(day, month)

    plt.plot(df_day["Hour"],df_day["Radiation"], label = label)
fig = plt.figure(figsize=(20, 6))

plot_radiation_profil(1, 10)

plot_radiation_profil(2, 10)

plot_radiation_profil(3, 10)



plt.title("Radiation in function of time")

plt.legend()

plt.show()
data = df



Rise_Hour = [extract_time(data["TimeSunRise"][index]) for index in range(data.shape[0])]

Set_Hour = [extract_time(data["TimeSunSet"][index]) for index in range(data.shape[0])]



data["Rise_Hour"] = Rise_Hour

data["Set_Hour"] = Set_Hour

data = data.drop(["Time", "TimeSunRise", "TimeSunSet"], axis = 1)

data.head()
from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

X = data.drop(["Radiation","Month","Day"], axis = 1)

y = data["Radiation"]

X_train, X_test, y_train, y_test = train_test_split(X, y)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)



MSE = mean_squared_error(y_true = y_test, y_pred = y_pred)

R2 = r2_score(y_true = y_test, y_pred = y_pred)



print("MSE : {0} and R2 : {1}".format(MSE, R2))
def see_prediction_error(number_to_see):

    Y = y_test.iloc[:number_to_see]

    X = [i for i in range(len(Y))]



    fig = plt.figure(figsize=(20, 10))



    for index in range(len(X)):

        plt.plot([X[index], X[index]], [Y.iloc[index], y_pred[index]], c="black")

    plt.plot(X, Y, 'o', label='True')

    plt.plot(X, y_pred[:number_to_see], 'o', label='Predicted')



    plt.title("Prediction vs real radiation values")

    plt.legend()

    plt.show()

    

number_to_see = 100

see_prediction_error(number_to_see)
for index in range(len(y_pred)):

    if y_pred[index]<0:

        y_pred[index] = 0



MSE = mean_squared_error(y_true = y_test, y_pred = y_pred)

R2 = r2_score(y_true = y_test, y_pred = y_pred)

print("MSE : {0} and R2 : {1}".format(MSE, R2))



see_prediction_error(number_to_see)
from datetime import time

from math import cos, sin, pi



def extract_and_convert_time_to_trigonometric(List):

    cos_time = [] ; sin_time = []

    period = 2 * pi / (3600 * 24)

    for time in List:

        converted_time = datetime.datetime.strptime(time, '%H:%M:%S').time()

        time_in_second = 3600 * converted_time.hour + 60 * converted_time.minute + converted_time.second

        cos_time.append(cos(time_in_second * period)) ; sin_time.append(sin(time_in_second * period))

    return cos_time, sin_time



def extract_and_replace(column_name, data, name_cos, name_sin):

    cos_time, sin_time = extract_and_convert_time_to_trigonometric(data[column_name])

    data[name_cos] = cos_time ; data[name_sin] = sin_time

    data = data.drop(column_name, axis = 1)

    return data



df = extract_and_replace(column_name = "Time", data = df, name_cos = "Cos_time", name_sin = "Sin_time")

df = extract_and_replace(column_name = "TimeSunRise", data = df, name_cos = "Cos_rise_time", name_sin = "Sin_rise_time")

df = extract_and_replace(column_name = "TimeSunSet", data = df, name_cos = "Cos_set_time", name_sin = "Sin_set_time")

df.head()  
from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

X = df.drop(["Radiation","Month","Day"], axis = 1)

y = df["Radiation"]

X_train, X_test, y_train, y_test = train_test_split(X, y)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)



MSE = mean_squared_error(y_true = y_test, y_pred = y_pred)

R2 = r2_score(y_true = y_test, y_pred = y_pred)



print("MSE : {0} and R2 : {1}".format(MSE, R2))
see_prediction_error(number_to_see)
for index in range(len(y_pred)):

    if y_pred[index]<0:

        y_pred[index] = 0



MSE = mean_squared_error(y_true = y_test, y_pred = y_pred)

R2 = r2_score(y_true = y_test, y_pred = y_pred)

print("MSE : {0} and R2 : {1}".format(MSE, R2))



see_prediction_error(number_to_see)