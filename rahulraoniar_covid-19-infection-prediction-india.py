import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
India_covid19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv', parse_dates=['Date'], dayfirst=True)

print(India_covid19.head())

print(India_covid19.tail())
India_covid19.columns
India_covid19.drop(["Sno"], axis = 1, inplace = True)

India_covid19.rename(columns = {"State/UnionTerritiry": "States"}, inplace=True)

print(India_covid19.head())

print(India_covid19.tail())
India_per_day = India_covid19.groupby(["Date"])["Confirmed"].sum().reset_index().sort_values("Date", ascending = True)

print(India_per_day.head())

print(India_per_day.tail())

print(India_per_day.shape)
India_per_day.shape[0]
India_per_day['Date']=pd.to_datetime(India_per_day.Date,dayfirst=True)

India_daily= India_per_day.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Date',ascending=True)

India_daily["day_count"] = np.arange(0, India_daily.shape[0])



daily_infection = India_daily.loc[:, ["day_count", "Confirmed"]]

print(daily_infection.head())

print(daily_infection.tail())
plt.scatter(daily_infection["day_count"], daily_infection["Confirmed"], alpha=0.3, c="red")

plt.plot(daily_infection["day_count"], daily_infection["Confirmed"])

plt.title("Daily Infection Plot")

plt.xlabel("Day")

plt.ylabel("Infections")

plt.show()
# Taking log of dependent variable

daily_infection["logConfirmed"] = np.log(daily_infection.Confirmed)

daily_infection.head(4)
plt.scatter(daily_infection["day_count"], daily_infection["logConfirmed"], alpha=0.3, c="red")

plt.plot(daily_infection["day_count"], daily_infection["logConfirmed"])

plt.title("Daily Infection Plot")

plt.xlabel("Day")

plt.ylabel("log_Infections")

plt.show()
import statsmodels.api as sm



X = daily_infection.day_count

X = sm.add_constant(X)

y = daily_infection.logConfirmed
model = sm.OLS(y, X)

reg = model.fit()

print(reg.summary())
x0 = np.exp(reg.params[0])

b = np.exp(reg.params[1])

x0, b
t1 = np.arange(India_daily.shape[0])

y = (x0 + b**t1).round()

y
upto_now = pd.DataFrame({'day_count': t1, "Actual": daily_infection["Confirmed"], "Predicted": y, })

upto_now
plt.plot(upto_now.day_count, upto_now.Actual, alpha=0.4, c="green")

plt.plot(upto_now.day_count, upto_now.Predicted, alpha=0.4, c="red")



plt.title("Actual vs Predicted Plot")

plt.legend(["Actual Count", "Predicted Count"])

plt.xlabel("Day")

plt.ylabel("Infections")

plt.show()
from sklearn.metrics import mean_squared_error



mean_squared_error(upto_now.Actual, upto_now.Predicted, squared=False)
India_daily.shape[0] + 14
t = np.arange(India_daily.shape[0], India_daily.shape[0] + 14)

t
xt = (x0 + b**t).round()

xt
next2weeks = pd.DataFrame({'day_count': t, "Confirmed": xt})

next2weeks
X = daily_infection.day_count

y = daily_infection.Confirmed



X1 = next2weeks.day_count

y1 = next2weeks.Confirmed
plt.scatter(X, y, alpha=0.2, c="blue")

plt.scatter(X1, y1, alpha=0.1, c="blue")

plt.plot(X, y)

plt.plot(X1, y1)

plt.title("Future Infection Prediction")

plt.legend(["Up to the Present Day", "Next 14 Days Prediction"])

plt.xlabel("Day")

plt.ylabel("Infections")

plt.show()