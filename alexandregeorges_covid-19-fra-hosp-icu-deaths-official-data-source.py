import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

plt.style.use('fivethirtyeight')
# Open data french gvt

url = 'https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7'

df = pd.read_csv(url, sep= ";")

df.head()
df.info()
# sex = 0 => total M + W

# Using groupby

df_group = df[df['sexe'] == 0].groupby('jour')['hosp','rea','dc'].sum()



df_group
days_in_future = 10

future_forcast = np.array([i for i in range(len(df['jour'].unique())+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
start_date = datetime.date(2020, 3, 18)

dates_cases = [start_date + datetime.timedelta(days=x) for x in range(len(df['jour'].unique()))]

forecast_dates = [start_date + datetime.timedelta(days=x) for x in range(len(df['jour'].unique()) + days_in_future)]
plt.figure(figsize=(16, 9))

plt.plot(dates_cases, df_group['hosp'])

plt.title('# of Hosp', size=30)

plt.xlabel('Dates', size=25)

plt.ylabel('# of Cases', size=30)





import matplotlib.dates as mdates



plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))

plt.gca().xaxis.set_major_locator(mdates.DayLocator())



plt.gca().get_yaxis().set_ticks([i for i in range(max(df_group['hosp'])) if i % 2000 == 0])

plt.xticks(size=8)

plt.yticks(size=10)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(dates_cases, df_group['rea'])

plt.title('# of ICU', size=30)

plt.xlabel('Dates', size=25)

plt.ylabel('# of Cases', size=30)





import matplotlib.dates as mdates



plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))

plt.gca().xaxis.set_major_locator(mdates.DayLocator())



plt.gca().get_yaxis().set_ticks([i for i in range(max(df_group['rea'])) if i % 1000 == 0])

plt.xticks(size=8)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(dates_cases, df_group['dc'])



plt.title('# of Deaths', size=30)

plt.xlabel('Dates', size=25)

plt.ylabel('# of Cases', size=30)



plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))

plt.gca().xaxis.set_major_locator(mdates.DayLocator())



plt.gca().get_yaxis().set_ticks([i for i in range(max(df_group['dc'])) if i % 500 == 0])

plt.xticks(size=11)

plt.yticks(size=20)

plt.show()



import operator



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
# training (deaths)

y = df_group.iloc[:,2].values

X = np.array([i for i in range(len(df['jour'].unique()))]).reshape(-1, 1)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree= 4)

X_poly = poly_reg.fit_transform(X)



lin_reg = LinearRegression()

lin_reg.fit(X_poly, y)
plt.scatter(X,y,color =  'red')

lines = plt.plot(X, lin_reg.predict(X_poly))

plt.setp(lines, linewidth=2,linestyle='--')
plt.figure(figsize=(16, 9))

plt.scatter(X,y,color =  'red')

lines = plt.plot(future_forcast, lin_reg.predict(poly_reg.fit_transform(future_forcast)))

plt.setp(lines, linewidth=2,linestyle='--')



plt.title('# of Deaths', size=30)

plt.xlabel('Days Since 3/18/2020', size=30)

plt.ylabel('# of Cases', size=30)

# axis

plt.gca().get_xaxis().set_ticks([i for i in range(len(future_forcast)) if i % 1 == 0])



plt.xticks(size=10)

plt.yticks(size=20)

plt.show()
max(df_group['dc'])
pd.DataFrame(future_forcast, lin_reg.predict(poly_reg.fit_transform(future_forcast)))

# training (ICU)

y = df_group.iloc[:,1].values

X = np.array([i for i in range(len(df['jour'].unique()))]).reshape(-1, 1)

poly_reg = PolynomialFeatures(degree= 2)

X_poly = poly_reg.fit_transform(X)



lin_reg = LinearRegression()

lin_reg.fit(X_poly, y)



plt.scatter(X,y,color =  'red')

lines = plt.plot(X, lin_reg.predict(X_poly))

plt.setp(lines, linewidth=2,linestyle='--')
plt.figure(figsize=(16, 9))

plt.scatter(X,y,color =  'red')

lines = plt.plot(future_forcast, lin_reg.predict(poly_reg.fit_transform(future_forcast)))

plt.setp(lines, linewidth=2,linestyle='--')



plt.title('# of ICU', size=30)

plt.xlabel('Days Since 3/18/2020', size=30)

plt.ylabel('# of Cases', size=30)

# axis

plt.gca().get_xaxis().set_ticks([i for i in range(len(future_forcast)) if i % 1 == 0])



plt.xticks(size=20)

plt.yticks(size=10)

plt.show()
pd.DataFrame(future_forcast, lin_reg.predict(poly_reg.fit_transform(future_forcast)))