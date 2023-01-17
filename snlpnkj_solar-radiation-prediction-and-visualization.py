import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

% matplotlib inline
data_path = "../input/"

df = pd.read_csv(data_path+"SolarPrediction.csv")
df.head()
df.tail()
print ("min radiation: ", df['Radiation'].min()," max radiation: ",  df['Radiation'].max())
df.describe()
#Covert time to_datetime

#Add column 'hour'

df['Time_conv'] =  pd.to_datetime(df['Time'], format='%H:%M:%S')

df['hour'] = pd.to_datetime(df['Time_conv'], format='%H:%M:%S').dt.hour



#Add column 'month'

df['month'] = pd.to_datetime(df['UNIXTime'].astype(int), unit='s').dt.month



#Add column 'year'

df['year'] = pd.to_datetime(df['UNIXTime'].astype(int), unit='s').dt.year



#Duration of Day

df['total_time'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S').dt.hour - pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S').dt.hour

df.head()
df.describe()
ax = plt.axes()

sns.barplot(x="hour", y='Radiation', data=df, palette="BuPu", ax = ax)

ax.set_title('Mean Radiation by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Radiation', data=df, palette="BuPu", ax = ax, order=[9,10,11,12,1])

ax.set_title('Mean Radiation by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='Temperature', data=df, palette=("coolwarm"), ax = ax)

ax.set_title('Mean Temperature by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Temperature', data=df, palette=("coolwarm"), ax = ax,order=[9,10,11,12,1])

ax.set_title('Mean Temperature by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='Humidity', data=df, palette=("coolwarm"), ax = ax)

ax.set_title('Mean Humidity by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Humidity', data=df, palette=("coolwarm"), ax = ax,order=[9,10,11,12,1])

ax.set_title('Mean Humidity by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='WindDirection(Degrees)', data=df, palette=("coolwarm"),)

ax.set_title('Mean WindDirection(Degrees) by Hour')

plt.show()

ax = plt.axes()

sns.barplot(x="month", y='WindDirection(Degrees)', data=df, palette=("coolwarm"), ax = ax,order=[9,10,11,12,1])

ax.set_title('Mean WindDirection(Degrees) by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='Speed', data=df, palette=("coolwarm"), ax = ax)

ax.set_title('Mean Speed by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Speed', data=df, palette=("coolwarm"), ax = ax,order=[9,10,11,12,1])

ax.set_title('Mean Speed by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='Pressure', data=df, palette=("coolwarm"), ax = ax)

ax.set_title('Mean Pressure by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Pressure', data=df, palette=("coolwarm"), ax = ax,order=[9,10,11,12,1])

ax.set_title('Mean Pressure by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="total_time", y='Radiation', data=df, palette="BuPu", ax = ax)

ax.set_title('Radiation by Total Hours')

plt.show()
y = df['Radiation']

X = df.drop(['Radiation', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet','Time_conv',], axis=1)
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))