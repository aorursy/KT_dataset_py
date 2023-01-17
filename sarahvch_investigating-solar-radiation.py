import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data_path = "../input/"

df = pd.read_csv(data_path+"SolarPrediction.csv")

df.head()
corrmat = df.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
g = sns.jointplot(x="Radiation", y="Temperature", data=df)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Temp vs. Radiation')
#drop low radiation values

df = df[df['Radiation'] >= 10]
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
ax = plt.axes()

sns.barplot(x="hour", y='Radiation', data=df, palette="BuPu", ax = ax)

ax.set_title('Mean Radiation by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Radiation', data=df, palette="BuPu", ax = ax, order=[9,10,11,12,1])

ax.set_title('Mean Radiation by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='Humidity', data=df, palette=("coolwarm"), ax = ax)

ax.set_title('Humidity by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="total_time", y='Radiation', data=df, palette="BuPu", ax = ax)

ax.set_title('Radiation by Total Daylight Hours')

plt.show()
y = df['Radiation']

X = df.drop(['Radiation', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet','Time_conv',], axis=1)
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
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients