import numpy as np

import pandas as pd

import datetime



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error



import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd



raw_df = pd.read_csv("/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")



temp = raw_df['Measurement date'].values 

temp2 = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M") for x in temp]

raw_df['Measurement date'] = temp2



del temp, temp2



raw_df.describe()
meas_item_info = pd.read_csv("/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv")



by_time = raw_df.groupby('Measurement date').mean()

items = list(meas_item_info['Item name'].values)



dt = np.diff(by_time.index) / np.timedelta64(1,'h')



drifts = []

vols = []

for item in items:

    drift = np.mean(np.diff(by_time[item]) / dt)

    vol = np.std(np.diff(by_time[item]) / dt)



    drifts.append(drift)

    vols.append(vol)



dynamics_df = pd.DataFrame({

    'Item': items,

    'Drift': drifts,

    'Volatility': vols

})



dynamics_df
df_focussed = raw_df[['Measurement date', 'Station code', 'Latitude', 'Longitude'] + items]



mean_by_stn = df_focussed.groupby('Station code').mean()



# Load the shapefile (sourced from: https://github.com/southkorea/seoul-maps, credit: Lucy Park, http://lucypark.kr)

seoul_map = gpd.read_file("/kaggle/input/seoul-neighborhoods/kostat/2013/shp/seoul_municipalities.shx")



# Für alle Schadstoffe zeigen wir die Bewertung in jedem Gebiet

fig, ax = plt.subplots(3,2, figsize=(15,15))

items_arr = np.array(items).reshape(3,2)

sns.set()

for i in range(3):

    for j in range(2):

        item = items_arr[i,j]

        seoul_map.plot(ax=ax[i,j])

        sns.scatterplot(x="Longitude", y="Latitude", hue=item, data=mean_by_stn, ax=ax[i,j])

        ax[i,j].set_title("Concentration of " + item)



fig.show()

del items_arr
items1 = ['SO2', 'NO2', 'O3']

items2 = ['PM10', 'PM2.5']

items3 = 'CO'



temp = by_time[items1]

temp['Measurement date'] = temp.index

t_melted = temp.melt('Measurement date', var_name="Item", value_name='Measurement')



fig, ax = plt.subplots(3,1, figsize=(10,30))

sns.lineplot(x='Measurement date', y='Measurement', hue='Item', data=t_melted, ax=ax[0])

ax[0].set_title("Timeseries of SO2, NO2, CO and O3")



temp = by_time[items2]

temp['Measurement date'] = temp.index

t_melted = temp.melt('Measurement date', var_name="Item", value_name='Measurement')



sns.lineplot(x='Measurement date', y='Measurement', hue='Item', data=t_melted, ax=ax[1])

ax[1].set_title("Timeseries of PM10 and PM2.5")



temp = by_time[[items3]]

temp['Measurement date'] = temp.index



sns.lineplot(x=temp.index, y=temp[items3], data=temp, ax=ax[2])

ax[2].set_ylabel("Measurement")

ax[2].set_title("Timeseries of CO")

fig.show()
cor_mat = np.corrcoef(by_time[items], rowvar=False)

sns.heatmap(cor_mat, cmap = sns.light_palette("green"), yticklabels=items, xticklabels=items, linewidths=0.1)

plt.title("Pollutant concentration correlation")

plt.show()
def smoothed_series(x_series, window_s):

    windows = x_series.rolling(window_s)

    moving_avgs = windows.mean()

    moving_avg_list = moving_avgs.tolist()

    no_nans = moving_avg_list[window_s-1:]

    return pd.Series(no_nans), window_s-1



window_size = 24*7 # Smooth by weeks



# Extract ozone data and smooth

O3_means = by_time[by_time['O3']>0]['O3']

tO3 = O3_means.index

O3_smoothed, O3_tCut = smoothed_series(O3_means, window_size)



O3_smoothed = pd.DataFrame({

    'Time': tO3[O3_tCut:],

    'O3_Measurements': O3_smoothed

})



# Extract NO2 data and smooth

NO2_means = by_time[by_time['NO2']>0]['NO2']

tNO2 = NO2_means.index

NO2_smoothed, NO2_tCut = smoothed_series(NO2_means, window_size)



NO2_smoothed = pd.DataFrame({

    'Time': tNO2[NO2_tCut:],

    'NO2_Measurements': NO2_smoothed

})



# Extract SO2 data and smooth

SO2_means = by_time[by_time['SO2']>0]['SO2']

tSO2 = SO2_means.index

SO2_smoothed, SO2_tCut = smoothed_series(SO2_means, window_size)



SO2_smoothed = pd.DataFrame({

    'Time': tSO2[SO2_tCut:],

    'SO2_Measurements': SO2_smoothed

})



# Combine pairings by inner joins

O3NO2 = pd.merge(left=O3_smoothed, right=NO2_smoothed, how="inner", left_on="Time", right_on="Time")

O3SO2 = pd.merge(left=O3_smoothed, right=SO2_smoothed, how="inner", left_on="Time", right_on="Time")
fig,ax = plt.subplots(figsize=(10,10))

sns.lineplot(x="Time", y="O3_Measurements", data=O3NO2, ax=ax)

sns.lineplot(x="Time", y="NO2_Measurements", data=O3NO2, ax=ax)

ax.set_ylabel("Measurements")

ax.legend(['O3', 'NO2'])

fig.show()
model_O3NO2 = LinearRegression()



X = O3NO2['NO2_Measurements'].values.reshape(-1,1)

Y = np.log(O3NO2['O3_Measurements'].values).reshape(-1,1)



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model_O3NO2.fit(x_train, y_train)



y_pred = model_O3NO2.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean squared error:", format(mse, '.2f'))

print("R2 score:", format(r2_score(y_test, y_pred), '.2f'))



# Plot

fitY = np.exp(model_O3NO2.predict(X))



fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(X,np.exp(Y),s=5) 

sns.lineplot(x=X[:,0], y=fitY[:,0], ax=ax, color="red")

ax.set_ylabel("O3 measurements")

ax.set_xlabel("NO2 measurements")

ax.set_title("Regression of O3 and NO2 measurements")

fig.show()
fig,ax = plt.subplots(figsize=(10,10))

sns.lineplot(x="Time", y="O3_Measurements", data=O3SO2, ax=ax)

sns.lineplot(x="Time", y="SO2_Measurements", data=O3SO2, ax=ax)

ax.set_ylabel("Measurements")

ax.legend(['O3', 'SO2'])

fig.show()
model_O3SO2 = LinearRegression()



X = O3SO2['SO2_Measurements'].values.reshape(-1,1)

Y = np.log(O3SO2['O3_Measurements'].values).reshape(-1,1)



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model_O3SO2.fit(x_train, y_train)



y_pred = model_O3SO2.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean squared error:", format(mse, '.2f'))

print("R2 score:", format(r2_score(y_test, y_pred), '.2f'))



# Plot

fitY = np.exp(model_O3SO2.predict(X))



fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(X,np.exp(Y),s=5) 

sns.lineplot(x=X[:,0], y=fitY[:,0], ax=ax, color="red")

ax.set_ylabel("O3 measurements")

ax.set_xlabel("SO2 measurements")

ax.set_title("Regression of O3 and SO2 measurements")

fig.show()