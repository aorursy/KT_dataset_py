import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.impute import SimpleImputer



%matplotlib inline
df = pd.read_csv("../input/sensor.csv")

print(df.shape)

df.head()
df.isnull().values.any()
print('Deletting rows:')

print('Shape before elimination:', df.shape)

df_flag = df.dropna(axis=0, how='any')

print('Shape after elimination:', df_flag.shape,'\n')



print('Deletting collumns:')

print('Shape before elimination:', df.shape)

df_flag = df.dropna(axis=1, how='any')

print('Shape after elimination:', df_flag.shape)
df_flag.head()
X = df.iloc[:,2:54].fillna(0)

normalizer_scaler = preprocessing.Normalizer(norm='max')

X = normalizer_scaler.fit_transform(X.transpose())

X = pd.DataFrame(X.transpose())



y = df['machine_status']



X.head()
one_hot = pd.get_dummies(y)

one_hot.head()
fig, axes = plt.subplots(figsize=(20, 20), dpi=120, nrows=7, ncols=2)



ax0 = X.iloc[::1500,0:4].plot(ax=axes[0,0])

ax0.set_xlim([0,220320])

ax0.grid()

ax0.set_xlabel('Time [minutes]')



ax1 = X.iloc[::1500,4:8].plot(ax=axes[0,1])

ax1.set_xlim([0,220320])

ax1.grid()

ax1.set_xlabel('Time [minutes]')



ax2 = X.iloc[::1500,8:12].plot(ax=axes[1,0])

ax2.set_xlim([0,220320])

ax2.grid()

ax2.set_xlabel('Time [minutes]')



ax3 = X.iloc[::1500,12:16].plot(ax=axes[1,1])

ax3.set_xlim([0,220320])

ax3.grid()

ax3.set_xlabel('Time [minutes]')



ax4 = X.iloc[::1500,16:20].plot(ax=axes[2,0])

ax4.set_xlim([0,220320])

ax4.grid()

ax4.set_xlabel('Time [minutes]')



ax5 = X.iloc[::1500,20:24].plot(ax=axes[2,1])

ax5.set_xlim([0,220320])

ax5.grid()

ax5.set_xlabel('Time [minutes]')



ax6 = X.iloc[::1500,24:28].plot(ax=axes[3,0])

ax6.set_xlim([0,220320])

ax6.grid()

ax6.set_xlabel('Time [minutes]')



ax7 = X.iloc[::1500,28:32].plot(ax=axes[3,1])

ax7.set_xlim([0,220320])

ax7.grid()

ax7.set_xlabel('Time [minutes]')



ax8 = X.iloc[::1500,32:36].plot(ax=axes[4,0])

ax8.set_xlim([0,220320])

ax8.grid()

ax8.set_xlabel('Time [minutes]')



ax9 = X.iloc[::1500,36:40].plot(ax=axes[4,1])

ax9.set_xlim([0,220320])

ax9.grid()

ax9.set_xlabel('Time [minutes]')



ax10 = X.iloc[::1500,40:44].plot(ax=axes[5,0])

ax10.set_xlim([0,220320])

ax10.grid()

ax10.set_xlabel('Time [minutes]')



ax11 = X.iloc[::1500,44:48].plot(ax=axes[5,1])

ax11.set_xlim([0,220320])

ax11.grid()

ax11.set_xlabel('Time [minutes]')



ax12 = X.iloc[::1500,48:52].plot(ax=axes[6,0])

ax12.set_xlim([0,220320])

ax12.grid()

ax12.set_xlabel('Time [minutes]')



plt.tight_layout()
# Instanciate selector

selector = SelectKBest(chi2, k=10) # select k = 11 < 51



# Fit it to data

X_fitted = selector.fit_transform(X, one_hot['NORMAL'])



# Determine k-best features

mask = selector.get_support() #list of booleans

new_features = [] # The list of your K best features



feature_names = list(X.columns.values)



for bool, feature in zip(mask, feature_names):

    if bool:

        new_features.append(feature)



X_fitted = X.iloc[:,new_features]

X_fitted.head()
fig, axes = plt.subplots(figsize=(20, 5), dpi=120, nrows=1, ncols=2)



ax0 = X_fitted.iloc[::1500,0:5].plot(ax=axes[0])

ax0.set_xlim([0,220320])

ax0.set_ylim([0,1])

ax0.grid()

ax0.set_xlabel('Time [minutes]')



ax1 = X_fitted.iloc[::1500,5:10].plot(ax=axes[1])

ax1.set_xlim([0,220320])

ax1.set_ylim([0,1])

ax1.grid()

ax1.set_xlabel('Time [minutes]')



plt.tight_layout
Y = one_hot['NORMAL']

data = pd.concat([X_fitted.iloc[:, 0:8],Y], axis=1, sort=False)



# Data visualisation imposing machine status information

fig, axes = plt.subplots(figsize=(20, 5), dpi=120, nrows=1, ncols=2)



ax0 = data.iloc[::1500, 0:4].plot(ax=axes[0])

ax0 = data.iloc[::1500, -1].plot(drawstyle="steps", ax=axes[0])

ax0.set_xlim([0,220320])

ax0.grid()

ax0.set_xlabel('Time [minutes]')



ax1 = data.iloc[::1500,4:8].plot(ax=axes[1])

ax1 = data.iloc[::1500, -1].plot(drawstyle="steps",ax=axes[1])

ax1.set_xlim([0,220320])

ax1.grid()

ax1.set_xlabel('Time [minutes]')



plt.tight_layout
# instanciate imputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')



new_data = imp.fit_transform(df.iloc[:,2:54])  # new_data originates from raw data

new_data = pd.DataFrame(new_data)
# Data visualisation imposing machine status information

fig, axes = plt.subplots(figsize=(20, 5), dpi=120, nrows=2, ncols=2)



ax0 = df.iloc[0:50,2].plot(ax=axes[0,0], legend = 'sensor_00')

ax0.set_xlim([0,50])

ax0.grid()

ax0.set_xlabel('Time [minutes]')



ax1 = df.iloc[0:50,6].plot(ax=axes[0,1], legend = 'sensor_04')

ax1.set_xlim([0,50])

ax1.grid()

ax1.set_xlabel('Time [minutes]')



ax2 = df.iloc[0:50,8].plot(ax=axes[1,0], legend = 'sensor_06')

ax2.set_xlim([0,50])

ax2.grid()

ax2.set_xlabel('Time [minutes]')



ax3 = df.iloc[0:50,9].plot(ax=axes[1,1], legend = 'sensor_07')

ax3.set_xlim([0,50])

ax3.grid()

ax3.set_xlabel('Time [minutes]')



plt.tight_layout
def interpol(X):

    X_interpolled = X.interpolate(method='linear')

    return X_interpolled
# new_data originates from raw data

new_data = df.iloc[:,2:54].astype(float)



# each independent signal corresponds to a given collumn

for i in range(0,52):

    feat = interpol(new_data.iloc[:,i])

    new_data.iloc[:,i] = feat



new_data.head()
new_data = new_data[['sensor_00','sensor_04','sensor_06','sensor_07','sensor_08','sensor_09','sensor_10','sensor_11']]

new_data.isnull().values.any()
#normalizer_scaler = preprocessing.Normalizer(norm='max')

new_data = normalizer_scaler.fit_transform(new_data.transpose())

new_data = pd.DataFrame(new_data.transpose())

new_data = pd.concat([new_data, Y], axis=1, sort=False)

new_data.columns = ['sensor_00','sensor_04','sensor_06','sensor_07','sensor_08','sensor_09','sensor_10','sensor_11','machine_status']

new_data.head()
new_data.to_csv("sensor_new_data.csv")
def find_gaps(numbers, gap_size):

    adjacent_differences = [(y - x) for (x, y) in zip(numbers[:-1], numbers[1:])]

    # If adjacent_differences[i] > gap_size, there is a gap of that size between

    # numbers[i] and numbers[i+1]. We return all such indexes in a list - so if

    # the result is [] (empty list), there are no gaps.

    return [i for (i, x) in enumerate(adjacent_differences) if x > gap_size]
null_ind = Y[Y == 0].index  # get null indexes

v = find_gaps(null_ind, 1)  # vector with null indexes gaps

v
# create gaps look-up table of the form (gap start, gap finish)

fail_ind = np.zeros((6,2)).astype(int)



for i in range(0,len(v)):

    if i == 0:

        fail_ind[i,0] = null_ind[0:v[i]][0]

        fail_ind[i,1] = null_ind[0:v[i]][-1]

    else:

        fail_ind[i,0] = null_ind[v[i-1]+1:v[i]][0]

        fail_ind[i,1] = null_ind[v[i-1]+1:v[i]][-1]



fail_ind
# each time-series will be chuncked into 6 new ones

def series_partitioning(feature, table):

    ts1 = feature.iloc[0:table[0][1]]

    ts2 = feature.iloc[table[0][1]:table[1][1]]

    ts3 = feature.iloc[table[1][1]:table[2][1]]

    ts4 = feature.iloc[table[2][1]:table[3][1]]

    ts5 = feature.iloc[table[3][1]:table[4][1]]

    ts6 = feature.iloc[table[4][1]:table[5][1]]

    

    return ts1,ts2,ts3,ts4,ts5,ts6
s00_1,s00_2,s00_3,s00_4,s00_5,s00_6 = series_partitioning(new_data['sensor_00'], fail_ind)
plt.figure(figsize=(20, 5), dpi=120)

plt.plot(s00_1.iloc[::1500], label ='s00_1')

plt.plot(s00_2.iloc[::1500], label ='s00_2')

plt.plot(s00_3.iloc[::1500], label ='s00_3')

plt.plot(s00_4.iloc[::1500], label ='s00_4')

plt.plot(s00_5.iloc[::1500], label ='s00_5')

plt.plot(s00_6.iloc[::1500], label ='s00_6')

Y.iloc[::1500].plot(drawstyle="steps")

plt.ylim(0,1)

plt.xlim(0,150000)

plt.legend()

plt.grid()

plt.tight_layout()
s04_1,s04_2,s04_3,s04_4,s04_5,s04_6 = series_partitioning(new_data['sensor_04'], fail_ind)

s06_1,s06_2,s06_3,s06_4,s06_5,s06_6 = series_partitioning(new_data['sensor_06'], fail_ind)

s07_1,s07_2,s07_3,s07_4,s07_5,s07_6 = series_partitioning(new_data['sensor_07'], fail_ind)

s08_1,s08_2,s08_3,s08_4,s08_5,s08_6 = series_partitioning(new_data['sensor_08'], fail_ind)

s09_1,s09_2,s09_3,s09_4,s09_5,s09_6 = series_partitioning(new_data['sensor_09'], fail_ind)

s10_1,s10_2,s10_3,s10_4,s10_5,s10_6 = series_partitioning(new_data['sensor_10'], fail_ind)

s11_1,s11_2,s11_3,s11_4,s11_5,s11_6 = series_partitioning(new_data['sensor_11'], fail_ind)

y_1,y_2,y_3,y_4,y_5,y_6 = series_partitioning(new_data['machine_status'], fail_ind)
sxx_1 = pd.concat([s00_1, s04_1, s06_1, s07_1, s08_1, s09_1, s10_1, s11_1, y_1], axis=1, sort=False)

sxx_1.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',

                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']

sxx_1.head()
sxx_2 = pd.concat([s00_2, s04_2, s06_2, s07_2, s08_2, s09_2, s10_2, s11_2, y_2], axis=1, sort=False)

sxx_2.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',

                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']



sxx_3 = pd.concat([s00_3, s04_3, s06_3, s07_3, s08_3, s09_3, s10_3, s11_3, y_3], axis=1, sort=False)

sxx_3.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',

                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']



sxx_4 = pd.concat([s00_4, s04_4, s06_4, s07_4, s08_4, s09_4, s10_4, s11_4, y_4], axis=1, sort=False)

sxx_4.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',

                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']



sxx_5 = pd.concat([s00_5, s04_5, s06_5, s07_5, s08_5, s09_5, s10_5, s11_5, y_5], axis=1, sort=False)

sxx_5.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',

                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']



sxx_6 = pd.concat([s00_6, s04_6, s06_6, s07_6, s08_6, s09_6, s10_6, s11_6, y_6], axis=1, sort=False)

sxx_6.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',

                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']
sxx_1.to_csv("sxx_1_data.csv")

sxx_2.to_csv("sxx_2_data.csv")

sxx_3.to_csv("sxx_3_data.csv")

sxx_4.to_csv("sxx_4_data.csv")

sxx_5.to_csv("sxx_5_data.csv")

sxx_6.to_csv("sxx_6_data.csv")