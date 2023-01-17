# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import os

from textwrap import wrap



# Set default fontsize and colors for graphs

SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 10, 12, 20

plt.rc('font', size=MEDIUM_SIZE)       

plt.rc('axes', titlesize=BIG_SIZE)     

plt.rc('axes', labelsize=MEDIUM_SIZE)  

plt.rc('xtick', labelsize=MEDIUM_SIZE) 

plt.rc('ytick', labelsize=MEDIUM_SIZE) 

plt.rc('legend', fontsize=SMALL_SIZE)  

plt.rc('figure', titlesize=BIG_SIZE)

my_colors = 'rgbkymc'



# Disable scrolling for long output

from IPython.display import display, Javascript

disable_js = """

IPython.OutputArea.prototype._should_scroll = function(lines) {

    return false;

}

"""

display(Javascript(disable_js))



print(os.listdir('../input'))
# Read the input training and test data

input_data_dir = "../input/bangalore-accident-data"

input_data_file = "bangalore-cas-alerts.csv"

input_data_path = os.path.join(input_data_dir, input_data_file)

train_data = pd.read_csv(input_data_path)

train_data.head(10)
train_data.shape
# Remove duplicates and check dataset size

train_data.drop_duplicates(inplace=True)

train_data.shape
train_data.info()
train_data.describe().transpose()
# Renaming the columns

columns={

            "deviceCode_deviceCode" : "deviceCode",

            "deviceCode_location_latitude" : "latitude",

            "deviceCode_location_longitude" : "longitude",

            "deviceCode_location_wardName" : "wardName",

            "deviceCode_pyld_alarmType" : "alarmType",

            "deviceCode_pyld_speed" : "speed",

            "deviceCode_time_recordedTime_$date" : "recordedDateTime"

        }



train_data.rename(columns=columns, inplace=True)

print("Updated column names of train dataframe:", train_data.columns)
lat_max = train_data.latitude.max()

lat_min = train_data.latitude.min()

print("Range of latitude:", lat_max, lat_min)



lon_max = train_data.longitude.max()

lon_min = train_data.longitude.min()

print("Range of longitude:", lon_max, lon_min)
fig, axes = plt.subplots(figsize=(9,12))

axes.scatter(train_data.longitude, train_data.latitude, s=0.1, alpha=0.5, c='r')

plt.show()
# Scatter plot with map

#bangalore_map_img = "bangalore_map.png"

#bangalore_map = plt.imread(os.path.join(input_data_dir, bangalore_map_img))



bangalore_map_img = 'https://lh3.googleusercontent.com/np8igtYRrHpe7rvJwMzVhbyUZC4Npgx5fRznofRoLVhP6zcdBW9tfD5bC4FbF2ITctElCtBrOn7VH_qEBZMVoPrTFipBdodufT0QU1NeeQVyokMAKtvSHS9BfYMswXodz_IrkiZStg=w500-h664-no'

bangalore_map = plt.imread(bangalore_map_img)

cmap = plt.get_cmap("jet")



axes = train_data.plot(figsize=(15,20), kind='scatter', 

                    x='longitude', y='latitude', 

                    alpha=0.5, marker="o", s=train_data["speed"]*2,

                    c=train_data["speed"], cmap=cmap,

                    colorbar=False)



epsilon = 0.01

bound_box = [lon_min + epsilon, lon_max + epsilon, 

             lat_min + epsilon, lat_max + epsilon]

im = plt.imshow(bangalore_map, extent=bound_box, zorder=0, 

           cmap=cmap, interpolation='nearest', alpha=0.7)



axes.set_ylabel("Latitude")

axes.set_xlabel("Longitude")

axes.set_title('Accident Heatmap of city of Bangalore')



# Colorbar

speed = train_data["speed"].values

tick_values = np.linspace(speed.min(), speed.max(), num=6, dtype=np.int64)



cbar = plt.colorbar(im, fraction=0.05, pad=0.02)

cbar.set_label("Speed (km / hour)")

cbar.ax.set_yticklabels(["%d"%(val) for val in tick_values])



plt.tight_layout()



#output_image = os.path.join(input_data_dir, "output_bangalore_map_traffic")

#plt.savefig(output_image + ".png", format='png', dpi=300)



plt.show()
unique_data = train_data['deviceCode'].unique()

print("Number of unique devices used in buses =", len(unique_data))
# Capitalize ward name

train_data['wardName'] = train_data['wardName'].str.capitalize()

print("Total number of wards in CDS dataset:", len(train_data['wardName'].unique()))
fig, axes = plt.subplots(figsize=(12,6))

data = train_data['wardName'].value_counts(normalize=True).sort_values(ascending=False)

data = data.head(10)

axes.bar(data.index, data*100, color=my_colors)

axes.set_ylabel("% of CDS alarms")

axes.set_xticklabels(data.index, rotation=90)

axes.set_title("% of incidents in each ward")

plt.show()
# Read data file

input_data_dir = '../input/bbmpwarddata'

input_data_file = "bbmp-wards.csv"

input_data_path = os.path.join(input_data_dir, input_data_file)

bbmp_data = pd.read_csv(input_data_path)

bbmp_data.sample(5)
# Capitalize ward name

bbmp_data['WARD_NAME'] = bbmp_data['WARD_NAME'].str.capitalize()

print("Total number of wards in BBMP dataset:", len(bbmp_data['WARD_NAME'].unique()))



# Create a dict mapping of ward number and names

ward_numbers = bbmp_data['WARD_NO'].unique()

ward_names = bbmp_data['WARD_NAME'].unique()

ward_dict = dict(zip(ward_numbers, ward_names))
# Apply KNN classification

# Input: Lat, Lon, Output: Ward No (since this is numeric variable)

from sklearn.neighbors import KNeighborsClassifier



knn_clf = KNeighborsClassifier(n_neighbors=1, weights='distance')

knn_clf.fit(bbmp_data[['LAT', 'LON']], bbmp_data['WARD_NO'])



# Estimate ward no in train_data from learnt model of bbmp_data

train_data['estimatedWardNo'] = knn_clf.predict(train_data[['latitude', 'longitude']])

# Estimate ward name in train_data from the ward no - ward name mapping dictionary

train_data['estimatedWardName'] = train_data['estimatedWardNo'].map(ward_dict)
# Check accuracy

validation_orig = train_data['wardName'][~train_data['wardName'].str.contains('Other')]

validation_estimated = train_data['estimatedWardName'][~train_data['wardName'].str.contains('Other')]

accuracy = np.mean(validation_orig == validation_estimated)

print("Out of ward names which are not 'Other', % of accurate predictions =", accuracy*100)
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

data = train_data['estimatedWardName'].value_counts(normalize=True).sort_values(ascending=False)

data1 = data.head(10)



axes1.bar(data1.index, data1*100, color=my_colors)

axes1.set_ylabel("% of CDS alarms")

axes1.set_xticklabels(data1.index, rotation=90)

axes1.set_title("Most dangerous wards")



data2 = data.tail(10)

axes2.bar(data2.index, data2*100, color=my_colors)

axes2.set_ylabel("% of CDS alarms")

axes2.set_xticklabels(data2.index, rotation=90)

axes2.set_title("Most safe wards")



plt.show()
fig, axes = plt.subplots(figsize=(12,6))

data = train_data['alarmType'].value_counts(normalize=True)

axes.bar(data.index, data*100, color=my_colors)

axes.set_title('% of Alarm Types')

axes.set_xlabel('')

axes.set_ylabel('%')

plt.show()
alarm_mapping = {"UFCW":"LSCW", "HMW":"HSCW", "FCW":"HSCW", "LDWL":"LDW", "LDWR":"LDW", "Overspeed":"Overspeed", "PCW":"PCW"}

train_data['alarmTypeCat'] = train_data['alarmType'].map(alarm_mapping)
fig, axes = plt.subplots(figsize=(12,6))

data = train_data['alarmTypeCat'].value_counts(normalize=True)

axes.bar(data.index, data*100, color=my_colors)

axes.set_title('% of Alarm Types (updated categorization)')

axes.set_xlabel('')

axes.set_ylabel('%')

plt.show()
fig, axes = plt.subplots(figsize=(12,6))

data = train_data['speed']

#axes.hist(data, bins=15, color='green')

sns.distplot(data, bins=15, color='green')

axes.axvline(data.mean(), color='k', linestyle='dashed', linewidth=2)



axes.set_xticks(np.arange(0, data.max()+5, 5))

axes.set_xticklabels([str(val) for val in np.arange(0, data.max()+5, 5)])

axes.set_xlim(0, data.max()+5)

axes.set_xlabel('Speed in kmph')

axes.set_title('Distribution of Speed')

axes.grid(True)



_ymin, _ymax = axes.get_ylim()

axes.text(data.mean() + data.mean()/100,

          (_ymax+_ymin)*0.5,

          'Mean Speed: {:.2f} kmph'.format(data.mean()))



plt.show()
train_data.recordedDateTime = train_data.recordedDateTime.map(lambda x : pd.Timestamp(x, tz='Asia/Kolkata'))
print("Alarm data are spanning across:")

print("Years: ", train_data.recordedDateTime.dt.year.unique())

print("Months: ", train_data.recordedDateTime.dt.month_name().unique())

print("Dates: ", train_data.recordedDateTime.dt.day.unique())
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6), sharey=True)

train_data["monthName"] = train_data.recordedDateTime.dt.month_name()

data = train_data["monthName"].value_counts(normalize=True)



axes1.bar(data.index, data*100, color=my_colors)

axes1.set_xticklabels(data.index, rotation=90)

axes1.set_ylabel('Percentage')

axes1.set_title('% of alarms by Month')



train_data["dayName"] = train_data.recordedDateTime.dt.day_name()

data = train_data["dayName"].value_counts(normalize=True)

axes2.bar(data.index, data*100, color=my_colors)

axes2.set_xticklabels(data.index, rotation=90)

axes2.set_title('% of alarms by Days of Week')



plt.show()
fig, axes = plt.subplots(figsize=(15,6))

train_data["dayOfMonth"] = train_data.recordedDateTime.dt.day

data = train_data["dayOfMonth"].value_counts(normalize=True).sort_index()

axes.bar(data.index, data, color='pink', width=0.5, zorder=0)

axes.plot(data.index, data, color='red', zorder=1)

axes.scatter(data.index, data, s=50, color='darkred', zorder=2)



axes.set_xlabel('Date of Month')

axes.set_xticks(np.arange(1, 32))

axes.set_xticklabels([str(val) for val in np.arange(1, 32)])

axes.set_ylim(0, 0.1)

axes.set_title('% of alarms by Date of Month')



plt.show()
fig, axes = plt.subplots(figsize=(15,6))

train_data["hour"] = train_data.recordedDateTime.dt.hour

data = train_data["hour"].value_counts(normalize=True).sort_index()



axes.bar(data.index, data, color='tan', width=0.5, zorder=0)

axes.plot(data.index, data, color='red', zorder=1)

axes.scatter(data.index, data, s=50, color='darkred', zorder=2)



axes.set_xlabel('Hour of Day')

axes.set_xticks(np.arange(1, 25))

axes.set_xticklabels([str(val) for val in np.arange(1, 25)])

axes.set_ylim(0, 0.3)

axes.set_title('% of alarms by Hours of Day')



plt.show()
# Seggregate data by alarm types

data_lscw = train_data[train_data.alarmTypeCat == 'LSCW']

data_hscw = train_data[train_data.alarmTypeCat == 'HSCW']

data_speed = train_data[train_data.alarmTypeCat == 'Overspeed']

data_pcw = train_data[train_data.alarmTypeCat.str.contains('PCW', 'LDW')]
fig, axes = plt.subplots(2, 2, figsize=(15,20), sharex=True, sharey=True)

cmap = plt.get_cmap("jet")



# Plot LSCW

axes[0][0].scatter(data_lscw.longitude, data_lscw.latitude,

                   alpha=0.5, marker="o", s=10,

                   c=data_lscw.hour, cmap=cmap, zorder=1)

axes[0][0].set_title("Low Speed Collisions")



# Plot HSCW

axes[0][1].scatter(data_hscw.longitude, data_hscw.latitude,

                   alpha=0.5, marker="o", s=10,

                   c=data_hscw.hour, cmap=cmap, zorder=1)

axes[0][1].set_title("High Speed Collisions")



# Plot Overspeed

axes[1][0].scatter(data_speed.longitude, data_speed.latitude,

                   alpha=0.5, marker="o", s=10,

                   c=data_speed.hour, cmap=cmap, zorder=1)

axes[1][0].set_title("Over Speeding")



# Plot PCW and LDW

axes[1][1].scatter(data_pcw.longitude, data_pcw.latitude,

                   alpha=0.5, marker="o", s=10,

                   c=data_pcw.hour, cmap=cmap, zorder=1)

axes[1][1].set_title("\n".join(wrap("Lane Departure without indicator, and Collisions with Pedestrians and Cyclists", 30)))



# Plot Bangalore map image

epsilon = 0.01

bound_box = [lon_min + epsilon, lon_max + epsilon, 

             lat_min + epsilon, lat_max + epsilon]



for ax in axes.flat:

    im = ax.imshow(bangalore_map, extent=bound_box,

                      alpha=0.5, zorder=0, cmap=cmap)



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(2, 2, figsize=(20,10), sharey=True)



# LSCW

sns.kdeplot(data_lscw.speed, color='green', shade=True, ax=axes[0][0])

axes[0][0].axvline(data_lscw.speed.mean(), color='green', linestyle='dashed', linewidth=2)

axes[0][0].set_title('Low Speed Collisions')



# LSCW

sns.kdeplot(data_hscw.speed, color='red', shade=True, ax=axes[0][1])

axes[0][1].axvline(data_hscw.speed.mean(), color='red', linestyle='dashed', linewidth=2)

axes[0][1].set_title('High Speed Collisions')



# LSCW

sns.kdeplot(data_speed.speed, color='darkorange', shade=True, ax=axes[1][0])

axes[1][0].axvline(data_speed.speed.mean(), color='darkorange', linestyle='dashed', linewidth=2)

axes[1][0].set_title('Over Speeding')



# LSCW

sns.kdeplot(data_pcw.speed, color='magenta', shade=True, ax=axes[1][1])

axes[1][1].axvline(data_pcw.speed.mean(), color='magenta', linestyle='dashed', linewidth=2)

axes[1][1].set_title('Lane Departure and Pedestrian Collision')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(figsize=(18,8))

data = train_data['estimatedWardName'].value_counts(normalize=True).sort_values(ascending=False)

data = data.head(10)

ward_top = data.index



ward_top_data = train_data[train_data.estimatedWardName.isin(ward_top)]

sns.countplot(x='estimatedWardName', hue='alarmTypeCat', data=ward_top_data, ax=axes)



axes.legend(title='Alarm Type')

axes.set_xlabel('')

axes.set_ylabel('# of Incidents')

axes.set_title('What\'s going on in most dangerous wards?')



plt.show()
fig, axes = plt.subplots(figsize=(15,15))

cmap = plt.get_cmap("jet")



data = bbmp_data[bbmp_data.WARD_NAME.isin(ward_top)]



axes.scatter(data.LON, data.LAT, marker="v", s=300,

                   c='red', zorder=1)

axes.set_title("Most dangerous wards for traffic")



# Plot Bangalore map image

epsilon = 0.01

bound_box = [lon_min + epsilon, lon_max + epsilon, 

             lat_min + epsilon, lat_max + epsilon]



axes.imshow(bangalore_map, extent=bound_box, alpha=0.7, zorder=0)



# Add names of wards as text

for _idx, _ward_data in data.iterrows():

    axes.text(_ward_data.LON  - epsilon/2, _ward_data.LAT + epsilon/2, 

              _ward_data.WARD_NAME, color='blue', fontsize=11)



axes.set_yticklabels([])

axes.set_xticklabels([])



plt.tight_layout()

plt.show()