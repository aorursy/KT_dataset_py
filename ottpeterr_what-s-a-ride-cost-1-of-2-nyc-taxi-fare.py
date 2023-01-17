import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gc
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
# thanks to szelee for this quick loading method
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

n_rows = file_len(TRAIN_PATH)
# thanks to madhurisivalenka for this function
def add_haversine_distance_feature(df, lat1='pickup_latitude', long1='pickup_longitude', lat2='dropoff_latitude', long2='dropoff_longitude'):
    #R = 6371  # radius of earth in kilometers
    R = 3959 # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    #c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    #d = R*c
    d = (R * c)
    df["distance"] = d.astype('float32')
def clean_data(df, test=False):
    add_haversine_distance_feature(df)
    add_date_features(df, test)
    if not test:
        drop_conditional(df) 

MIN_FARE = 2.50
MAX_FARE = 150

MIN_PASSENGER = 1
MAX_PASSENGER = 6

LAT_MIN  = 40.55
LAT_MAX  = 40.95
LONG_MIN = -74.05
LONG_MAX = -73.85

# miles
DIST_MIN = 0
DIST_MAX = 20
        
def drop_conditional(df):
    # 1.
    # we have so much data, we can afford to just remove the nulls
    df.drop(df[df.isnull().any(1)].index, axis = 0, inplace=True)
    # 2.
    df.drop(df[df.fare_amount < MIN_FARE].index, axis=0, inplace=True)
    df.drop(df[df.fare_amount > MAX_FARE].index, axis=0, inplace=True)
    # 3.
    df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis = 0, inplace=True)
    df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis = 0, inplace=True)
    # 4.
    df.drop(df[df.pickup_latitude > LAT_MAX].index, axis=0, inplace=True)
    df.drop(df[df.pickup_latitude < LAT_MIN].index, axis=0, inplace=True)

    df.drop(df[df.pickup_longitude > LONG_MAX].index, axis=0, inplace=True)
    df.drop(df[df.pickup_longitude < LONG_MIN].index, axis=0, inplace=True)
    
    df.drop(df[df.dropoff_latitude > LAT_MAX].index, axis=0, inplace=True)
    df.drop(df[df.dropoff_latitude < LAT_MIN].index, axis=0, inplace=True)
    
    df.drop(df[df.dropoff_longitude > LONG_MAX].index, axis=0, inplace=True)
    df.drop(df[df.dropoff_longitude < LONG_MIN].index, axis=0, inplace=True)
    # 5.
    df.drop(df[df.distance > DIST_MAX].index, axis = 0, inplace=True)
    df.drop(df[df.distance < DIST_MIN].index, axis = 0, inplace=True)
            
def add_date_features(df, test=False):
    # 6.
    # df.drop(columns=['key'], inplace=True)
    df["pickup_datetime_clone"] = df["pickup_datetime"].values
    df.pickup_datetime_clone = df.pickup_datetime_clone.str.slice(0, 16)
    df.pickup_datetime_clone = pd.to_datetime(df.pickup_datetime_clone, utc=True, format='%Y-%m-%d %H:%M')
    # 7.
    df['year'] = df.pickup_datetime_clone.dt.year.astype('uint8')
    df['month'] = df.pickup_datetime_clone.dt.month.astype('uint8')
    df['day'] = df.pickup_datetime_clone.dt.day.astype('uint8')
    df['dayofweek'] = df.pickup_datetime_clone.dt.dayofweek.astype('uint8')
    df['hour'] = df.pickup_datetime_clone.dt.hour.astype('uint8')
    df['minute'] = df.pickup_datetime_clone.dt.minute.astype('uint8')
    # don't need the pickup_datetime anymore since it's been divided into the above cols
    df.drop(columns=['pickup_datetime_clone'], inplace=True) 
    df.drop(columns=['pickup_datetime'], inplace=True)
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
cols = list(traintypes.keys())
chunksize = 2**20 # 1,048,576
total_chunk = n_rows // chunksize + 1
df_list = [] # list to hold the batch dataframe
i=0

for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize):    
    i = i+1
    # Each chunk is a corresponding dataframe
    print(f'DataFrame Chunk {i:02d}/{total_chunk}')
    clean_data(df_chunk)
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk)
    del df_chunk
#     break
print("Complete")
# Merge all dataframes into one dataframe
X = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list

# See what we have loaded
X.info()
# Number of Passengers vs Fare Amount
plt.figure(figsize=(10,4))
plt.hist(X.passenger_count, bins = 16, range=(0,8), color="royalblue")
plt.xlabel("# Passengers")
plt.ylabel("Count")


plt.figure()
p1 = X[X.passenger_count==1].fare_amount
p2 = X[X.passenger_count==2].fare_amount
p3 = X[X.passenger_count==3].fare_amount
p4 = X[X.passenger_count==4].fare_amount
p5 = X[X.passenger_count==5].fare_amount
p6 = X[X.passenger_count==6].fare_amount

box = plt.boxplot([p1, p2, p3, p4, p5, p6], patch_artist=True)
plt.xlabel("# Passengers")
plt.ylabel("Fare ($)")
plt.yscale('log', nonposy='clip')
color = "g"
for element in ["boxes", "caps", "fliers"]:
    plt.setp(box[element], color=color)
    
plt.figure()
p1 = X[X.passenger_count==1].distance
p2 = X[X.passenger_count==2].distance
p3 = X[X.passenger_count==3].distance
p4 = X[X.passenger_count==4].distance
p5 = X[X.passenger_count==5].distance
p6 = X[X.passenger_count==6].distance

box = plt.boxplot([p1, p2, p3, p4, p5, p6], patch_artist=True)
plt.xlabel("# Passengers")
plt.ylabel("Distance (miles)")
plt.yscale('log', nonposy='clip')
color = "gold"
for element in ["boxes", "caps", "fliers"]:
    plt.setp(box[element], color=color)
# cleanup
del p1
del p2
del p3
del p4
del p5
del p6
gc.collect()
# Day of the Week vs Fare Amount
plt.figure()
plt.hist(X["dayofweek"], range=(0,7), bins=14, color="firebrick")
plt.xticks([0,1,2,3,4,5,6], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.ylabel("Count")

plt.figure()
mon = X[X.dayofweek==0].fare_amount
tue = X[X.dayofweek==1].fare_amount
wed = X[X.dayofweek==2].fare_amount
thu = X[X.dayofweek==3].fare_amount
fri = X[X.dayofweek==4].fare_amount
sat = X[X.dayofweek==5].fare_amount
sun = X[X.dayofweek==6].fare_amount

box = plt.boxplot([mon, tue, wed, thu, fri, sat, sun], patch_artist=True)
plt.xticks([1,2,3,4,5,6,7], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.yscale('log', nonposy='clip')
plt.ylabel("Fare ($)")
color = "g"
for element in ["boxes", "caps", "fliers"]:
    plt.setp(box[element], color=color)
# cleanup
del mon
del tue
del wed
del thu
del fri
del sat
del sun
gc.collect()
plt.figure()
plt.scatter(x=X.pickup_longitude, y=X.pickup_latitude, color="gold", s=0.01)
plt.title("Pickup")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.figure()
plt.scatter(x=X.dropoff_longitude, y=X.dropoff_latitude, color="gold", s=0.01)
plt.title("Dropoff")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
# Hour vs Fare Amount
plt.figure()
plt.hist(X.hour, bins=48, range=(0,24), color="firebrick")
plt.xlabel("Hour of Day")
plt.ylabel("# of Rides")

plt.figure()
plt.hist(X.fare_amount, bins=125, range=(MIN_FARE, MAX_FARE), color="g")
plt.yscale('log', nonposy='clip')
plt.xlabel("Fare ($)")
plt.ylabel("Count")
plt.figure()
plt.hist(X.distance, bins=50, range=(DIST_MIN, DIST_MAX), color="gold")
plt.yscale('log', nonposy='clip')
plt.xlabel("Distance")
plt.ylabel("Count")
# Fare vs Distance
plt.figure()
plt.scatter(x=X.distance, y=X.fare_amount, color="g", s=0.1)
plt.xlabel("Distance (miles)")
plt.ylabel("Fare ($)")
plt.show()