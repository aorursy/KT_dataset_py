# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import geopandas as gpd

import numpy as np

import requests

import shapely

import matplotlib.pyplot as plot

%matplotlib inline





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
travel_times = pd.read_csv('/kaggle/input/uber-movement-data/Travel_Times.csv')

travel_times_daily = pd.read_csv('/kaggle/input/uber-movement-data/Travel_Times_Daily.csv')

travel_times_day = pd.read_csv('/kaggle/input/uber-movement-data/Travel_Times_time_of_day.csv')

travel_times_week = pd.read_csv('/kaggle/input/uber-movement-data/Travel_Times_day_of_week.csv')

bnglr_wards_hourly = pd.read_csv('/kaggle/input/uber-movement-data/bangalore-wards-2020-1-All-HourlyAggregate.csv')

bnglr_wards_weekly = pd.read_csv('/kaggle/input/uber-movement-data/bangalore-wards-2020-1-WeeklyAggregate.csv')

bnglr_wards_monthly = pd.read_csv('/kaggle/input/uber-movement-data/bangalore-wards-2020-1-All-MonthlyAggregate.csv')
bnglr_wards_hourly.head(2)
mean_travel_time_by_hour_of_day = bnglr_wards_hourly.groupby('hod')['mean_travel_time'].mean()/60

plt = mean_travel_time_by_hour_of_day.plot(kind="bar", figsize=(16,7))

plt.set_title('Mean travel times around Bangalore',fontsize=20)

plt.set_xlabel('Hour of day', fontsize=16)

_ = plt.set_ylabel('Mean travel time in mins', fontsize=16)
std_dev_time_by_hour_of_day = bnglr_wards_hourly.groupby('hod')['standard_deviation_travel_time'].mean()/60

plt = std_dev_time_by_hour_of_day.plot(kind="bar", figsize=(16,7))

plt.set_title('Std Dev of travel times around Bangalore',fontsize=20)

plt.set_xlabel('Hour of day', fontsize=16)

_ = plt.set_ylabel('Std Dev of travel time in mins', fontsize=16)
bglr=gpd.read_file('/kaggle/input/uber-movement-data/bangalore_wards.json')

bglr.plot()
bglr.head()
bglr = bglr.drop(columns=['WARD_NO', 'MOVEMENT_ID'])

bglr_c = bglr.copy()

bglr_c.geometry = bglr_c['geometry'].centroid

fig, ax = plot.subplots(figsize=(9,9))

bglr.plot(color='grey',ax=ax)

bglr_c.plot(color='red',ax=ax)
id_to_dest = travel_times[['Destination Movement ID', 'Destination Display Name']]

id_to_dest.columns = ['id', 'name']

id_to_dest.head()

from shapely.geometry import Point

import random



# The number of rows with random points to be created corresponding to each row in source df

number = 3



def random_points_in_polygon(number, polygon):

    points = []

    min_x, min_y, max_x, max_y = polygon.bounds

    i= 0

    while i < number:

        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        if polygon.contains(point):

            points.append(point)

            i += 1

    return points



def diversify_geo_data(df):

    new_df = gpd.GeoDataFrame()

    common_columns = df.columns

    common_columns.tolist().remove('geometry')

    for row in df.itertuples():

        points = random_points_in_polygon(number, row.geometry)

        for point in points:

            tmp = gpd.GeoDataFrame(columns=df.columns, data=[list(row[1:3]) + [point]])

            new_df = new_df.append(tmp, ignore_index=True)

    return new_df

            



diversified_points = diversify_geo_data(bglr)

diversified_points.sample(5)
diversified_points.shape
fig, ax = plot.subplots(figsize=(9,9))

bglr.plot(color='grey',ax=ax)

diversified_points.plot(color='red',ax=ax)
time_df = pd.merge(bnglr_wards_hourly, id_to_dest, left_on=['sourceid'], right_on=['id'], how='inner')

time_df = time_df.drop(columns=['id', 'geometric_mean_travel_time', 'geometric_standard_deviation_travel_time'])

time_df = time_df.rename(columns={'name': 'Source Name'})

time_df = pd.merge(time_df, id_to_dest, left_on=['dstid'], right_on=['id'], how='inner')

time_df = time_df.drop(columns=['id'])

time_df = time_df.rename(columns={'name': 'Destination Name'})

time_df = time_df.sort_values(by=['sourceid', 'dstid', 'hod'])

time_df.tail(5)
bglr_c.shape
diversified_points.shape
full_bglr = bglr_c.append(diversified_points, ignore_index=True)

full_bglr.shape
time_df2 = pd.merge(time_df, full_bglr, left_on=['Source Name'], right_on=['DISPLAY_NAME'], how='inner')

time_df2 = time_df2.drop(columns=['DISPLAY_NAME'])

time_df2 = time_df2.rename(columns = {'WARD_NAME': 'Source Ward Name', 'geometry': 'Source Geometry'})

time_df2 = pd.merge(time_df2, full_bglr, left_on=['Destination Name'], right_on=['DISPLAY_NAME'], how='inner')

time_df2 = time_df2.drop(columns=['DISPLAY_NAME'])

time_df2 = time_df2.rename(columns = {'WARD_NAME': 'Destination Ward Name', 'geometry': 'Destination Geometry'})

time_df2.sample(3)
time_df2.shape
import pickle

def save_object(obj, filename):

    with open(filename, 'wb') as output:  # Overwrites any existing file.

        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



import os.path

def file_exists(filename):

    return os.path.exists(filename)



input_path = '../input/uber-travel-time-prediction/'
import geopy.distance



def calc_distance(x):

    src_point = (x['Source Geometry'].y, x['Source Geometry'].x)

    dest_point = (x['Destination Geometry'].y, x['Destination Geometry'].x)

    return geopy.distance.geodesic(src_point, dest_point).kilometers



filename = 'Df_for_modelling.bin'

path = input_path + filename

if file_exists(path):

    # skip to next section since the results here are already precalculated

    pass

else:

    print('Creating distance file')

    time_df2['Geodesic Distance'] = time_df2.apply(func = calc_distance, axis=1)

    df = time_df2

filename = 'Df_for_modelling.bin'

path = input_path + filename

if file_exists(path):

    with open(path, 'rb') as file:

        final_df = pickle.load(file)

else:

    print('Creating final df file')

    final_df = df.copy()

    final_df['Source lat'] = final_df['Source Geometry'].apply(lambda pt: float(pt.y))

    final_df['Source long'] = final_df['Source Geometry'].apply(lambda pt: float(pt.x))

    final_df['Dest lat'] = final_df['Destination Geometry'].apply(lambda pt: float(pt.y))

    final_df['Dest long'] = final_df['Destination Geometry'].apply(lambda pt: float(pt.x))



    

features = ['Source lat', 'Source long', 'Dest lat', 'Dest long', 'hod', 'Geodesic Distance']

outcome = ['mean_travel_time']

final_df = final_df[features + outcome]



try:

    save_object(final_df, filename)

except:

    pass


X = final_df[features]

y = final_df[outcome]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)



import xgboost as xgb



filename = 'XGB_model_5.bin'

path = input_path + filename

if file_exists(path):

    with open(path, 'rb') as file:

        my_model = pickle.load(file)

else:

    my_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4, objective='reg:squarederror')

    my_model.fit(X_train, y_train,verbose=False)

    

try:    

    save_object(X_train, 'train_set_x.bin')

    save_object(y_train, 'train_set_y.bin')

    save_object(X_test, 'test_set_x.bin')

    save_object(y_test, 'test_set_y.bin')

    save_object(my_model, filename)

except:

    pass
xgb.plot_importance(my_model)
predictions = my_model.predict(X_test)



from sklearn import metrics



r2 = metrics.r2_score(y_test, predictions)

print('R2: {}\n'.format(r2))



mse = metrics.mean_squared_error(y_test, predictions)

print('MSE: {}\n'.format(mse))



print('RMSE: {}\n'.format(np.sqrt(mse)))



mae = metrics.mean_absolute_error(y_test, predictions)

print('MAE: {}\n'.format(mae))

x_ax = range(len(y_test))

plot.plot(x_ax, y_test, label="original")

plot.plot(x_ax, predictions, label="predicted")



plot.title("Mean travel times actual and predicted data")



plot.legend()

plot.show()
fig, ax = plot.subplots()

ax.scatter(y_test, predictions)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured Time')

ax.set_ylabel('Predicted Time')

plot.show()
from sklearn.model_selection import cross_val_score, KFold



score = my_model.score(X_train, y_train)  

print("Training score: ", score)



# These take more than 9 hours to complete so skipped



# scores = cross_val_score(my_model, X_train, y_train, cv=10)

# print("Mean cross-validation score: %.2f" % scores.mean())



# kfold = KFold(n_splits=10, shuffle=True)

# kf_cv_scores = cross_val_score(my_model, X_train, y_train, cv=kfold )

# print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
features = ['Source lat', 'Source long', 'Dest lat', 'Dest long', 'hod', 'Geodesic Distance']

outcome = ['mean_travel_time']



def get_distance(lat1, long1, lat2, long2):

    src_point = (lat1, long1)

    dest_point = (lat2, long2)

    return geopy.distance.geodesic(src_point, dest_point).kilometers



def prepare_df(lat1, long1, lat2, long2, hod):

    distance = get_distance(lat1, long1, lat2, long2)

    return pd.DataFrame(columns = ['Source lat', 'Source long', 'Dest lat', 'Dest long', 'hod', 'Geodesic Distance'],

                 data = [[lat1, long1, lat2, long2, hod, distance]])

    

def predict(df):

    return my_model.predict(df[features])
def compare(actual, predicted):

#     actual = [act[0] for act in actual[outcome].values.tolist()]

#     predicted = predicted.tolist()

    return pd.DataFrame(data = {'actual': actual, 'prediction': predicted})

# entire bangalore geojson from https://github.com/datameet/PincodeBoundary/tree/master/Bangalore



# bangalore_polygon = gpd.read_file('../input/external-geodata/bangalore boundary.geojson')

# bangalore_polygon.plot()
def get_random_points_in_bangalore(number):

    points = []

    min_x, min_y, max_x, max_y = 12.85, 77.45, 13.0, 77.75  

    i= 0

    while i < number:

        point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        i += 1

        points.append(point)

    return points



# ORS_API_KEY = '5b3ce3597851110001cf6248f4d0d79bab0a4f5a8b95a4403ba8b839'



# def get_ORS_data(source, dest):

#     parameters = {

#         'locations' : [['{},{}'.format(source[1], source[0])],['{},{}'.format(dest[1], dest[0])]]

#     }

#     headers = {'Authorization ': ORS_API_KEY}

#     response = requests.post(

#         'https://api.openrouteservice.org/v2/matrix/driving-car', data=parameters, headers=headers)



#     if response.status_code == 200:

#         print('Request successful.')

#         data = response.json()

#         summary = data['features'][0]['properties']['summary']

#         print(summary)

# #         distance = summary['distance']/1000

#         return duration

#     else:

#         print('Request failed.')

#         print(response.text)
# get_ORS_data((12.928781971722811, 77.6121303701099), (12.854079170010449, 77.55145104575789))
# points1 = get_random_points_in_bangalore(10)

# points2 = get_random_points_in_bangalore(10)



# travel_times_ORS = []

# for point1, point2 in zip(points1, points2):

#     travel_times_ORS.append(get_ORS_data(point1, point2))

    

# travel_times_model = []

# for point1, point2 in zip(points1, points2):

#     lat1, long1, lat2, long2 = point1[0], point1[1], point2[0], point2[1]

#     hod = random.uniform(0, 23)

#     p = predict(prepare_df(lat1, long1, lat2, long2, hod))

#     p = p.tolist()[0]

#     travel_times_model.append(p)

    

# print({points: time for points, time in zip(zip(points1, points2), travel_times_model)})

# compare(travel_times_ORS, travel_times_model)
points = [((12.999289603200602, 77.72750046509455),

  (12.900586869608652, 77.57751972070913)),

 ((12.935917259525278, 77.61353555551875),

  (12.856440148061886, 77.48546536718554)),

 ((12.897459477917653, 77.7095308106631),

  (12.996842230621631, 77.64938231715406)),

 ((12.936661694778596, 77.71873838420447),

  (12.907286088453898, 77.4772158363428)),

 ((12.88453533783865, 77.70914845848147),

  (12.893719016027402, 77.71748845762684)),

 ((12.90661849980788, 77.6359030720415),

  (12.986913454190185, 77.6667791482254)),

 ((12.89600641097292, 77.70444731327966),

  (12.98987526206819, 77.60524558740501)),

 ((12.897523292566822, 77.56067149766076),

  (12.894608188894253, 77.66337661025209)),

 ((12.86194623889842, 77.56338966329798),

  (12.944443289052925, 77.4887976744094)),

 ((12.858621405573748, 77.46556646993612),

  (12.90990445061544, 77.58937083560666))]



hours_of_day = [18, 7, 18, 2, 16, 23, 10, 8, 4, 16]
# taking average of thetime bounds given by manual google maps travel times for same coordinates and departure time

travel_times_gmaps_in_mins = [75, 55, 43, 63, 7, 45, 60, 50, 40, 50]

travel_times_gmaps = [t*60 for t in travel_times_gmaps_in_mins]
travel_times_model = []

points1 = [p[0] for p in points]

points2 = [p[1] for p in points]

for point1, point2, hod in zip(points1, points2, hours_of_day):

    lat1, long1, lat2, long2 = point1[0], point1[1], point2[0], point2[1]

    p = predict(prepare_df(lat1, long1, lat2, long2, hod))

    p = p.tolist()[0]

    travel_times_model.append(p)

[t/60 for t in travel_times_model]
compare(travel_times_gmaps, travel_times_model)
x_ax = range(len(travel_times_model))

plot.plot(x_ax, travel_times_gmaps, label="original")

plot.plot(x_ax, travel_times_model, label="predicted")



plot.title("Mean travel times actual and predicted data")



plot.legend()

plot.show()
lat1 = 13.002385

long1 = 77.568491

lat2 = 13.061071

long2 = 77.597371

hod = 10



df = prepare_df(lat1, long1, lat2, long2, hod)

predict(df)