import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import cluster

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

import xgboost as xgb
df = pd.read_csv('../input/base-imoveis-123i/base_123i.csv')

df.head()
len(df)
# A visual check of each column's value

def check_df_columns(df):

    for col in df.columns:



        print(f" ----- {col} ----- ")

        print(100 * (df[col].value_counts() / df.shape[0]))

        

check_df_columns(df)
# Print how many nulls on each column

def print_null_cols(df):

    for col in df.columns:

        nulls_value = df[col].isna().sum()

        percentage = 100*(nulls_value / df.shape[0])

        message = "Column {} has {} nulls / {}% ".format(col, nulls_value, percentage)

        print(message)

        

print_null_cols(df)

df.dtypes
# Function to clean data

def clean_data(df):

    

    # remove weird data with values equal to -1

    df = df[df.maximum_estimate != -1]

    df = df[df.minimum_estimate != -1]

    df = df[df.point_estimate != -1]

    df = df[df.garages != -1]

    df = df[df.rooms != -1]

    df = df[df.useful_area != -1]

    

    # remove cities with low data

    city_dominance = 100 * (df['city'].value_counts() / df.shape[0])

    for a, b in zip(city_dominance.index, city_dominance):

        print(a)

        print(b)

        if b < 1:

            df = df[df.city != a]

            

    # lower case state

    df['state'] = [x.lower() for x in df['state']]

    

    return df



df = clean_data(df)



df = df.reset_index(drop=True)
print(100 * (df['city'].value_counts() / df.shape[0]))
len(df)
"""

neighborhood_list = []

for latitude, longitude in tqdm_notebook(zip(df['latitude'], df['longitude']), total=len(df)):



    #ex: https://maps.googleapis.com/maps/api/geocode/json?key=APIKEY&latlng=-23.56417040,-46.65790930

    

    try:

        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?key=APIKEY&latlng={0},{1}'.format(latitude, longitude))

        resp_json_payload = response.json()        

        #type_component = resp_json_payload['results'][0]['address_components'][2]['types']

        type_component = resp_json_payload['results'][0]['address_components']

        

        neighborhood = None

        for i in type_component:            

            list_component_type = i['types']            

            if 'sublocality' in list_component_type:            

                neighborhood = i['long_name']

                break

    except:

        neighborhood = None

        

    neighborhood_list.append(neighborhood)



df['neighborhood'] = neighborhood_list

"""
# Create cluster using Kmeans

def find_cluster(df):



    location = df.copy()



    columns_to_keep = ['latitude', 'longitude']



    for col in location:

        if col not in columns_to_keep:

            location = location.drop(col, 1)

    

    location['lng_parsed'] = pd.to_numeric(location['longitude'], errors='coerce')

    location['lat_parsed'] = pd.to_numeric(location['latitude'], errors='coerce')

    

    location = location.drop('longitude', 1)

    location = location.drop('latitude', 1)



    # Remove Outliers do dataframe

    location = location[((location.lat_parsed - location.lat_parsed.mean()) / location.lat_parsed.std()).abs() < 3]

    location = location[((location.lng_parsed - location.lng_parsed.mean()) / location.lng_parsed.std()).abs() < 3]

    location = location[np.abs(location.lat_parsed-location.lat_parsed.mean()) <= (3*location.lat_parsed.std())]

    location = location[np.abs(location.lng_parsed-location.lng_parsed.mean()) <= (3*location.lng_parsed.std())]



    for i in range(0, len(df)):

        if i not in location.index:

            df = df[df.index != i]



    df = df.reset_index(drop=True)



    x1 = location.lng_parsed

    x2 = location.lat_parsed

    

    # Plot charts and execute Kmeans clustering

    plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')

    plt.title('Latitude x Longitude')

    plt.xlabel('Longitude')

    plt.ylabel('Latitude')

    plt.scatter(x1, x2)

    plt.show()



    

    plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')

    plt.scatter(location.lng_parsed, location.lat_parsed)

    plt.title('Clustering neighborhoods')

    plt.xlabel('Longitude')

    plt.ylabel('Latitude')



    location_full = location



    location_np = np.array(location)



    # Execute Kmeans clustering

    # São Paulo has approximately 100 neighborhoods

    k = 100 # Define the value of k

    kmeans = cluster.KMeans(n_clusters=k, random_state=42)

    kmeans.fit(location_np)



    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_



    for i in range(k):

        # select only data observations with cluster label == i

        ds = location_np[np.where(labels==i)]

        # plot the data observations

        plt.plot(ds[:,0],ds[:,1],'o', markersize=6)

        # plot the centroids

        lines = plt.plot(centroids[i,0],centroids[i,1],'kx')

        # make the centroid x's bigger

        plt.setp(lines,ms=8.0)

        plt.setp(lines,mew=3.0)

    plt.show()



    list_areas_kmeans = []

    for f, b in zip(labels, location_full.index):

        list_areas_kmeans.append(f)



    return list_areas_kmeans, df, location
list_areas_kmeans, df, location = find_cluster(df)



df['area_kmeans'] = list_areas_kmeans
df.head()
order_by_median = df.groupby(by=["area_kmeans"])["point_estimate"].median().sort_values(ascending=True).index



import matplotlib.ticker as ticker



sns.set(rc={'figure.figsize':(17,5)})

sns.set(palette="pastel")

sns.boxplot(x="area_kmeans", y="point_estimate", color="orange", data=df, showfliers=False)

sns.despine(offset=10, trim=True)

ax = plt.gca()

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))

plt.show()



sns.set(rc={'figure.figsize':(17,5)})

sns.set(font_scale=0.78)

sns.boxplot(x="area_kmeans", y="point_estimate", palette="rainbow", data=df, showfliers=False, order=order_by_median)

sns.despine(offset=10, trim=True)

#ax = plt.gca()

#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

#ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))

plt.show()
print(order_by_median[-5:].tolist())
latitude_list = []

longitude_list = []

label_area_kmeans = []

for i in order_by_median[-5:]:

    expensive_areas = df.loc[(df['area_kmeans'] == i)]

    latitude_list.append(expensive_areas['latitude'][0:1])

    longitude_list.append(expensive_areas['longitude'][0:1])

    label_area_kmeans.append(i)

    

latitude_list_less = []

longitude_list_less = []

label_area_kmeans_less = []

for i in order_by_median[0:5]:

    less_expensive_areas = df.loc[(df['area_kmeans'] == i)]

    latitude_list_less.append(less_expensive_areas['latitude'][0:1])

    longitude_list_less.append(less_expensive_areas['longitude'][0:1])

    label_area_kmeans_less.append(i)
import folium

from folium import plugins



heatmap = df.copy()

heatmap['count'] = 1

base_heatmap = folium.Map(location=['-23.5713874', '-46.6522521'], zoom_start=12)



for a, b, c in zip(latitude_list, longitude_list, label_area_kmeans):    

    folium.Marker((a, b), popup=c, icon=folium.Icon(color='red')).add_to(base_heatmap)

    

for a, b, c in zip(latitude_list_less, longitude_list_less, label_area_kmeans_less):    

    folium.Marker((a, b), popup=c, icon=folium.Icon(color='darkblue')).add_to(base_heatmap)



plugins.HeatMap(data=heatmap[['latitude', 'longitude', 'count']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=4).add_to(base_heatmap)



base_heatmap
# Remove unuseful columns

def drop_columns(df, list_columns):

    df = df.drop(list_columns, axis=1)    

    

    return df



list_columns = ['address', 'tower_name', 'latitude', 'longitude', 'city', 'state']

df = drop_columns(df, list_columns)
# Convert some columns to dummies

def one_hot_encoder(df):

    one_hot = pd.get_dummies(df['building_type'])

    df = df.drop('building_type',axis = 1)

    df = df.join(one_hot)

    

    return df



df = one_hot_encoder(df)
# Split data/target

def get_data_target(df, target):

    y = df[target]    

    X = df.drop(target,axis = 1)

    

    return X, y
# Function to train the model

def train(model):



    model = model



    model.fit(X_train, y_train,

            eval_set = [(X_train, y_train), (X_test, y_test)],

            eval_metric = 'rmse',

            early_stopping_rounds = 5,

            verbose=True)



    best_iteration = model.get_booster().best_ntree_limit

    preds = model.predict(X_test, ntree_limit=best_iteration)

    

    return preds, model, best_iteration
# Function to evaluate some metrics

def evaluate_metrics(preds, model, best_iteration):

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    r2 = r2_score(y_test, preds, multioutput='variance_weighted')

    

    evals_result = model.evals_result()  

    plt.rcParams["figure.figsize"] = (8, 6)

    xgb.plot_importance(model, max_num_features=None)

    plt.show()



    range_evals = np.arange(0, len(evals_result['validation_0']['rmse']))



    val_0 = evals_result['validation_0']['rmse']

    val_1 = evals_result['validation_1']['rmse']

    plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')

    plt.plot(range_evals, val_1, range_evals, val_0)

    plt.ylabel('Validation error')

    plt.xlabel('Iteration')

    plt.show()

    

    print("Best iteration: %f" % (int(best_iteration)))

    print("RMSE: %f" % (rmse))

    print("R2: %f" % (r2))

    print('Observed value single sample: {}'.format(y_test[0:1].tolist()[0]))

    print('Predicted value single sample: {}'.format(int(model.predict(X_test[0:1], ntree_limit=best_iteration)[0])))
# First model

target = 'point_estimate'

X, y = get_data_target(df, target)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



xg_reg = xgb.XGBRegressor()



preds, model, best_iteration = train(xg_reg)



evaluate_metrics(preds, model, best_iteration)
# Second model

list_columns = ['minimum_estimate', 'maximum_estimate']

df = drop_columns(df, list_columns)



target = 'point_estimate'

X, y = get_data_target(df, target)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



xg_reg = xgb.XGBRegressor()



preds, model, best_iteration = train(xg_reg)



evaluate_metrics(preds, model, best_iteration)
# Function to create new features

def feature_engineering(df):



    mean_rooms_list = []

    mean_garages_list = []

    mean_useful_area_list = []    

    std_rooms_list = []

    std_garages_list = []

    std_useful_area_list = []

    

    for i in tqdm_notebook(df.index):



        rooms_df = df.loc[(df['rooms']) & ((df['area_kmeans'] ==  df['area_kmeans'][i]))]

        mean_rooms = rooms_df['rooms'].mean()

        mean_rooms_list.append(float(mean_rooms))

        std_rooms = rooms_df['rooms'].std()

        std_rooms_list.append(float(std_rooms))



        garages_df = df.loc[(df['garages']) & ((df['area_kmeans'] ==  df['area_kmeans'][i]))]

        mean_garages = garages_df['garages'].mean()

        mean_garages_list.append(float(mean_garages))

        std_garages = garages_df['garages'].std()

        std_garages_list.append(float(std_garages))

        

        useful_area_df = df.loc[(df['useful_area']) & ((df['area_kmeans'] ==  df['area_kmeans'][i]))]

        mean_useful_area = useful_area_df['useful_area'].mean()

        mean_useful_area_list.append(float(mean_useful_area))

        std_useful_area = useful_area_df['useful_area'].std()

        std_useful_area_list.append(float(std_useful_area))



    df['mean_rooms'] = mean_rooms_list

    df['mean_garages'] = mean_garages_list

    df['mean_useful_area'] = mean_useful_area_list

    df['std_rooms'] = std_rooms_list

    df['std_garages'] = std_garages_list

    df['std_useful_area'] = std_useful_area_list    

    

    return df
df = feature_engineering(df)

df.head()
# Third model

target = 'point_estimate'

X, y = get_data_target(df, target)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



xg_reg = xgb.XGBRegressor()



preds, model, best_iteration = train(xg_reg)



evaluate_metrics(preds, model, best_iteration)
df = pd.read_csv('../input/sao-paulo-real-estate-sale-rent-april-2019/sao-paulo-properties-april-2019.csv')

df = df[df['Negotiation Type'] == 'sale']

df.head()
list_columns = ['Property Type', 'Latitude', 'Longitude', 'Negotiation Type']

df = drop_columns(df, list_columns)



#df['District'] = df['District'].astype('category')



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df['District'] = labelencoder.fit_transform(df['District'])



target = 'Price'

X, y = get_data_target(df, target)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



xg_reg = xgb.XGBRegressor()



preds, model, best_iteration = train(xg_reg)



evaluate_metrics(preds, model, best_iteration)
sample = pd.DataFrame({'Condo': 1300, 'Size': 100, 'Rooms': 2, 'Toilets': 2, 'Suites': 1, 'Parking': 1, 'Elevator': 1, 'Furnished': 0, 'Swimming Pool': 0, 'New': 0, 'District': 40}, index=[0])



print(model.predict(sample, ntree_limit=best_iteration))