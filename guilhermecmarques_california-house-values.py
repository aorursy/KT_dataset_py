%matplotlib inline

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import os

df = pd.read_csv("../input/atividade-3-pmr3508/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
df = df.replace(np.nan,' ', regex=True)
df.info()
df.describe()
df.head()
_ = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(17, 17), diagonal='hist')
df.corr(method='pearson', min_periods=1)
X = df.copy()
X["coord"] = X["latitude"].map(str) + ', ' + X["longitude"].map(str)
from geopy.geocoders import AzureMaps
geolocator = AzureMaps(subscription_key ='31bvyiOntGBj50ADppmbf1V9ZO5Do-NeYjjXF56bcbo')
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.08)

trainLocation = pd.DataFrame(X.Id)

from tqdm import tqdm
from functools import partial
tqdm.pandas()

# as próximas linhas demoram horas para serem completadas, por isso foram comentadas

# trainLocation["location"] = X["coord"].progress_apply(partial(geocode, exactly_one=True))
# trainLocation.to_csv("locations.csv", index=False)
address = pd.read_csv("../input/california-prices/locations.csv",
          sep=r'\s*"\s*',
          engine='python')

address.columns = [col.replace('"', '') for col in address.columns]
address = address.replace('"','', regex=True)

address['location'] = address['location'].str.split(",")

address = address.replace(',','', regex=True)
cities = pd.DataFrame(address.Id)
cities = cities.set_index('Id')
cities['city'] = ''

for index, row in address.iterrows():
    city = row['location'][-2]
    row_id = row['Id']
    cities.at[row_id, 'city'] = city.lstrip(' ')

cities.to_csv("cities.csv", index=False)
cities = pd.read_csv("../input/california-prices/cities.csv",
          sep=r'\s*"\s*',
          engine='python')
X['cities'] = cities['city']
calPrices = pd.read_csv("../input/california-prices/calif.csv",
        sep=r'\s*,\s*\s*"',
        engine='python')


calPrices = calPrices.replace('---',' ', regex=True)
calPrices.columns = [col.replace('"', '') for col in calPrices.columns]
calPrices = calPrices.filter(['Region Name', 'Current'])
calPrices = calPrices.replace('"','', regex=True)
calPrices = calPrices.replace('\$','', regex=True)
calPrices = calPrices.replace(',','', regex=True)

calPrices = calPrices.drop(calPrices.index[0])
calPrices.head()
X['city_price'] = ''

for index, row in calPrices.iterrows():
    city = row['Region Name']
    avg_price = row['Current']
    Xcity = X[X['cities'].str.match(city)]
    for index, row in Xcity.iterrows():
        X.at[index, 'city_price'] = avg_price

X = X.replace('', np.NaN)
X = X.replace(' ', np.NaN)
X['city_price'] = pd.to_numeric(X['city_price'])
X['people_pb'] = X.population/X.total_bedrooms
X['people_ph'] = X.population/X.households
X['income_pr'] = X.median_income/X.total_rooms
X = X.replace('', np.NaN)
X = X.replace(' ', np.NaN)
X = X.dropna()
X = X.drop(['Id','longitude','latitude','coord'], axis = 1)
X_train = X.filter(['median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income', 'city_price', 'people_pb','people_ph', 'income_pr'], axis = 1)
X_train.describe()
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
Y = X.median_house_value
Y.describe()
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

reg = linear_model.LinearRegression().fit(X_train_scaled, Y)
reg.score(X_train_scaled, Y)
from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train_scaled, Y)
knn_scores = cross_val_score(knn, X_train_scaled, Y, cv=10)

np.mean(knn_scores)
from sklearn import ensemble

params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)

gbr.fit(X_train_scaled, Y)
Yreg = reg.predict(X_train_scaled)

Yknn = knn.predict(X_train_scaled)

Ygbr = gbr.predict(X_train_scaled)
from sklearn.metrics import mean_squared_log_error

mean_squared_log_error(Y, Yknn) 
mean_squared_log_error(Y, Ygbr) 
testdf = pd.read_csv("../input/atividade-3-pmr3508/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testdf = testdf.replace(np.nan,' ', regex=True)
testdf["coord"] = testdf["latitude"].map(str) + ', ' + testdf["longitude"].map(str)
testLocation = pd.DataFrame(testdf.Id)

from tqdm import tqdm
from functools import partial
tqdm.pandas()

geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.08)

# testLocation["location"] = testdf["coord"].progress_apply(partial(geocode, exactly_one=True))
# testLocation.to_csv("test_loc.csv", index=False)
test_loc = pd.read_csv("../input/california-prices/test_loc.csv",
          sep=r'\s*"',
          engine='python')

test_loc.columns = [col.replace('"', '') for col in test_loc.columns]
test_loc = test_loc.replace('"','', regex=True)

test_loc['location'] = test_loc['location'].str.split(",")

test_loc = test_loc.replace(',','', regex=True)
test_cities = pd.DataFrame(test_loc.Id)
test_cities = test_cities.set_index('Id')
test_cities['city'] = ''

for index, row in test_loc.iterrows():
    row_id = row['Id']
    city = row['location'][-2]
    if city != 'n':
        test_cities.at[row_id, 'city'] = city.lstrip(' ')

test_cities.to_csv("test_cities.csv", index=False)
test_cities = pd.read_csv("../input/california-prices/test_cities.csv",
          engine='python')

testdf['cities'] = test_cities['city']

testdf['city_price'] = ''

for index, row in calPrices.iterrows():
    city = row['Region Name']
    avg_price = row['Current']
    Xcity = testdf[testdf['cities'].str.match(city)]
    for index, row in Xcity.iterrows():
        testdf.at[index, 'city_price'] = avg_price

# segundo o dataset com as informações de preço médio, a média da
# California é de $544900
testdf['city_price'] = testdf['city_price'].replace('', 544900.0)
testdf['city_price'] = testdf['city_price'].replace(' ', 544900.0)
testdf['city_price'] = pd.to_numeric(testdf['city_price'])
X_test = testdf.copy()

X_test['people_pb'] = X_test.population/X_test.total_bedrooms
X_test['people_ph'] = X_test.population/X_test.households
X_test['income_pr'] = X_test.median_income/X_test.total_rooms

X_test = X_test.filter(['median_age','total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'city_price', 'people_pb','people_ph', 'income_pr'], axis = 1)
X_test.describe()
X_test_scaled = scaler.transform(X_test)
Ypred = knn.predict(X_test_scaled)
prediction = pd.DataFrame(testdf.Id)
prediction['median_house_value'] = Ypred
prediction['median_house_value'] = prediction['median_house_value'].abs()
prediction.to_csv("prediction.csv", index=False)