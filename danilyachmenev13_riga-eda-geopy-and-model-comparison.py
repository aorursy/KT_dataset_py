import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.utils import resample

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import KFold





import re



from geopandas.tools import geocode

import warnings

warnings.filterwarnings("ignore")



seed = 42
def missing(df):

    df_missing = pd.DataFrame(df.isna().sum().sort_values(ascending = False), columns = ['missing_count'])

    df_missing['missing_share'] = df_missing.missing_count / len(df)

    return df_missing
def simple_chart(df, x, title = None, hue = None):

    plt.figure(figsize = (10, 6))

    plt.title(title, fontsize=14)

    ax = sns.countplot(x = x, hue = hue, data = df)
def factor_chart(df, x, y, hue = None):

    ax = sns.factorplot(x = x, y = y, data = df, hue = hue, kind = 'box', size=6, aspect = 2)
def scatter(df, x, y, hue = None):

    plt.figure(figsize = (20, 10))

    ax = sns.scatterplot(x = x, y = y, data = df, hue = hue)

    plt.show()
sns.set(style="darkgrid")
df = pd.read_csv("../input/riga-real-estate-dataset/riga_re.csv")
df.head()
missing(df)
df[df.price.isna()]
df_all = df[~df.price.isna()].reset_index(drop = True).copy()
missing(df_all) 
df_all.dtypes
df_all.describe()
print('Number of observations:', len(df_all), '\n')

print('Unique values:')

print(df_all.nunique().sort_values(ascending = False))
df_all[df_all.street.isna()]
df_all = df_all.drop(df_all[df_all.street.isna()].index).reset_index(drop = True)
missing(df_all) 
# Function for removing digits from a string



def no_digits(text):

    return ''.join([i for i in text if not i.isdigit()])
df_all['street_name_0'] = df_all['street'].apply(lambda x: no_digits(re.sub('\W+',' ', str(x))).strip())
df_all.head(3)
# set(df_all.street_name_0.values)
df_all['st_n'] = None

for i in range(len(df_all)):

    if ((df_all.loc[i, 'street_name_0'][:3] != 'St ') & (df_all.loc[i, 'street_name_0'][:2] != 'J ') & 

        (df_all.loc[i, 'street_name_0'][:2] != 'M ')):

        df_all.loc[i, 'st_n'] = df_all.loc[i, 'street_name_0'].split(' ')[0]

    elif (df_all.loc[i, 'street_name_0'][:3] != 'St '):

         df_all.loc[i, 'st_n'] = df_all.loc[i, 'street_name_0'].split(' ')[0] + ' ' + df_all.loc[i, 'street_name_0'].split(' ')[1]

    else:

        df_all.loc[i, 'st_n'] = 'St ' + df_all.loc[i, 'street_name_0'].split(' ')[1]
#set(df_all.st_n.values)
df_all.drop(['street_name_0'], axis = 1, inplace = True)
df_all[df_all.district.isna()]
df_all[df_all.st_n == 'Ogļu'].groupby('district').count()
df_all.loc[1107, 'district'] = 'Ogļu'
df_all[df_all.st_n == 'Pupuku iela'].groupby('district').count()
df_all.loc[3172, 'district'] = 'Bišumuiža'
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="specify_your_app_name_here")
def lat(add):

    try:

        return geolocator.geocode(add).latitude

    except:

        return None



def lon(add):

    try:

        return geolocator.geocode(add).longitude

    except:

        return None
scatter(df_all, x = 'lon', y = 'lat')
scatter(df_all[(df_all.lat>56.88)&(df_all.lat<57.1)&(df_all.lon>20)], x = 'lon', y = 'lat')
df_all.loc[~((df_all.lat>56.88)&(df_all.lat<57.1)&(df_all.lon>20)), ['lat', 'lon']] = None
df_all['district'] = df_all["district"].replace('Krasta r-ns', 'Krasta masīvs')
df_all['Street_New'] = df_all['street']
df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' g.', ' gatve'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k-1', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k-2', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k1', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k 1', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k-1', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k-3', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('-k-3', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k-4', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' k. 1', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('k5', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('krastm.', 'krastmala'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' pr.', ' prospekts'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('Pulkv.', 'Pulkveža'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('bulv.', 'bulvāris'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('šķ. l.', 'šķērslīnija'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('šķ l.', 'šķērslīnija'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' l. ', ' līnija '))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' d. ', ' dambis '))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('J. Daliņa', 'Jāņa Daliņa'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('J. Vācieša', 'Jukuma Vācieša'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' g. ', ' gatve '))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' lauk.', ' laukums'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('k1', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('k2', '').strip())

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('-13d', '-13'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('-36d', '-36'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('-45d', '-45'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('-94b', '-94'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' 19/1', ' 19'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('M. Balasta', 'Mazais Balasta'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('M. Kuldīgas', 'Mazā Kuldīgas'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('M. Nometņu', 'Mazā Nometņu'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace('Asteres', 'Aisteres'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' 17 a', ' 17'))

df_all['Street_New'] = df_all["Street_New"].apply(lambda x: str(x).replace(' š. ', ' šoseja '))
df_all['Street_Full'] = df_all.apply(lambda x: str(x['Street_New']).split(' ')[0] + ' iela ' + str(x['Street_New']).split(' ')[1] +

                                    ', ' + str(x['district']) + ', ' + 'Rīga' if 

                                    len(x['Street_New'].split(' ')) == 2 else str(x['Street_New']) + ', ' + 

                                    'Rīga', axis = 1)
# This lines request full address that is stored in Street_Full feature. 

# Has to be launched 3-4 times, until the number of missing values stops decreasing (reaching 24 for both lat and lon specifically in this case)



#df_all['lat'] = df_all.apply(lambda x: lat(str(x['Street_Full'])) if np.isnan(x['lat']) == True else x['lat'], axis=1)

#df_all['lon'] = df_all.apply(lambda x: lon(str(x['Street_Full'])) if np.isnan(x['lon']) == True else x['lon'], axis=1)
# However, some full addresses do not work with district name, so for the left missings we use only street name and 'Riga'

# Also 2-3 times to execute (until 1 missing left for both lat and lon). 



#df_all['lat'] = df_all.apply(lambda x: lat(str(x['Street_Full'].split(',')[0]) + str(x['Street_Full'].split(',')[-1])) if np.isnan(x['lat']) == True else x['lat'], axis=1)

#df_all['lon'] = df_all.apply(lambda x: lon(str(x['Street_Full'].split(',')[0]) + str(x['Street_Full'].split(',')[-1])) if np.isnan(x['lon']) == True else x['lon'], axis=1)
# Remaining missing did not work with full address, but only street name was enough here.

# 1 execution is enough here



#df_all['lat'] = df_all.apply(lambda x: lat(str(x['Street_Full'].split(',')[0])) if np.isnan(x['lat']) == True else x['lat'], axis=1)

#df_all['lon'] = df_all.apply(lambda x: lon(str(x['Street_Full'].split(',')[0])) if np.isnan(x['lon']) == True else x['lon'], axis=1)
riga_fixed_coordinates = pd.read_csv('../input/riga-fixed-coordinates/riga_fixed_coordinates.csv')

missing(riga_fixed_coordinates)
df_all = riga_fixed_coordinates.copy()
df_all[~(df_all.lat>56.88)&(df_all.lat<57.1)&(df_all.lon>20)]
scatter(df_all, x = 'lon', y = 'lat')
df_all[df_all.area.isna()]
df_all[df_all.street == 'Slokas 130']
# Therefore

df_all.loc[3981, 'area'] = 80.0
df_all[df_all.rooms.isna()]
df_all[df_all.street == 'Dārzaugļu 1']
df_all.groupby(['rooms']).area.median()
df_all[df_all.rooms == 'Citi']
df_all = df_all.drop(df_all[df_all.rooms == 'Citi'].index, axis = 0).reset_index(drop = True)
# Look at the missing again as index was reseted

df_all[df_all.rooms.isna()]
df_all.loc[1610, 'rooms'] = '6'
df_all['rooms_num']= df_all['rooms'].astype('int64')
df_all[df_all.total_floors.isna()]
df_all[df_all.street == 'Zentenes 18']
df_all.loc[1902, 'total_floors'] = 9.0
missing(df_all)
plt.figure(figsize = (10, 6))

ax = sns.distplot(df_all.price, bins = 20) 
simple_chart(df_all, x = 'op_type')

factor_chart(df_all, x = 'op_type', y = 'price', hue = None)
df_all[~df_all.op_type.isin(['For rent', 'For sale'])]
df_all.loc[~df_all.op_type.isin(['For rent', 'For sale']) & (df_all.price < 1000), 'op_type'] = 'For rent'
df_all.loc[~df_all.op_type.isin(['For rent', 'For sale']) & (df_all.price > 1000), 'op_type'] = 'For sale'
simple_chart(df_all, x = 'op_type')
scatter(df_all[df_all.op_type == 'For sale'], x = 'area', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'area', y = 'price', hue = None)
simple_chart(df_all, x = 'condition')

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'condition', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'condition', y = 'price', hue = None)
df_all['All_Amen'] = 0

df_all.loc[df_all.condition == 'All amenities', 'All_Amen'] = 1
simple_chart(df_all, x = 'rooms') 

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'rooms', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'rooms', y = 'price', hue = None)
simple_chart(df_all, x = 'floor')

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'floor', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'floor', y = 'price', hue = None)
simple_chart(df_all, x = 'total_floors')

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'total_floors', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'total_floors', y = 'price', hue = None)
simple_chart(df_all, x = 'house_seria')

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'house_seria', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'house_seria', y = 'price', hue = None)
simple_chart(df_all, x = 'house_type')

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'house_type', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'house_type', y = 'price', hue = None)
simple_chart(df_all, x = 'district')

factor_chart(df_all[df_all.op_type == 'For sale'], x = 'district', y = 'price', hue = None)

factor_chart(df_all[df_all.op_type == 'For rent'], x = 'district', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For sale'], x = 'lon', y = 'lat', hue = 'district')

scatter(df_all[df_all.op_type == 'For rent'], x = 'lon', y = 'lat', hue = 'district')
scatter(df_all[df_all.op_type == 'For sale'], x = 'lat', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'lat', y = 'price', hue = None)
scatter(df_all[df_all.op_type == 'For sale'], x = 'lon', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'lon', y = 'price', hue = None)
Riga_Center_Lat = 56.949600

Riga_Center_Lon = 24.105200
import geopy.distance
def center_dist(lat_i, lon_i):

    return geopy.distance.vincenty((Riga_Center_Lat, Riga_Center_Lon), (lat_i, lon_i)).km
df_all['center_dist'] = df_all.apply(lambda x: center_dist(x['lat'], x['lon']), axis = 1)
scatter(df_all[df_all.op_type == 'For sale'], x = 'center_dist', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'center_dist', y = 'price', hue = None)
df_all['Area_Room_Ratio'] = df_all.area / df_all.rooms_num
scatter(df_all[df_all.op_type == 'For sale'], x = 'Area_Room_Ratio', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'Area_Room_Ratio', y = 'price', hue = None)
df_all['Floor_Ratio'] = df_all.floor / df_all.total_floors 
scatter(df_all[df_all.op_type == 'For sale'], x = 'Floor_Ratio', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'Floor_Ratio', y = 'price', hue = None)
df_all[df_all['Floor_Ratio'] > 1]
df_all.loc[(df_all['Floor_Ratio'] > 1), 'floor'] = df_all['floor'] / df_all['Floor_Ratio']

df_all.loc[(df_all['Floor_Ratio'] > 1), 'total_floors'] = df_all['Floor_Ratio'] * df_all['total_floors']
df_all.loc[(df_all['Floor_Ratio'] > 1), 'Floor_Ratio'] = df_all['floor'] / df_all['total_floors']
df_all[df_all['Floor_Ratio'] > 1]
scatter(df_all[df_all.op_type == 'For sale'], x = 'Floor_Ratio', y = 'price', hue = None)

scatter(df_all[df_all.op_type == 'For rent'], x = 'Floor_Ratio', y = 'price', hue = None)
df_sale = df_all[df_all.op_type == 'For sale'].reset_index(drop = True).copy()

df_rent = df_all[df_all.op_type == 'For rent'].reset_index(drop = True).copy()
scatter(df_sale, x = 'area', y = 'price', hue = None)

scatter(df_rent, x = 'area', y = 'price', hue = None)
scatter(df_rent[df_rent.price < 300], x = 'area', y = 'price', hue = 'center_dist')
df_sale_clean = df_sale[(df_sale.price < 300000) & (df_sale.area <160)  

                  & (~((df_sale.price < 50000) &(df_sale.area > 80))) 

                 & (~((df_sale.price < 100000)&(df_sale.area > 130)))

                  & (df_sale.Area_Room_Ratio<80)

                 ].copy()
df_rent_clean = df_rent[(df_rent.price < 1390) & (df_rent.area <125) & (df_rent.price > 60) 

                  & (~((df_rent.price < 110) &(df_rent.area > 40))) 

                 & (~((df_rent.price < 400)&(df_rent.area > 100)))

                  & (~((df_rent.price > 1000)&(df_rent.area < 70)))

                  &(df_rent.Area_Room_Ratio < 65)

                 ].copy()
scatter(df_sale_clean, x = 'area', y = 'price', hue = None)

scatter(df_rent_clean, x = 'area', y = 'price', hue = None)
df_sale_clean.columns
df_sale_clean = df_sale_clean.drop(['op_type', 'street', 'rooms', 'condition', 'Street_New', 'Street_Full'], axis = 1)

df_rent_clean = df_rent_clean.drop(['op_type', 'street', 'rooms', 'condition', 'Street_New', 'Street_Full'], axis = 1)
def get_splits(df):

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['price'], axis = 1), 

                                                          df['price'], train_size=0.8, test_size=0.2, 

                                                          random_state = seed)

    return X_train, X_test, y_train, y_test
OH_sale_clean = pd.get_dummies(df_sale_clean, drop_first = True)

OH_rent_clean = pd.get_dummies(df_rent_clean, drop_first = True)
OH_sale_train, OH_sale_test, OH_y_sale_train, OH_y_sale_test = get_splits(OH_sale_clean)

OH_rent_train, OH_rent_test, OH_y_rent_train, OH_y_rent_test = get_splits(OH_rent_clean)
cols_to_drop_sale = OH_sale_train.columns[(OH_sale_train == 0).all()]

OH_sale_train = OH_sale_train.drop(cols_to_drop_sale, axis = 1)

OH_sale_test = OH_sale_test.drop(cols_to_drop_sale, axis = 1)
cols_to_drop_rent = OH_rent_train.columns[(OH_rent_train == 0).all()]

OH_rent_train = OH_rent_train.drop(cols_to_drop_rent, axis = 1)

OH_rent_test = OH_rent_test.drop(cols_to_drop_rent, axis = 1)
models = [RandomForestRegressor(random_state = seed), 

          Ridge(random_state = seed), 

          RidgeCV(), 

          Lasso(random_state = seed), 

          LassoCV(random_state = seed), 

          ElasticNet(random_state = seed),

          HuberRegressor(), 

          KernelRidge(), 

          GradientBoostingRegressor(random_state = seed), 

          ExtraTreesRegressor(random_state = seed), 

          XGBRegressor(random_state = seed)]
models_names = [str(i).split('(')[0] for i in models]
def models_summary(train, test, y_train, y_test):

    models_MAE = []

    models_RMSE = []

    models_RMSLE = []

    for model in models:

        model.fit(train, y_train)

        preds = model.predict(test)

        models_MAE.append(mean_absolute_error(y_test, preds))

        models_RMSE.append(np.sqrt(mean_squared_error(y_test, preds)))

        models_RMSLE.append(np.sqrt(mean_squared_log_error(y_test, abs(preds))))

    return pd.DataFrame(list(zip(models_names, models_MAE, models_RMSE, models_RMSLE)),

              columns=['models','MAE', 'RMSE', 'RMSLE']).sort_values(by = 'MAE').set_index('models')
models_sale_res = models_summary(OH_sale_train, OH_sale_test, OH_y_sale_train, OH_y_sale_test)

models_sale_res
models_rent_res = models_summary(OH_rent_train, OH_rent_test, OH_y_rent_train, OH_y_rent_test)

models_rent_res
kf = KFold(n_splits=5, random_state=seed)
ET_model = ExtraTreesRegressor(random_state = seed,

                                n_estimators=400, 

                                min_samples_split=2,

                                min_samples_leaf=1, 

                                max_features=200,

                              )



params_grid = {#'n_estimators': range(50,50,201),

               #'max_features': range(50,401,50),

               #'min_samples_split': range(2,5),

               #'min_samples_leaf': range(1,4)

              }  



ET_grid = GridSearchCV(estimator = ET_model, param_grid = params_grid, n_jobs = -1,

                               cv = kf, scoring = 'neg_mean_absolute_error')
ET_grid_sale = ET_grid.fit(OH_sale_clean.drop(['price'], axis = 1), OH_sale_clean.price)

print(ET_grid_sale.best_params_)

print(ET_grid_sale.best_score_)
ET_grid_rent = ET_grid.fit(OH_rent_clean.drop(['price'], axis = 1), OH_rent_clean.price)

print(ET_grid_rent.best_params_)

print(ET_grid_rent.best_score_)
ET_sale = ExtraTreesRegressor(random_state = seed,

                                n_estimators=400, 

                                min_samples_split=2,

                                min_samples_leaf=1, 

                                max_features=200,

                              )

ET_rent = ExtraTreesRegressor(random_state = seed,

                                n_estimators=400, 

                                min_samples_split=2,

                                min_samples_leaf=1, 

                                max_features=200,

                              )
def score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)
score(ET_sale, OH_sale_train, OH_sale_test, OH_y_sale_train, OH_y_sale_test)
score(ET_rent, OH_rent_train, OH_rent_test, OH_y_rent_train, OH_y_rent_test)
def results(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    res_tab = pd.DataFrame({'y_test': y_test, 'preds': preds, 'error': (preds - y_test),

             'error_share': abs(y_test - preds)/y_test}).sort_values(by = 'error_share', ascending = False)

    return res_tab
sale_results = results(ET_sale, OH_sale_train, OH_sale_test, OH_y_sale_train, OH_y_sale_test)

rent_results = results(ET_rent, OH_rent_train, OH_rent_test, OH_y_rent_train, OH_y_rent_test)
sale_results[:10]
rent_results[:10]
def error_lines(df, y):

    plt.figure(figsize = (10, 6))

    ax = sns.lineplot(x = range(len(df)), y = y, data = df.sort_values(by = [y], ascending = False))
error_lines(sale_results, 'error')
error_lines(sale_results, 'error_share')
error_lines(rent_results, 'error')
error_lines(rent_results, 'error_share')