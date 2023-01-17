import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/zomato.csv',low_memory=False)
df.info()
print('Features \t # unique values\n')

for col in list(df):

    print(f'{col}:\t{df[col].nunique()}')
df.head(2)
df.rename({'approx_cost(for two people)': 'approx_cost_2_people',

           'listed_in(type)':'listed_in_type',

           'listed_in(city)':'listed_in_city'

          }, axis=1, inplace=True)
# approx_cost constains some values of format '1,000' wich could not be directly convert to int

# we need to have this format '1000' in order to do the convertion

# We will use the lambda function below to transform '1,000' to '1000' and then to int

replace_coma = lambda x: int(x.replace(',', '')) if type(x) == np.str and x != np.nan else x 

df.votes = df.votes.astype('int')

df['approx_cost_2_people'] = df['approx_cost_2_people'].apply(replace_coma)

df = df.drop(['url', 'phone'], axis=1)
df.rate.dtype, df.rate[0]
df.rate.unique()
(df.rate =='NEW').sum(), (df.rate =='-').sum()
df = df.loc[df.rate !='NEW']

df = df.loc[df.rate !='-'].reset_index(drop=True)
print(f'The new shape of the date is {df.shape}')
new_format = lambda x: x.replace('/5', '') if type(x) == np.str else x

df.rate = df.rate.apply(new_format).str.strip().astype('float')

df.rate.head()
def label_encode(df):

    for col in df.columns[~df.columns.isin(['rate', 'approx_cost_2_people', 'votes'])]:

        df[col] = df[col].factorize()[0]

    return df
df_encoded = label_encode(df.copy())

df_encoded.head()
target = df_encoded.rate.fillna(df_encoded.rate.mean()) # Filling nan values in target by mean
plt.figure(figsize=(10,4))

sns.distplot(target)

plt.title('Target distribution')
corr = df_encoded.corr(method='kendall') # kendall since some of our features are ordinal.

df_encoded = df_encoded.drop(['rate'], axis=1).fillna(-1) #filling nan values by -1
plt.figure(figsize=(10,8))

sns.heatmap(corr, annot=True)
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split
def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))
# Helper function for scaling data to range [0, 1]

# Linear models are very sensitve to outliers, so let's scaled the data.

minmax = lambda x: (x - x.min())/(x.max() - x.min())
x_train, x_test, y_train, y_test = train_test_split(minmax(df_encoded),target, random_state=2)
lr = LinearRegression(n_jobs=-1)

svr = SVR()

rf = RandomForestRegressor(random_state=44, n_jobs=-1)

models = [lr, svr, rf]

for model in models:

    model.fit(x_train, y_train)

    pred_model = model.predict(x_test)

    print(f'The RMSE of {model.__class__.__name__} is {rmse(y_test, pred_model)}')
def plot_importances(model, cols):

    plt.figure(figsize=(12,6))

    f_imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=True)

    f_imp.plot(kind='barh')
plot_importances(rf, list(x_train))
preds_rf = rf.predict(x_test)

pd.Series(preds_rf).plot(kind='hist', label='predictions')

y_test.reset_index(drop=True).plot(kind='hist', label='true values')

plt.legend()
from sklearn.tree import export_graphviz

from IPython.display import Image
def convert_dot_to_png(model, max_depth=3, feature_names=list(x_train)):

    export_graphviz(model.estimators_[0], out_file='tree.dot', max_depth=max_depth, feature_names=feature_names, rounded=True)

    !dot -Tpng tree.dot -o tree.png
convert_dot_to_png(rf)

Image('tree.png')
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
rmse_scoring = make_scorer(rmse, greater_is_better=False)
param_grid = {'n_estimators':[20, 50, 100], 'max_features': [None, 'sqrt', 0.5]}

grid_search = GridSearchCV(estimator= rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring=rmse_scoring)

grid_search.fit(x_train, y_train)

None
grid_search_pred = grid_search.predict(x_test)

score_grid_search = rmse(y_test, grid_search_pred)

best_estimator = grid_search.best_estimator_

print(f'The best estimator is:\n {best_estimator} \n and the it\'s score is {score_grid_search}')
N_ITER=10

def run_n_iter(estimator, train, target,test, N_ITER=N_ITER):

    pred_n_iter = np.zeros((y_test.shape[0],), dtype='float')

    for i in range(N_ITER):

        estimator.set_params(random_state= i, )

        estimator.fit(train, target)

        pred_n_iter += estimator.predict(test) / N_ITER

    return pred_n_iter



pred_n_iter = run_n_iter(best_estimator, x_train, y_train, x_test)

print(f'The RMSE of {N_ITER} iterations is {rmse(y_test, pred_n_iter)}')
# Defining a helper function for count encoding

def count_encoding(df, cat_cols):

    for col in cat_cols:

        count = df[col].value_counts()

        new_colname = col + '_count'

        df[new_colname] = df[col].map(count)

    return df
cat_cols = x_train.columns[~x_train.columns.isin(['votes', 'approx_cost_2_people'])]
df_ce = count_encoding(df_encoded.copy(), cat_cols)

x_train_ce, x_test_ce = df_ce.iloc[x_train.index], df_ce.iloc[x_test.index]
new_features = [col for col in list(x_train_ce) if 'count' in col]

x_train_ce.loc[:, new_features].head()
pred_n_iter_2 = run_n_iter(best_estimator, x_train_ce, y_train, x_test_ce)

print(f'The RMSE of with engineered features is {rmse(y_test, pred_n_iter_2)}')
# Defining a helper function for One-hot encoding

# For computational reasons, we will limit the one-hot encoding to 

# features having unique values less or equal 100.

def ohe(df, max_nunique_vals=100, drop_encoded_feature=True):

    for col in list(df):

        if df[col].nunique() <= max_nunique_vals:

            dummies = pd.get_dummies(df[col].astype('category'), prefix=col)

            df = pd.concat([df, dummies], axis=1)

            if drop_encoded_feature:

                df.drop(col, axis=1, inplace=True)

    return df
df_ohe = ohe(df_encoded.copy())

x_train_ohe, x_test_ohe = df_ohe.iloc[x_train.index], df_ohe.iloc[x_test.index]
x_train_ohe.iloc[:, 8:].head()
best_estimator.set_params(random_state=5, n_jobs=-1) # we add n_jobs to speed up the computation.

best_estimator.fit(x_train_ohe, y_train)

pred_n_iter_3 = best_estimator.predict(x_test_ohe)

print(f'The RMSE of with engineered features is {rmse(y_test, pred_n_iter_3)}')
def get_lat_lon(df):

    # modified code from https://www.kaggle.com/shahules/zomato-complete-eda-and-lstm-model

    locations=pd.DataFrame({"Name":df['location'].unique()})

    locations['Name']=locations['Name'].apply(lambda x: "Bangalore " + str(x))

    lat=[]

    lon=[]

    geolocator=Nominatim(user_agent="app")

    for location in locations['Name']:

        location = geolocator.geocode(location)

        if location is None:

            lat.append(np.nan)

            lon.append(np.nan)

        else:    

            lat.append(location.latitude)

            lon.append(location.longitude)

    locations['lat']=lat

    locations['lon']=lon

    return locations
locations = get_lat_lon(df)

# Merging the coordiantes data to the original one

df_coord = df_encoded.copy()

unique_locations = df_coord.location.unique()

df_coord['lat'] = df_coord.location.replace(unique_locations, locations.lat)

df_coord['lon'] = df_coord.location.replace(unique_locations, locations.lon)

df_coord.iloc[:, -5:].head()
df_coord = df_coord.fillna(-1)

x_train_coord, x_test_coord = df_coord.iloc[x_train.index], df_coord.iloc[x_test.index]
best_estimator.fit(x_train_coord, y_train)

pred_n_iter_4 = best_estimator.predict(x_test_coord)

print(f'The RMSE of with coordinates features is {rmse(y_test, pred_n_iter_4)}')
# Intalling the package

# If not install in your kaggle docker, uncomment to install it.

# !pip install haversine

from haversine import haversine
# default unit of distance is in km

lal_bagh_coordinates= (12.9453, 77.5901)

df_coord['distance_to_lalbagh_km'] = [haversine((lat, lon), lal_bagh_coordinates)

                                    for (lat, lon) in df_coord[['lat','lon']].values]

df_coord.iloc[:, -5:].head() 
x_train_dist, x_test_dist = df_coord.iloc[x_train.index,:], df_coord.iloc[x_test.index,:]
best_estimator.fit(x_train_dist, y_train)

pred_n_iter_5 = best_estimator.predict(x_test_dist)

print(f'The RMSE of with coordinates features and distance to Lal Bagh is {rmse(y_test, pred_n_iter_5)}')
df_coord['nb_review'] = [len(val) for val in df['reviews_list']]

x_train_dist_coord_review, x_test_dist_coord_review = df_coord.iloc[x_train.index,:], df_coord.iloc[x_test.index,:]

df_coord.iloc[:, -5:].head()
best_estimator.fit(x_train_dist_coord_review, y_train)

pred_n_iter_6 = best_estimator.predict(x_test_dist_coord_review)

print(f'The RMSE coordinates features, distance to Lal Bagh and # of reviews is {rmse(y_test, pred_n_iter_6)}')
import lightgbm as lgb

import catboost as cat

import xgboost as xgb
def run_models(models, x_train, y_train, x_test, y_test):

    preds = np.zeros((x_test.shape[0], 3), dtype='float')

    for i, model in enumerate(models):

        model.fit(x_train, y_train)

        tmp = model.predict(x_test)

        print(f'The RMSE of {model.__class__.__name__} is {rmse(y_test, tmp)}')

        preds[:, i] = tmp

    return preds
clf_lgb = lgb.LGBMRegressor(random_state=97)

clf_cat = cat.CatBoostRegressor(random_state=2019, verbose=0)

clf_xgb = xgb.XGBRegressor(random_state=500)

models = [clf_lgb, clf_cat, clf_xgb]

preds_models = run_models(models, x_train, y_train, x_test, y_test)
preds_models_2 = run_models(models, x_train_dist, y_train, x_test_dist, y_test)
from scipy.stats import ks_2samp
GBM_models = ['LGB', 'CatBoost', 'XGBoost']

for i in range(2):

    print('The p-value between {0} and {1} is {2}'.format(GBM_models[0], GBM_models[i+1], ks_2samp(preds_models_2[:, 0],

                                                                                                   preds_models_2[:, i + 1])[1]))
print('The p-value between {0} and {1} is {2}'.format('Random forest regressor', GBM_models[0], 

                                                      ks_2samp(preds_rf,preds_models_2[:, 0])[1]))
blend_1 = 0.95*preds_rf + 0.05*preds_models_2[:, 0]

blend_2 = 1.05*preds_rf - 0.05*preds_models_2[:, 0]

print(f'The p-value of the blended predictions (1) between RF and LGBM is {rmse(y_test, blend_1)}')

print(f'The p-value of the blended predictions (2) between RF and LGBM is {rmse(y_test, blend_2)}')
from sklearn.model_selection import KFold
def cv(model, train, target, n_splits=5):

    oof = np.zeros((train.shape[0],), dtype='float')

    kf = KFold(n_splits=n_splits, shuffle=True)

    scores = pd.Series(np.zeros((n_splits,)))

    for i, (tr_idx, te_idx) in enumerate(kf.split(train, target)):

        x_tr, x_te = train.loc[tr_idx], train.loc[te_idx]

        y_tr, y_te = target[tr_idx], target[te_idx]

        model.fit(x_tr, y_tr)

        tmp = model.predict(x_te)

        oof[te_idx] = tmp

        scores[i] = rmse(y_te, tmp)

        print('Fold {} score {}'.format(i, scores[i]))

    return oof, scores
# The cross-validation is done on the original data + engineered features (count, coordinates, distance)

oof, scores = cv(best_estimator, df_coord.drop('nb_review', axis=1), target)
print('Mean score {}, STD {}'.format(scores.mean(),scores.std()))