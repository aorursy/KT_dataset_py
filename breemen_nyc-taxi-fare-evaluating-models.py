# load some default Python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')
# read data in pandas dataframe
df_train =  pd.read_csv('../input/train.csv', nrows = 50_000, parse_dates=["pickup_datetime"])
# define bounding box
BB = (-75, -72.9, 40, 41.8)

# this function will be used with the test set below
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            
# This function is based on https://stackoverflow.com/questions/27928/
# calculate-distance-between-two-latitude-longitude-points-haversine-formula 
# Returns distance in miles
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
# add distance in miles
df_train['distance_miles'] = distance(df_train.pickup_latitude, df_train.pickup_longitude, \
                                      df_train.dropoff_latitude, df_train.dropoff_longitude)
# add distance to NYC center
nyc = (-74.0063889, 40.7141667)
df_train['distance_to_center'] = distance(nyc[1], nyc[0], df_train.pickup_latitude, df_train.pickup_longitude)
# add year
df_train['year'] = df_train.pickup_datetime.apply(lambda t: t.year)
# add hour
df_train['hour'] = df_train.pickup_datetime.apply(lambda t: t.hour)
# add weekday 0:monday, 6:sunday
df_train['weekday'] = df_train.pickup_datetime.apply(lambda t: t.weekday())
print('Old size: %d' % len(df_train))
# remove non-zero fare
df_train = df_train[df_train.fare_amount>=0]
# remove missing data
df_train = df_train.dropna(how = 'any', axis = 'rows')
# remove datapoints outside boundingbox near NYC
df_train = df_train[select_within_boundingbox(df_train, BB)]
# remove datapoints with zero distance traveled
df_train = df_train[df_train.distance_miles >= 0]
# remove datapoints with zero passengers
df_train = df_train[df_train.passenger_count > 0]
print('New size: %d' % len(df_train))
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'distance_miles', 'distance_to_center', 'year', 'weekday', 'hour']
X = df_train[features].values
y = df_train['fare_amount'].values
# create training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# define some handy analysis support function
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

def calculate_kfold_rmse(model, X, y, nfolds):
    kf = KFold(n_splits=nfolds, shuffle=False, random_state=None)
    return np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")).mean()


def plot_prediction_analysis(y, y_pred, figsize=(10,4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))
    
    axs[1].hist(y-y_pred, bins=50)
    avg = np.mean(y-y_pred)
    std = np.std(y-y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title('Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(avg, std))
    
    if title!='':
        fig.suptitle(title)
        
        
# some handy function to see how sensitive the model is to the selection
# of the training and test set
def plot_rmse_analysis(model, X, y, N=400, test_size=0.25, figsize=(10,4), title=''):
    rmse_train, rmse_test = [], []
    for i in range(N):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    g = sns.jointplot(np.array(rmse_train), np.array(rmse_test), kind='scatter', stat_func=None, size=5)
    g.set_axis_labels("RMSE training ($\mu$={:.2f})".format(np.mean(rmse_train)), 
                      "RMSE test ($\mu$={:.2f})".format(np.mean(rmse_test)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('{} (N={}, test_size={:0.2f})'.format(title, N, test_size))
    
def plot_learning_curve(model, X_train, X_test, y_train, y_test, nsteps=1, figsize=(6, 5), title=''):
    train_error, test_error = [], []
    number_of_samples = []
    m_samples = X_train.shape[0]
    for m in range(int(m_samples/nsteps), m_samples+1, int(m_samples/nsteps)):
        number_of_samples.append(m)
        model.fit(X_train[:m,:], y_train[:m])
        y_train_pred = model.predict(X_train[:m,:])
        train_error.append(np.sqrt(mean_squared_error(y_train[:m], y_train_pred)))
        y_test_pred = model.predict(X_test)
        test_error.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    plt.figure(figsize=figsize)
    plt.plot(number_of_samples, train_error, label='Training data')
    plt.plot(number_of_samples, test_error, label='Test data')
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')
    plt.legend()
    if title!='':
        plt.title(title)
# prepare python dictionary with models to test
models = {}
# Add linear regression model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

models['linear_model'] = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
# Add linear model with polynomial features. Use Ridge for L2 regularization
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

models['polynomial'] = Pipeline((
        ("standard_scaler", StandardScaler()),    
        ("poly_features", PolynomialFeatures(degree=2)),
        ("ridge", Ridge()),
    ))
# Add KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

models['kneighbors'] = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("kneighborsregressor", KNeighborsRegressor()),
    ))
# Add RandomForestRegressor with several different parameters
from sklearn.ensemble import RandomForestRegressor

models['random_forest_regressor_n10'] = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=10)
models['random_forest_regressor_n100'] = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10)
# Add RandomForestRegressor with several different parameters
from sklearn.ensemble import GradientBoostingRegressor

models['gradient_boosting_n10'] = GradientBoostingRegressor(max_depth=2, n_estimators=10, learning_rate=1.0)
models['gradient_boosting_n100'] = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=0.2)
from xgboost import XGBRegressor

models['xgboost10'] = XGBRegressor(n_estimators=10, max_depth=3)
models['xgboost100'] = XGBRegressor(n_estimators=100, max_depth=3)
nfolds = 10
scores = []
print("Starting evaluating all models: datapoints = {}, nfolds = {}".format(X.shape[0], nfolds))
for name, model in models.items():
    print('\n... calculating {} ...'.format(name))
    %time score = calculate_kfold_rmse(model, X, y, nfolds)
    scores.append((name, score))
    
print("\n")
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=False)
print("rmsr - model (nfolds={})".format(nfolds))
print("============================================")
for r in sorted_scores:
    print("{:0.4f} - {}".format(r[1], r[0]))
name_best_model, best_model = sorted_scores[0][0], models[sorted_scores[0][0]]

best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title='{} - Trainingset'.format(name_best_model))

y_test_pred = best_model.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title='{} - Testset'.format(name_best_model))
plot_rmse_analysis(best_model, X, y, N=100)
plot_learning_curve(best_model, X_train, X_test, y_train, y_test, nsteps=20, title=name_best_model)
# read test data
df_test =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])
# add distance in km
df_test['distance_miles'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                                     df_test.dropoff_latitude, df_test.dropoff_longitude)
# add distance to NYC center
df_test['distance_to_center'] = distance(nyc[1], nyc[0], df_test.pickup_latitude, df_test.pickup_longitude)
# add year
df_test['year'] = df_test.pickup_datetime.apply(lambda t: t.year)
# add hour
df_test['hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)
# add weekday 0:monday, 6:sunday
df_test['weekday'] = df_test.pickup_datetime.apply(lambda t: t.weekday())
# define dataset
XTEST = df_test[features].values

filename = 'submission_best_model_{}.csv'.format(name_best_model)

y_pred_final = best_model.predict(XTEST)

submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv(filename, index = False)
submission
