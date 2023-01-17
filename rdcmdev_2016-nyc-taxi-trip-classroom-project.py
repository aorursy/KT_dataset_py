# used to work with dataframes(excels like tables)
import pandas as pd

# used to work properly with numerical operations
import numpy as np

# used to avoid deprecated messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
!ls ../input
df_train = pd.read_csv('../input/train.csv')
df_train.head()
plg, plt = 'pickup_longitude', 'pickup_latitude'
dlg, dlt = 'dropoff_longitude', 'dropoff_latitude'
pdt, ddt = 'pickup_datetime', 'dropoff_datetime'
df_train.dropna(inplace=True)
mean, std_deviation = np.mean(df_train['trip_duration']), np.std(df_train['trip_duration'])
df_train = df_train[df_train['trip_duration'] <= mean + 2 * std_deviation]
df_train = df_train[df_train['trip_duration'] >= mean - 2 * std_deviation]
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def haversine_distance(x):
    x1, y1 = np.float64(x[plg]), np.float64(x[plt])
    x2, y2 = np.float64(x[dlg]), np.float64(x[dlt])    
    return haversine(x1, y1, x2, y2)
%%time
df_train['distance'] = df_train[[plg, plt, dlg, dlt]].apply(haversine_distance, axis=1)
df_train.head()
from datetime import datetime

df_train[pdt] = df_train[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df_train[ddt] = df_train[ddt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df_train['month'] = df_train[pdt].apply(lambda x : x.month)
df_train['weekDay'] = df_train[pdt].apply(lambda x : x.weekday())
df_train['dayMonth'] = df_train[pdt].apply(lambda x : x.day)
df_train['pickupTimeMinutes'] = df_train[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
df_train.head()
df_train.drop(['id', pdt, ddt, dlg, dlt, 'store_and_fwd_flag'], inplace=True, axis=1)
df_train.head()
df_train = df_train[
    [
        plg, 
        plt, 
        'distance', 
        'month', 
        'dayMonth', 
        'weekDay', 
        'pickupTimeMinutes', 
        'passenger_count', 
        'vendor_id', 
        'trip_duration'
    ]
]
df_train.head()
from sklearn.model_selection import train_test_split

X, y = df_train.iloc[:, :-1], df_train.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=4305)
X_train.shape, y_train.shape, X_val.shape, y_val.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler = StandardScaler().fit(X_val)
X_val = scaler.transform(X_val)
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=3, shuffle=True, random_state=4305)
models = {}
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(
    activation='relu',
    alpha=0.0001, 
    batch_size='auto',
    beta_1=0.9,
    beta_2=0.999, 
    early_stopping=False, 
    epsilon=1e-08,
    hidden_layer_sizes=(3, 3),
    learning_rate='adaptive',
    learning_rate_init=0.001, 
    max_iter=1000, 
    momentum=0.5,
    nesterovs_momentum=True,
    power_t=0.5,
    random_state=None,
    shuffle=True, 
    solver='adam',
    tol=0.0001, 
    validation_fraction=0.1,
    verbose=False, 
    warm_start=True
)

models['mlp'] = mlp
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(
    criterion='mse', 
    max_depth=17, 
    max_features=None,       
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,  
    min_impurity_split=None, 
    min_samples_leaf=1,      
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,  
    presort=False, 
    random_state=None,
    splitter='best'
)

models['dtree'] = dtree
from sklearn.linear_model import LinearRegression

lreg = LinearRegression(
    copy_X=True, 
    fit_intercept=True, 
    n_jobs=1, 
    normalize=False
)

models['lreg'] = lreg
models
from sklearn.metrics import make_scorer

def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.square(np.subtract(np.log1p(y_true), np.log1p(y_pred)))))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
%%time
for model in models.values():    
    model.fit(X_train, y_train)
for model_name, model in models.items():    
    score = cross_val_score(model, X_val, y_val, cv=kf)
    print(f'model name: {model_name} \t score: {score} \t mean_score: {np.mean(score)}')
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test['distance'] = df_test[[plg, plt, dlg, dlt]].apply(haversine_distance, axis=1)
df_test[pdt] = df_test[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df_test['month'] = df_test[pdt].apply(lambda x : x.month)
df_test['weekDay'] = df_test[pdt].apply(lambda x : x.weekday())
df_test['dayMonth'] = df_test[pdt].apply(lambda x : x.day)
df_test['pickupTimeMinutes'] = df_test[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
df_test.drop(['pickup_datetime', dlg, dlt, 'store_and_fwd_flag'], inplace=True, axis=1)
df_test = df_test[['id', plg, plt, 'distance', 'month', 'dayMonth', 'weekDay', 'pickupTimeMinutes', 'passenger_count', 'vendor_id']]
df_test.head()
X_id, X_test = df_test.iloc[:, 0], df_test.iloc[:, 1:]
X_id.shape, X_test.shape
scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
models_output = {}
for model_name, model in models.items():
    models_output[model_name] = model.predict(X_test)
for model_name, model_output in models_output.items():
    df_output = pd.DataFrame({'id' : X_id, 'trip_duration': model_output})
    df_output.to_csv(model_name + '.csv', index=False)