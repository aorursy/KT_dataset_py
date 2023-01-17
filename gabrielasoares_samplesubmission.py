import numpy as np 

import pandas as pd



# Visualizations

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

palette = sns.color_palette('Paired', 10)



# Set random seed 

RSEED = 100
train = pd.read_csv('../input/train_MV.csv')

train.head()
train.shape
train.drop('key', axis=1, inplace=True)

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train.dtypes
# Remove na

train = train.dropna()

train.shape
train.describe()
sns.distplot(train['fare_amount']);

plt.title('Distribution of Fare');
print(f"There are {len(train[train['fare_amount'] < 0])} negative fares.")

print(f"There are {len(train[train['fare_amount'] == 0])} $0 fares.")

print(f"There are {len(train[train['fare_amount'] > 100])} fares greater than $100.")
#remove outliers

#keep only fares between 2.5 and 100

train = train[train['fare_amount'].between(left = 2.5, right = 100)]
#number of passengers

train['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');

plt.title('Passenger Counts'); plt.xlabel('Number of Passengers'); plt.ylabel('Count');

#should we remove the outliers based on number of passengers?
# Remove latitude and longtiude outliers

train = train.loc[train['pickup_latitude'].between(40, 42)]

train = train.loc[train['pickup_longitude'].between(-75, -72)]

train = train.loc[train['dropoff_latitude'].between(40, 42)]

train = train.loc[train['dropoff_longitude'].between(-75, -72)]



print(f'New number of observations: {train.shape[0]}')
train = train.loc[train['passenger_count'].between(0, 6)]
# Absolute difference in latitude and longitude

train['abs_lat_diff'] = (train['dropoff_latitude'] - train['pickup_latitude']).abs()

train['abs_lon_diff'] = (train['dropoff_longitude'] - train['pickup_longitude']).abs()





sns.lmplot('abs_lat_diff', 'abs_lon_diff', fit_reg = False,

           data = train.sample(10000, random_state=RSEED));

plt.title('Absolute latitude difference vs Absolute longitude difference');
no_diff = train[(train['abs_lat_diff'] == 0) & (train['abs_lon_diff'] == 0)]

no_diff.shape
def minkowski_distance(x1, x2, y1, y2, p):

    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)

train['manhattan'] = minkowski_distance(train['pickup_longitude'], train['dropoff_longitude'],

                                       train['pickup_latitude'], train['dropoff_latitude'], 1)



train['euclidean'] = minkowski_distance(train['pickup_longitude'], train['dropoff_longitude'],

                                       train['pickup_latitude'], train['dropoff_latitude'], 2)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



lr = LinearRegression()
# Split data

X_train, X_valid, y_train, y_valid = train_test_split(train, np.array(train['fare_amount']),

                                                      random_state = RSEED, test_size = 300_000)
lr.fit(X_train[['abs_lat_diff', 'abs_lon_diff', 'passenger_count']], y_train)



print('Intercept', round(lr.intercept_, 4))

print('abs_lat_diff coef: ', round(lr.coef_[0], 4), 

      '\tabs_lon_diff coef:', round(lr.coef_[1], 4),

      '\tpassenger_count coef:', round(lr.coef_[2], 4))
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore', category = RuntimeWarning)
def metrics(train_pred, valid_pred, y_train, y_valid):

    """Calculate metrics:

       Root mean squared error and mean absolute percentage error"""

    

    # Root mean squared error

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    

    # Calculate absolute percentage error

    train_ape = abs((y_train - train_pred) / y_train)

    valid_ape = abs((y_valid - valid_pred) / y_valid)

    

    # Account for y values of 0

    train_ape[train_ape == np.inf] = 0

    train_ape[train_ape == -np.inf] = 0

    valid_ape[valid_ape == np.inf] = 0

    valid_ape[valid_ape == -np.inf] = 0

    

    train_mape = 100 * np.mean(train_ape)

    valid_mape = 100 * np.mean(valid_ape)

    

    return train_rmse, valid_rmse, train_mape, valid_mape

def evaluate(model, features, X_train, X_valid, y_train, y_valid):

    """Mean absolute percentage error"""

    

    # Make predictions

    train_pred = model.predict(X_train[features])

    valid_pred = model.predict(X_valid[features])

    

    # Get metrics

    train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,

                                                             y_train, y_valid)

    

    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')

    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')
evaluate(lr, ['abs_lat_diff', 'abs_lon_diff', 'passenger_count'], 

        X_train, X_valid, y_train, y_valid)
test = pd.read_csv('../input/test_MV.csv')
#calculate the features for the test set

# Absolute difference in latitude and longitude

test['abs_lat_diff'] = (test['dropoff_latitude'] - test['pickup_latitude']).abs()

test['abs_lon_diff'] = (test['dropoff_longitude'] - test['pickup_longitude']).abs()



test['manhattan'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],

                                       test['pickup_latitude'], test['dropoff_latitude'], 1)



test['euclidean'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],

                                       test['pickup_latitude'], test['dropoff_latitude'], 2)
preds = lr.predict(test[['abs_lat_diff', 'abs_lon_diff', 'passenger_count']])
sub = pd.DataFrame({'key': test.key, 'fare_amount': preds})

sub.to_csv('sub_lr_simple.csv', index = False)