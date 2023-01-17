# Importing libraries
from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display
# importing data (only 10 million data points)
PATH = '../input'
df_raw = pd.read_csv(f'{PATH}/train.csv', nrows=10000000)
# Function to set display options
def display_all(df):
    with pd.option_context('display.max_rows',1000):
        with pd.option_context('display.max_columns',1000):
            display(df)
display_all(df_raw.head(5))
add_datepart(df_raw,'pickup_datetime',drop=True, time=True)
display_all(df_raw.head(5))
def distance(data):
    data['longitutde_traversed'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['latitude_traversed'] = (data.dropoff_latitude - data.pickup_latitude).abs()
distance(df_raw)
display_all(df_raw.head(2).T)
df_raw.isnull().sum()
df_raw.dropna(axis=0, how='any', inplace=True)
df_raw.shape
key = df_raw.key
df_raw.drop('key', axis=1, inplace = True)
df_raw.passenger_count.value_counts()
df_raw = df_raw[(df_raw.passenger_count>0)&(df_raw.passenger_count<10)]
len(df_raw)
df_raw.reset_index(drop=True, inplace=True)
outliers = []
# For each feature find the data points with extreme high or low values
for feature in df_raw.keys():
    Q1 = np.percentile(df_raw[feature],25,axis=0)
    Q3 = np.percentile(df_raw[feature],75,axis=0)
    step = 2*(Q3-Q1)
    feature_outlier = df_raw[~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))]
    outliers += feature_outlier.index.tolist()
len(outliers)/len(df_raw)
outliers = []
# For each feature find the data points with extreme high or low values
for feature in ['longitutde_traversed','latitude_traversed']:
    Q1 = np.percentile(df_raw[feature],25,axis=0)
    Q3 = np.percentile(df_raw[feature],75,axis=0)
    step = 10*(Q3-Q1)
    feature_outlier = df_raw[~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))]
    outliers += feature_outlier.index.tolist()
len(outliers)/len(df_raw)
df = df_raw.drop(df_raw.index[outliers]).reset_index(drop = True)
len(df)
y = df_raw.fare_amount
df_raw.drop('fare_amount', axis=1, inplace = True)
X_train, X_valid, y_train, y_valid = train_test_split(df_raw, y, test_size = 10000)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print(res)
set_rf_samples(10000)
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m,X_train)
fi[:10]
def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,8),legend=False)
plot_fi(fi)
test_set = pd.read_csv(f'{PATH}/test.csv')
test_key = test_set.key
test_set.drop('key', axis = 1, inplace = True)
add_datepart(test_set,'pickup_datetime',drop=True, time=True)
distance(test_set)
test_predictions = m.predict(test_set)
submission = pd.DataFrame({'key': test_key, 
                           'fare_amount': test_predictions})
submission.to_csv('submissions.csv', index=False)