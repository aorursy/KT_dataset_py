%load_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics
# Set the plot sizes
set_plot_sizes(12,14,16)
PATH = "../input/"
df_raw = pd.read_csv(f'{PATH}train.csv', nrows=100000, parse_dates=['pickup_datetime'], dtype={'passenger_count': 'int8', 'fare_amount': 'float16'}, 
                     usecols=['fare_amount', 'pickup_datetime','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count'])
df_raw_test = pd.read_csv(f'{PATH}test.csv', parse_dates=['pickup_datetime'], dtype={'passenger_count': 'int8'})
# Expands the summary tables if there are a lot of columns androws
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
# Shows the last 5 rows of the traning set
display_all(df_raw.tail().T)
# Shows summary of training set
display_all(df_raw.describe(include='all').T)
# Shows summary of test set
display_all(df_raw_test.describe(include='all').T)
plt.figure(figsize=(20, 4))
df_raw['pickup_datetime'].groupby([df_raw["pickup_datetime"].dt.year, df_raw["pickup_datetime"].dt.month]).count().plot(kind="bar")
plt.title('Traing Set Rides per Month and Year')
plt.show()
plt.figure(figsize=(20, 4))
df_raw_test['pickup_datetime'].groupby([df_raw_test["pickup_datetime"].dt.year, df_raw_test["pickup_datetime"].dt.month]).count().plot(kind="bar")
plt.title('Test Set Rides per Month and Year')
plt.show()
df_raw[df_raw['fare_amount'] < 0]
# Large negative longitudes
df_raw[df_raw['pickup_longitude'] < -75]
# Large positive longitudes
df_raw[df_raw['pickup_longitude'] > -73]
# Small positive latitudes
df_raw[df_raw['pickup_latitude'] < 40]
# Large positive latitudes
df_raw[df_raw['pickup_latitude'] > 42]
df_raw.shape
df_raw = df_raw[df_raw['pickup_longitude'] > -76]
df_raw = df_raw[df_raw['pickup_longitude'] < -73]
df_raw = df_raw[df_raw['pickup_latitude'] > 40]
df_raw = df_raw[df_raw['pickup_latitude'] < 44]
df_raw.shape
# Converts all strings to categorical features
train_cats(df_raw)
# Splits dates into subcomponenets
add_datepart(df_raw, 'pickup_datetime')
df_raw.info()
# Splits the data into independent and dependent features and keeps a column 'nas' that keeps track of features that had missing values and had to be imputed
df, y, nas = proc_df(df_raw, 'fare_amount')
m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, oob_score=True, n_jobs=-1)
m.fit(df, y)
# Calculate Feature Importance
fi = rf_feat_importance(m, df); fi[:10]
# Shows the table above visually
fi.plot('cols', 'imp', figsize=(10,6), legend=False)
plt.title('Feature Importance by Feature');
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
# Same visualisation but easier to see which feature contributes how much
plot_fi(fi[:30])
plt.title('Feature Importance by Feature');
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
plt.title('Feature Similarities')
plt.show()