from IPython.display import Image
Image("/kaggle/input/germany-car-sales-map/germany_car_sales.png")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
df = pd.read_csv('/kaggle/input/used-cars-database/autos.csv', sep=',', header=0, encoding='cp1252')
df.info()
df.isna().sum()
df.dropna(inplace = True, axis = 0, subset=['vehicleType','gearbox','model','fuelType'])
df.drop(['notRepairedDamage', 'name'], axis = 1, inplace = True)
print(df.duplicated().sum())
df.drop_duplicates(inplace = True)
categorical = df.select_dtypes(include=['object'])
categorical.info()
categorical.describe().transpose() #Transposing make the visualisation a bit better
to_binary = [
    'seller',
    'offerType',
    'abtest',
    'gearbox',
]
for col in to_binary:
    print(col)
    print(df[col].value_counts() / len(df))
    print('-' * 70)
df.drop(['seller', 'offerType'], axis = 1, inplace = True)
# Update to_binary array
to_binary = [
    'abtest',
    'gearbox',
]
df = pd.get_dummies(df, columns = to_binary, drop_first = True)
df.info(verbose = True, memory_usage = True)
# Check date formating and convert to proper dateTime format
to_date = ['dateCrawled', 'lastSeen', 'dateCreated']
df[to_date].head(1)
for col in to_date:
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")
g = sns.countplot(
    df['vehicleType'],
    order = df['vehicleType'].value_counts().index,
)
g.set_xticklabels(g.get_xticklabels(), rotation = 45)
plt.show()
vehicleType_dummies = pd.get_dummies(df['vehicleType'])
vehicleType_dummies.info(max_cols = 0,memory_usage = True)
g = sns.countplot(
    df['fuelType'],
    order = df['fuelType'].value_counts().index,
)
g.set_xticklabels(g.get_xticklabels(), rotation = 45)
plt.show()
df['fuelType'].value_counts()/len(df)
df = df[ df['fuelType'].isin(['benzin', 'diesel']) ]
df = pd.get_dummies(df, columns = ['fuelType'], drop_first = True)
g = sns.countplot(
    df['model'],
    order = df['model'].value_counts().index,
    log = True
)
g.set_xticklabels([])
plt.show()
model_dummies = pd.get_dummies(df['model'])
model_dummies.info(memory_usage = True)
print(model_dummies.shape)
g = sns.countplot(
    df['brand'],
    order = df['brand'].value_counts().index,
    log = True
)
g.set_xticklabels(g.get_xticklabels(), rotation = 90)
#g.set_xticklabels([])
plt.show()
brand_dummies = pd.get_dummies(df['brand'])
brand_dummies.info(max_cols = 0, memory_usage = True)
df.drop(['vehicleType','model','brand'], axis =1, inplace = True)
numerical = df.select_dtypes(exclude=['datetime64'])
numerical.info()
numerical.describe().transpose()
df['nrOfPictures'].unique()
df.drop(['nrOfPictures'], axis = 1, inplace = True)
numerical = df.select_dtypes(exclude=['datetime64'])
sns.heatmap(numerical.corr())
g  = plt.hist(df['yearOfRegistration'], bins = 20, rwidth = 0.8)
df['yearOfRegistration'].skew()
min(df['yearOfRegistration'])
df = df[df['yearOfRegistration'] > 1970]
plt.hist(df['price'], rwidth = 0.8, bins = 40)
plt.show()
len(df[df['price'] > 1e5])
df = df[df['price'] < 1e5]
plt.hist(df['price'], rwidth = 0.8, bins = 40, log = True)
plt.show()
df['postalCode'].value_counts()
import pgeocode
nomi = pgeocode.Nominatim('de') # Deuchland AKA germany
%%time
nomi.query_postal_code("10115")['place_name'] # Testing
postal_codes = df['postalCode'].unique()
#Takes a while
postal_places = {}
for postal in postal_codes:
    postal_places[postal] = nomi.query_postal_code(str(postal))['state_name']
    
df['states'] = df["postalCode"].apply(lambda s: postal_places[s])
df['states'].value_counts()
import geopandas as gpd

#
map_df = gpd.read_file('/kaggle/input/germany-states/mittel.geo.json')
map_df.plot()
pivot = df.pivot_table(index='states', values='price', aggfunc=np.sum).sort_values("price")
pivot.plot(kind='bar')
merged = map_df.set_index('name').join(pivot)
fig, ax = plt.subplots(1, figsize = (10, 6))
merged.plot( column = 'price', cmap='OrRd', linewidth = 0.8, ax = ax, edgecolor ='0.8', )
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=float(pivot.min()), vmax=float(pivot.max())))
sm._A = []
cbar = fig.colorbar(sm)
ax.annotate('Sorce: Kaggle',
           xy=(0.1, .08), xycoords='figure fraction',
           horizontalalignment='left', verticalalignment='top',
           fontsize=10, color='#555555')
plt.show()
states_dummies = pd.get_dummies(df['states'])
states_dummies.info(max_cols = 0, memory_usage = True)
df['kilometer'].value_counts()
g = sns.countplot(df['kilometer'])
g.set_xticklabels(g.get_xticklabels(), rotation = 45)
plt.show()
sns.countplot(df['monthOfRegistration'])
plt.show()
plt.hist(df['dateCrawled'], bins = 20, rwidth = 0.8, alpha = 0.3)
plt.hist(df['lastSeen'], bins = 20, rwidth = 0.8, alpha = 0.3)
plt.show()
df['lastSeen'].value_counts().head(5)
len(df[df['lastSeen'] > '2016-04-01']) / len(df)
df.loc[df['lastSeen'] >= '2016-04-01', 'isRecent'] = 1
df.loc[df['lastSeen'] < '2016-04-01', 'isRecent'] = 0
df['isRecent'].value_counts()
df['maturity'] = df['lastSeen'] - df['dateCrawled']
df['maturity'].describe()
len(df[df['maturity'].dt.total_seconds() > 0]) / len (df)
df['maturity'] = df['maturity'].dt.total_seconds()
df.info()
df.drop(['dateCrawled', 'dateCreated', 'lastSeen', 'states'], axis = 1, inplace = True)
df = pd.concat([df, vehicleType_dummies], axis=1)
df = pd.concat([df, states_dummies], axis=1)
df = pd.concat([df, vehicleType_dummies], axis=1)
df.info()
df.dropna(inplace = True)
from sklearn.model_selection import train_test_split

features = df.drop(['price', 'maturity', 'isRecent'], axis =1)

target  = df['price']
#target = np.log1p(target['price'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(lr.score(x_test, y_test))
print(mean_squared_error(y_pred, y_test))
sns.residplot(y_test, y_pred, color="orange", scatter_kws={"s": 3})
plt.show()
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

adar = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators = 600)
adar.fit(x_train, y_train)
y_pred = adar.predict(x_test)
adar.score(x_test, y_test)
df = pd.concat([df, model_dummies], axis=1)
df.dropna(inplace = True)
features = df.drop(['price', 'maturity', 'isRecent'], axis =1)

target  = df['price']
#target = np.log1p(target['price'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)

lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(lr.score(x_test, y_test))
print(mean_squared_error(y_pred, y_test))
adar_full = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators = 600)
adar_full.fit(x_train, y_train)
y_pred = adar_full.predict(x_test)
adar_full.score(x_test, y_test)