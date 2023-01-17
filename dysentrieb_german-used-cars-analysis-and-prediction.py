# Import librariers



# Basic libraries

import numpy as np

import pandas as pd



# Plot data

import matplotlib.pyplot as plt

%matplotlib inline



# Others

from scipy.stats import kde

import pickle



# Sklearn

#from sklearn.preprocessing import normalize

#from sklearn.preprocessing import StandardScaler

#from sklearn.preprocessing import scale



from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
# Read in data from csv files downloaded from https://www.kaggle.com/orgesleka/used-cars-database and save to dataframe

df_autos = pd.read_csv('autos.csv', encoding='latin8')
# Print data about data

print('Exact number of entries/rows in df_autos: {}'.format(df_autos.shape[0]))

print('Number of features/columns in df_autos: {}'.format(df_autos.shape[1]))

print('Feature/Column-names in df_autos: {}'.format(df_autos.columns.values))
df_autos.shape
# Basic statistics about the numeral values:

df_autos.describe()
# Have a first look at the data:

df_autos.head()
# Check column values for relevance and drop irrelevant columns

print(np.unique(df_autos['offerType'],return_counts=True))

print(np.unique(df_autos['abtest'],return_counts=True))

print(np.unique(df_autos['nrOfPictures'],return_counts=True))

print(np.unique(df_autos['seller'],return_counts=True))
# Having a more detailed look on the NaNs by columns:

df_autos.isnull().sum(axis = 0)
# Show registration years with counts and plot histogram

print(np.unique(df_autos['yearOfRegistration'],return_counts=True))

df_autos['yearOfRegistration'].hist()
# As the first car was built in 1885, check every year separately to manually inspect whether the entries make sense or not.

for year in range(1885, 1950):

    print('Year: {}'.format(year))

    print(df_autos[df_autos['yearOfRegistration'] == year].name.values)
# Plot boxplot of motor power

plt.figure(figsize=(5,5))



#test = df_autos_locs[df_autos_locs['powerPS'] < df_autos_locs['powerPS'].quantile(0.999)]

_ = plt.boxplot(df_autos['powerPS'])



print('0.1-quantile at: {}'.format(df_autos['powerPS'].quantile(0.1)))

print('0.999-quantile at: {}'.format(df_autos['powerPS'].quantile(0.999)))
# Plot boxplot of prices

plt.figure(figsize=(5,5))



#test = df_autos_locs[df_autos_locs['powerPS'] < df_autos_locs['powerPS'].quantile(0.999)]

_ = plt.boxplot(df_autos['price'])



print('0.001-quantile at: {}'.format(df_autos['price'].quantile(0.001)))

print('0.01-quantile at: {}'.format(df_autos['price'].quantile(0.01)))

print('0.99-quantile at: {}'.format(df_autos['price'].quantile(0.99)))

print('0.999-quantile at: {}'.format(df_autos['price'].quantile(0.999)))

print('0.999-quantile at: {}'.format(df_autos['price'].quantile(0.9999)))
# Check data types of all columns:

for col in df_autos.columns:

    print(col + ': ' + str(type(df_autos[col].values[0])))
# Drop rows that are requests and keep offers. Check whether 12 rows are removed.

print(df_autos.shape[0])

df_autos = df_autos[df_autos.offerType != 'Gesuch']

print(df_autos.shape[0])
# Drop all pf the above named columns. Check number of columns.

print(df_autos.shape[1])

remove_cols = ['offerType', 'abtest', 'nrOfPictures', 'seller', 'dateCrawled',

               'monthOfRegistration', 'dateCreated', 'lastSeen', 'name']

df_autos = df_autos.drop(columns=remove_cols)

print(df_autos.shape[1])
# Drop rows with years that do not make sense.

print(len(df_autos))

df_autos = df_autos[df_autos['yearOfRegistration'] > 1910]

df_autos = df_autos[df_autos['yearOfRegistration'] < 2017]

print(len(df_autos))
print(len(df_autos))

df_autos = df_autos[df_autos['powerPS'] > 0]

df_autos = df_autos[df_autos['powerPS'] < 1000]

print(len(df_autos))
print(len(df_autos))

df_autos = df_autos[df_autos['price'] < 1000000]

print(len(df_autos))
# Display data again:

df_autos.head()
# Load in dataset

df_zips = pd.read_csv('PLZ.tab', encoding='UTF-8', sep='\t')

df_zips.head(50)
# Drop the unneeded column #loc_id

df_zips = df_zips.drop(columns='#loc_id')

# Merge with the auto-dataset on 'postalCode' and 'plz' and create a new DataFrame from it

df_autos_locs = pd.merge(df_autos, df_zips, left_on='postalCode', right_on='plz')

df_autos_locs = df_autos_locs.drop(columns='plz')

df_autos_locs.rename(columns={'Ort': 'city'})

df_autos_locs.head()
print(df_autos.shape)

print(df_autos_locs.shape)
# Define function to count the unique values of a dataframe and return in a sorted fashion. Also drop NaNs before.

def count_col_vals(df, col):

    # Drop NaNs and return number of dropped rows and remaining rows

    len_before = len(df)

    df = df.dropna(subset=[col])

    print('Dropped {} NaN-Rows for column {}.'.format(len_before - len(df), col))

    print('Number of remaining rows is {}.'.format(len(df)))

    # Sort and count

    data_labels, data_counts = np.unique(df[col],return_counts=True)

    d = {'labels': data_labels, 'counts': data_counts}

    df_result = pd.DataFrame(data=d).sort_values(by='counts').reset_index(drop=True)

    df_result['percentage'] = df_result['counts'] / sum(df_result['counts'])

    # Print in descending order and return df

    print()

    print(df_result.tail(50).iloc[::-1])

    return df_result
# Define function to plot barchart from dataframe

def plot_barchart(data_labels, data_counts, y_label, chart_title, height=10, numbers_label='Count'):

    y_pos = np.arange(len(data_labels))

    plt.figure(figsize=(20,height))

    plt.barh(y_pos, data_counts, align='center', alpha=1)    

    plt.yticks(y_pos, data_labels)

    plt.xlabel(numbers_label)

    plt.ylabel(y_label)

    plt.title(chart_title)    

    plt.show()
df_types_results = count_col_vals(df_autos_locs, 'vehicleType')
#Plot barchart

plot_title = 'Vehicle types with numbers on german used car market'

plot_barchart(df_types_results.labels, df_types_results.counts, 'Vehicle type', plot_title)
df_brands_results = count_col_vals(df_autos_locs, 'brand')

plot_title = 'Brands with numbers on german used car market'

plot_barchart(df_brands_results.labels, df_brands_results.counts, 'Brand', plot_title, 20)
# Create age column and plot age-histrogram

df_autos_locs['age'] = 2016 - df_autos_locs['yearOfRegistration']

plt.figure(figsize=(20,10))

bins = plt.hist(df_autos_locs['age'], bins=50, histtype='bar')
# Display non-Oldtimer cars only

n_years = 30

plt.figure(figsize=(20,10))

bins = plt.hist(df_autos_locs[df_autos_locs['age'] < n_years].age, bins=n_years, histtype='bar')
# Display descriptive statistics on the yearOfRegistration and age

print(df_autos_locs['yearOfRegistration'].describe())

print()

print(df_autos_locs['age'].describe())
# Plot boxplot of years

plt.figure(figsize=(10,10))

_ = plt.boxplot(df_autos_locs['yearOfRegistration'])
# Plot boxplot of car ages

plt.figure(figsize=(10,10))

_ = plt.boxplot(df_autos_locs['age'])
# Shorten Dataframe and pick random samples

n = 10000

df_autos_locs_short = df_autos_locs.sample(n, random_state=47)



# Plot location-density according to https://python-graph-gallery.com/85-density-plot-with-matplotlib/

x = df_autos_locs_short.lon

y = df_autos_locs_short.lat



# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

nbins=100

k = kde.gaussian_kde([x,y])

xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*2j]

zi = k(np.vstack([xi.flatten(), yi.flatten()]))

 

# Make the plot

plt.figure(figsize=(12,15))

plt.pcolormesh(xi, yi, zi.reshape(xi.shape))

plt.colorbar()

plt.show()
df_cities_results = count_col_vals(df_autos_locs, 'Ort')
n = 20

df_cities_results_short = df_cities_results.tail(n)

plot_title = 'Cities with numbers on german used car market'

plot_barchart(df_cities_results_short.labels, df_cities_results_short.counts, 'City', plot_title, 15)
# Of those n cities with the most offers, group by city and display mean values



len_before = len(df_autos_locs)

df = df_autos_locs.dropna(subset=['Ort']).dropna(subset=['price'])

print('Dropped {} NaN-Rows for column {}.'.format(len_before - len(df), col))

print('Number of remaining rows is {}.'.format(len(df)))



# Sort and count

df = df[df.Ort.isin(df_cities_results_short.labels.values)]



df_result = df.groupby(['Ort']).median().drop(columns=['postalCode', 'lon', 'lat'])

df_result
# Print barchart of prizes

df_result = df_result.sort_values(by='price')

plot_title = 'Median car prices by city on german used cars market'

plot_barchart(df_result.index, df_result.price, 'City', plot_title, 15, 'Price')
# Print barchart of prizes

df_result = df_result.sort_values(by='powerPS')

plot_title = 'Median car power by city on german used cars market'

plot_barchart(df_result.index, df_result.powerPS, 'City', plot_title, 15, 'Power [PS]')
# Print barchart of prizes

df_result = df_result.sort_values(by='age')

plot_title = 'Median car age by city on german used cars market'

plot_barchart(df_result.index, df_result.age, 'City', plot_title, 15, 'age [years]')
df_brands_results = count_col_vals(df_autos_locs, 'fuelType')

plot_title = 'fuelTypes with numbers on german used car market'

plot_barchart(df_brands_results.labels, df_brands_results.counts, 'Fuel Type', plot_title, 5)
df_brands_results = count_col_vals(df_autos_locs, 'gearbox')

plot_title = 'fuelTypes with numbers on german used car market'

plot_barchart(df_brands_results.labels, df_brands_results.counts, 'Gear Type', plot_title, 5)
# Descriptive Statistics on Power

df_autos_locs['powerPS'].describe()
# Plot boxplot of motor power

plt.figure(figsize=(10,10))

_ = plt.boxplot(df_autos_locs['powerPS'])
# Descriptive Statistics on sale prices

df_autos_locs['price'].describe()
# Plot boxplot of sale prices

plt.figure(figsize=(10,10))

_ = plt.boxplot(df_autos_locs['price'])
# Plot histogram of sale prices

n_bins = 25

plt.figure(figsize=(20,10))

bins = plt.hist(df_autos_locs.price, bins=n_bins, histtype='bar')
# Plot histogram of sale prices

max_price = 50000

n_bins = 50

plt.figure(figsize=(20,10))

bins = plt.hist(df_autos_locs[df_autos_locs.price <= max_price].price, bins=n_bins, histtype='bar')
# Descriptive Statistics on Kilometers (mileage)

df_autos_locs['kilometer'].describe()
# Plot boxplot of kilometers

plt.figure(figsize=(10,10))

_ = plt.boxplot(df_autos_locs['kilometer'])
# Plot histogram of kilometers

n_bins = 10

plt.figure(figsize=(20,10))

bins = plt.hist(df_autos_locs.kilometer, bins=n_bins, histtype='bar')
# Check shape before preprocessing

df_autos_locs.shape
# Check columns

df_autos_locs.head(3)
print(df_autos_locs.model.nunique())

print(df_autos_locs.Ort.nunique())

print(df_autos_locs.postalCode.nunique())
# Delete some columns

df_autos_locs_reg = df_autos_locs.drop(columns=['model', 'postalCode', 'Ort', 'yearOfRegistration'])
# Show number of missing data per column

df_autos_locs.isnull().sum()
# Drop Rows, that have NaN-values in relevant columns

df_autos_locs_reg = df_autos_locs_reg.dropna()

df_autos_locs_reg.shape
# Transform discrete values in dataset

# There are two discrete columns: 'gearbox' and 'notRepairedDamage'.

# 'gearbox' will be transformed to 'automaticGear', with 1 for automatic and 0 for manuell. 'Gearbox' will be deleted.



d = {'automatik': 1,'manuell': 0}

df_autos_locs_reg.gearbox = [d[item] for item in df_autos_locs_reg.gearbox]

df_autos_locs_reg = df_autos_locs_reg.rename(columns={'gearbox': 'automaticGear'})



# Transform values in 'notRepairedDamage'

d = {'ja': 1,'nein': 0}

df_autos_locs_reg.notRepairedDamage = [d[item] for item in df_autos_locs_reg.notRepairedDamage]



df_autos_locs_reg.head(3)
# Display correlation matrix

corr = df_autos_locs_reg.corr()

corr.style.background_gradient(cmap='coolwarm')

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# Get dummies for columns 'vehicleType', 'fuelType', and 'brand'.

df_autos_locs_reg_dum = pd.get_dummies(df_autos_locs_reg)

df_autos_locs_reg_dum.head(3)
# Extract features and perform train test split



#X = normalize(df_autos_locs_reg_dum.drop('price', axis=1)) # No normalization, as it worsened the performance.

#scaler = StandardScaler()

#X = scaler.fit_transform(df_autos_locs_reg_dum.drop('price', axis=1)) # No scaling, performanced stayed the same.



y = df_autos_locs_reg_dum.price

X = df_autos_locs_reg_dum.drop('price', axis=1)



# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Quick test, which algorithm seems to perform best for this task

#for regressor in [Ridge(), GradientBoostingRegressor(verbose=1), BaggingRegressor(verbose=1, random_state=42), SVR(kernel='linear', verbose=1), SVR(kernel='sbf', verbose=1)]:

for regressor in [Ridge(random_state=42), GradientBoostingRegressor(verbose=1, random_state=42),

                  BaggingRegressor(verbose=1, random_state=42), XGBRegressor(TREE_METHOD = 'gpu_hist')]:

    print(regressor)

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)    

    print('R2 score: {}'.format(explained_variance_score(y_test, y_pred)))

    #print('Explained variance score: {}'.format(r2_score(y_test, y_pred)))

    print('Mean Squared Error score: {}'.format(mean_squared_error(y_test, y_pred)))

    print()
# Use gridsearch to tune parameters for regressor

#parameters = {'alpha':[0.5, 0.9, 1], 'learning_rate':[0.01, 0.1, 1], 'max_depth':[2, 3, 10],

#              'min_samples_leaf':[1, 3, 5], 'n_estimators':[100, 500], 'max_features':['auto', 'sqrt', 0.5]}



#parameters = {'alpha':[0.5, 0.9, 1], 'learning_rate':[0.001, 0.01, 0.1, 1], 'max_depth':[2, 3, 5, 10],

#              'min_samples_leaf':[1, 2, 3, 5], 'n_estimators':[50, 100, 500], 'max_features':['auto', 'sqrt', 0.3, 0.7]}



parameters = {'learning_rate':[0.05, 0.1, 0.5], 'max_depth':[3, 5, 7], 'min_child_weight':[1,3,6],

              'gamma':[0, 0.1, 0.5], 'colsample_bytree':[0.5, 0.8, 0.9], 'scale_pos_weight': [0, 1],

             'n_estimators':[50, 100, 500]}



reg = XGBRegressor(TREE_METHOD = 'gpu_hist', random_state=42)

gscv = GridSearchCV(reg, parameters, cv=3, verbose=5, n_jobs=-1, return_train_score=False)

gscv.fit(X_train, y_train)
# Display the best estimator from gridsearch

gscv.best_estimator_
# Display best score

gscv.best_score_
# Display best parameters

gscv.best_params_
# Evaluate untuned regressor

y_pred = regressor.predict(X_test)

print('R2 score: {}'.format(explained_variance_score(y_test, y_pred)))

print('Mean Squared Error score: {}'.format(mean_squared_error(y_test, y_pred)))
# Evaluate tuned regressor



y_pred = gscv.best_estimator_.predict(X_test)

print('R2 score: {}'.format(explained_variance_score(y_test, y_pred)))

print('Mean Squared Error score: {}'.format(mean_squared_error(y_test, y_pred)))
# Save untuned model as .pkl

filename = 'regressor.pkl'

pickle.dump(regressor, open(filename, 'wb'))

# Export zips

pickle.dump(df_zips, open('zips.pkl', 'wb'))

# Export dataframe for command line app

pickle.dump(df_autos_locs_reg_dum.drop('price', axis=1).columns, open('df_cols.pkl', 'wb'))
# Save gridsearchCV object

pickle.dump(gscv, open('gs_object.pkl', 'wb'))

# Save tuned models from gridsearchCV

pickle.dump(gscv.best_estimator_, open('gs_best_reg.pkl', 'wb'))
# Load regressor from .pkl and test

loaded_regressor = pickle.load(open('gs_best_reg.pkl', 'rb'))

y_pred = loaded_regressor.predict(X_test)

print('R2 score: {}'.format(explained_variance_score(y_test, y_pred)))

print('Mean Squared Error score: {}'.format(mean_squared_error(y_test, y_pred)))
# Export this notebook as .html

from subprocess import call

call(['python', '-m', 'nbconvert', 'German_used_Cars.ipynb'])