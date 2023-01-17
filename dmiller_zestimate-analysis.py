import numpy as np

import pandas as pd

import gc



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



from pandas.plotting import scatter_matrix



# Allows the use of display() for DataFrames

from IPython.display import display

# Pretty display for notebooks

%matplotlib inline 
def describe_dataset(dataset, threshold=0.90):

    ds = dataset.isnull().sum(axis=0).reset_index()

    ds.columns = ['feature_name', 'missing_count']

    ds['missing_ratio'] = ds['missing_count'] / dataset.shape[0]

    return ds



def visualise_missing_data(dataset):

    dataset = dataset.sort_values('missing_count', ascending=False)

    #ind = np.arange(dataset.shape[0])



    fig, ax = plt.subplots(figsize=(18,16))

    sns.set_color_codes('muted')

    sns.barplot(x='missing_ratio', y='feature_name', data=dataset, label='Ratio', color='b')

    ax.legend(ncol=2, loc='lower right', frameon=True)

    ax.set(xlim=(0, 1), ylabel='', xlabel='Missing data ratio')

    sns.despine(left=True, bottom=True)

    plt.show()
gc.collect()
properties = pd.read_csv('../input/properties_2016.csv', sep=',')

display(properties.shape)

display(properties.head())
missing_data = describe_dataset(properties)

visualise_missing_data(missing_data)
missing_data = describe_dataset(properties)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0.0])
gc.collect()
from sklearn.preprocessing import LabelEncoder



def encode_string_labels(dataset, target):

    nullvalues = dataset[target].isnull()

    dataset[target] = dataset[target].astype(str)

    encoder = LabelEncoder()

    

    dataset[target] = encoder.fit_transform(dataset[target].values)

    # restore the NaN values

    dataset.loc[nullvalues, target] = np.nan
for column in ['propertycountylandusecode', 'propertyzoningdesc']:

    if properties[column].dtype == object:

        print("Encoding feature {}".format(column))

        encode_string_labels(properties, column)

        display(properties[column].describe())
properties.loc[properties['hashottuborspa'].isnull(), 'hashottuborspa'] = 0

properties.loc[properties['hashottuborspa'] == True, 'hashottuborspa'] = 1
properties.loc[properties['taxdelinquencyflag'].isnull(), 'taxdelinquencyflag'] = 0

properties.loc[properties['taxdelinquencyflag'] == 'Y', 'taxdelinquencyflag'] = 1
gc.collect()
transactions = pd.read_csv('../input/train_2016_v2.csv', sep=',', parse_dates=['transactiondate'])



transactions['transactionyear'] = transactions['transactiondate'].dt.year.astype(np.int32)

transactions['transactionmonth'] = transactions['transactiondate'].dt.month.astype(np.int32)



display(transactions.shape)

display(transactions.head())
gc.collect()
transactions = pd.merge(transactions, properties, on='parcelid')
display(transactions.shape)

display(transactions.head())
missing_data = describe_dataset(transactions)

visualise_missing_data(missing_data)
missing_data = describe_dataset(transactions)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0.0])
features = list(transactions.columns.values)

corr = transactions.loc[:, features].corr(method='spearman')
corr = corr.sort_values('logerror', ascending=False)

display(corr['logerror'])
corr = corr.sort_values('taxamount', ascending=False)

display(corr['taxamount'].head())
corr = corr.sort_values('taxdelinquencyyear', ascending=False)

display(corr['taxdelinquencyyear'].head())
corr = corr.sort_values('bathroomcnt', ascending=False)

display(corr['bathroomcnt'].head())
corr = corr.sort_values('bedroomcnt', ascending=False)

display(corr['bedroomcnt'].head())
corr = corr.sort_values('calculatedfinishedsquarefeet', ascending=False)

display(corr['calculatedfinishedsquarefeet'])
corr = corr.sort_values('propertylandusetypeid', ascending=False)

display(corr['propertylandusetypeid'])
corr = corr.sort_values('propertycountylandusecode', ascending=False)

display(corr['propertycountylandusecode'].head())
corr = corr.sort_values('propertyzoningdesc', ascending=False)

display(corr['propertyzoningdesc'].head())
corr = corr.sort_values('regionidcounty', ascending=False)

display(corr['regionidcounty'].head())
corr = corr.sort_values('regionidcity', ascending=False)

display(corr['regionidcity'].head())
corr = corr.sort_values('regionidzip', ascending=False)

display(corr['regionidzip'].head())
# group and sort data transaction

transactions['abs_logerror'] = transactions['logerror'].abs()

ds = transactions.groupby(transactions['transactionmonth'], sort=True)['abs_logerror'].mean()

ds2 = transactions.groupby(transactions['transactionmonth'], sort=True)['abs_logerror'].median()



fig, ax = plt.subplots(figsize=(18,6))

ax2 = ax.twinx()



ax.hist(transactions['transactionmonth'], alpha=0.2)

ax2.plot(ds, color='red', alpha=0.8)

#ax2.plot(ds2, color='black', alpha=0.8)

plt.show()
ds = transactions.sort_values('logerror', ascending=True)



plt.figure(figsize=(18,8))

plt.scatter(range(ds.shape[0]), ds['logerror'].values)

plt.ylabel('Logerror')

plt.show()
ds = transactions[transactions['logerror'] >= 1.0]

display(ds.describe())
corr = ds.corr(method='spearman')

corr = corr.sort_values('logerror', ascending=False)

display(corr['logerror'])
# UNDER ESTIMATEDds = transactions[transactions['logerror'] <= -1.0]

display(ds.describe())
corr = ds.corr(method='spearman')

corr = corr.sort_values('logerror', ascending=False)

display(corr['logerror'])
# drop the features which we can't impute/derive -- too many missing data points

transactions.drop('buildingclasstypeid', axis=1, inplace=True)

transactions.drop('basementsqft', axis=1, inplace=True)

transactions.drop('storytypeid', axis=1, inplace=True) 

transactions.drop('fireplaceflag', axis=1, inplace=True)

transactions.drop('architecturalstyletypeid', axis=1, inplace=True)

transactions.drop('typeconstructiontypeid', axis=1, inplace=True)

transactions.drop('decktypeid', axis=1, inplace=True)
# drop because no correlation

transactions.drop('assessmentyear', axis=1, inplace=True)
attributes = ['propertycountylandusecode', 'propertyzoningdesc', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'fips']

scatter_matrix(transactions[attributes], figsize=(18, 20))
attributes = ['calculatedfinishedsquarefeet', 'finishedfloor1squarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet6', 'finishedsquarefeet15', 'finishedsquarefeet50']

scatter_matrix(transactions[attributes], figsize=(18, 20))
corr = transactions.corr(method='spearman')

corr = corr.sort_values('calculatedfinishedsquarefeet', ascending=False)

display(corr['calculatedfinishedsquarefeet'].head())
ds = transactions[transactions['calculatedfinishedsquarefeet'].isnull()]

display(ds['finishedsquarefeet13'].describe())

display(ds['finishedsquarefeet15'].describe())

display(ds['finishedsquarefeet12'].describe())

display(ds['finishedsquarefeet6'].describe())
ds = transactions[transactions['finishedsquarefeet12'] > 0]

display(ds['calculatedfinishedsquarefeet'].describe())

display(ds['finishedsquarefeet12'].describe())
ds = transactions[transactions['finishedsquarefeet50'] > 0]

display(ds['finishedfloor1squarefeet'].describe())

display(ds['finishedsquarefeet50'].describe())
# duplicated by calculatedfinishedsquarefeet or strongly related

transactions.drop('finishedsquarefeet6', axis=1, inplace=True)

transactions.drop('finishedsquarefeet12', axis=1, inplace=True)

transactions.drop('finishedsquarefeet13', axis=1, inplace=True)

transactions.drop('finishedsquarefeet15', axis=1, inplace=True)

transactions.drop('finishedsquarefeet50', axis=1, inplace=True)
display(transactions['finishedfloor1squarefeet'].describe())



display(transactions['numberofstories'].describe())



ds = transactions[transactions['numberofstories'] == 1]

display(ds['finishedfloor1squarefeet'].describe())
display(transactions[

    (transactions['numberofstories'] == 1) &

    (transactions['finishedfloor1squarefeet'].isnull())

])
transactions.loc[

    (transactions['finishedfloor1squarefeet'].isnull()) &

    (transactions['numberofstories'] == 1),

    'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet']
display(transactions[

    (transactions['numberofstories'] == 1) &

    (transactions['finishedfloor1squarefeet'].isnull())

])
transactions.loc[

    (transactions['finishedfloor1squarefeet'].isnull()) &

    (transactions['numberofstories'] > 1), 'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet'] / transactions['numberofstories']
display(transactions['finishedfloor1squarefeet'].describe())



ds = transactions[transactions['finishedfloor1squarefeet'] > transactions['calculatedfinishedsquarefeet']]

display(ds)
display(transactions['buildingqualitytypeid'].describe())
median = transactions['buildingqualitytypeid'].median()

transactions.loc[transactions['buildingqualitytypeid'].isnull(), 'buildingqualitytypeid'] = median
# values 1 to 13 inclusive

display(transactions['airconditioningtypeid'].describe())
median = transactions['airconditioningtypeid'].median()

transactions.loc[transactions['airconditioningtypeid'].isnull(), 'airconditioningtypeid'] = median
# values 1 to 25 inclusive

display(transactions['heatingorsystemtypeid'].describe())
median = transactions['heatingorsystemtypeid'].median()

transactions.loc[transactions['heatingorsystemtypeid'].isnull(), 'heatingorsystemtypeid'] = median
# assume no fireplace if data missing

display(transactions['fireplacecnt'].describe())

transactions.loc[transactions['fireplacecnt'].isnull(), 'fireplacecnt'] = 0
attributes = ['roomcnt', 'bedroomcnt', 'bathroomcnt', 'calculatedbathnbr', 'fullbathcnt', 'threequarterbathnbr']

scatter_matrix(transactions[attributes], figsize=(18, 20))
ds = transactions[transactions['bathroomcnt'] != transactions['calculatedbathnbr']]

display(ds['calculatedbathnbr'].describe())



# Difference is missing data points in calculatedbathnbr
# dropping following as they are represented by bathroomcnt

transactions.drop('calculatedbathnbr', axis=1, inplace=True)

transactions.drop('fullbathcnt', axis=1, inplace=True)

transactions.drop('threequarterbathnbr', axis=1, inplace=True)
ds = transactions[

    (transactions['bedroomcnt'] > 0) &

    (transactions['roomcnt'] == 0)]



display(ds.shape)

display(ds.describe())
# dropping as data captured in other features

transactions.drop('pooltypeid2', axis=1, inplace=True)

transactions.drop('pooltypeid7', axis=1, inplace=True)

transactions.drop('pooltypeid10', axis=1, inplace=True)
display(transactions['poolcnt'].describe())

display(transactions['poolsizesum'].describe())
# impute missing values for poolsizesum - set to median pool size for all pools

median = transactions[transactions['poolcnt'] == 1]['poolsizesum'].median()

transactions.loc[

    (transactions['poolsizesum'].isnull()) &

    (transactions['poolcnt'] == 1), 'poolsizesum'] = median

transactions.loc[(transactions['poolsizesum'].isnull()), 'poolsizesum'] = 0



transactions.drop('poolcnt', axis=1, inplace=True)
missing_data = describe_dataset(transactions)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0])
ds = transactions[transactions['taxdelinquencyflag'] == 1]

display(ds['taxdelinquencyflag'].describe())
transactions.drop('taxdelinquencyflag', axis=1, inplace=True)
display(transactions['taxdelinquencyyear'].describe())
# convert to actual year

transactions.loc[

    (transactions['taxdelinquencyyear'] == 99),

    'taxdelinquencyyear'] = 1999

transactions.loc[

    (transactions['taxdelinquencyyear'] > 0) &

    (transactions['taxdelinquencyyear'] < 1999),

    'taxdelinquencyyear'] = 2000 + transactions['taxdelinquencyyear']



transactions.loc[

    (transactions['taxdelinquencyyear'].isnull()),

    'taxdelinquencyyear'] = 0



#convert type

transactions.loc[:, 'taxdelinquencyyear'] = transactions['taxdelinquencyyear'].astype(np.int32)
ds = transactions[transactions['taxdelinquencyyear'] > 0]



plt.figure(figsize=(18,8))

plt.hist(ds['taxdelinquencyyear'], alpha=0.5, bins=40)

plt.xlabel('Tax Delinquency Year')

plt.ylabel('Frequency')

plt.show()
missing_data = describe_dataset(transactions)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0.0])
gc.collect()
display(transactions['garagetotalsqft'].describe())

display(transactions['garagecarcnt'].describe())



# if the garage data is missing, assume no garage

transactions.loc[transactions['garagetotalsqft'].isnull(), 'garagetotalsqft'] = 0

transactions.loc[transactions['garagecarcnt'].isnull(), 'garagecarcnt'] = 0
#transactions.drop('yardbuildingsqft26', axis=1, inplace=True)

#transactions.drop('yardbuildingsqft17', axis=1, inplace=True)



display(transactions['yardbuildingsqft26'].describe())

display(transactions['yardbuildingsqft17'].describe())



# if the yard building data is missing, assume no building

transactions.loc[transactions['yardbuildingsqft26'].isnull(), 'yardbuildingsqft26'] = 0 # patio in yard

transactions.loc[transactions['yardbuildingsqft17'].isnull(), 'yardbuildingsqft17'] = 0 # shed/building yard
# buildingqualitytypeid:  Overall assessment of condition of the building from best (lowest) to worst (highest)

display(transactions['buildingqualitytypeid'].describe())
missing_data = describe_dataset(transactions)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0.0])
# no building on the land

ds = transactions[transactions['structuretaxvaluedollarcnt'].isnull()]

display(ds[ds['bedroomcnt'] == 0])
# technically, taxvaluedollarcnt - landtaxvaluedollarcnt would yield the answer as well, turns out to be zero

# note: 14324521 has one story 

#transactions.loc[transactions['structuretaxvaluedollarcnt'].isnull(), 'structuretaxvaluedollarcnt'] = 0
# do we have parcels with no structures built?

ds = transactions[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'].isnull()) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0)]



display(ds.describe())
index = transactions[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'].isnull()) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0)

]
transactions.loc[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'].isnull()) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0),

    'finishedfloor1squarefeet'] = 0
# there is one reading with a calculatedfinishedsquarefeet of 1518 - preserve in case this helps identify errors

transactions.loc[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'].isnull()) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0),

    'calculatedfinishedsquarefeet'] = 0
transactions.loc[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'].isnull()) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0),

    'numberofstories'] = 0
# finally set structuretaxvaluedollarcnt to 0

transactions.loc[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'] == 0) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0),

    'structuretaxvaluedollarcnt'] = 0
# set year built to 0

transactions.loc[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] == transactions['landtaxvaluedollarcnt']) &

    (transactions['yearbuilt'].isnull()) &

    (transactions['bathroomcnt'] == 0) &

    (transactions['bedroomcnt'] == 0),

    'yearbuilt'] = 0
attributes = ['landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']

scatter_matrix(transactions[attributes], diagonal = 'kde', figsize=(18, 20))
# was going to predict the missing values - but this could cause other issues, and maybe property was

# exempt from tax; so setting to difference between total and land tax values (should == 0)



transactions.loc[

    (transactions['structuretaxvaluedollarcnt'].isnull()) &

    (transactions['taxvaluedollarcnt'] > 0),

    'structuretaxvaluedollarcnt'] = (transactions['taxvaluedollarcnt'] - transactions['landtaxvaluedollarcnt'])
attributes = ['landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']

scatter_matrix(transactions[attributes], diagonal = 'kde', figsize=(18, 20))
attributes = ['roomcnt', 'bedroomcnt', 'bathroomcnt', 'taxamount']

scatter_matrix(transactions[attributes], figsize=(18, 20))
ds = transactions[transactions['numberofstories'].isnull()]

display(ds['propertylandusetypeid'].describe())
transactions.loc[

    (transactions['propertylandusetypeid'] == 31) &

    (transactions['numberofstories'].isnull()),

    'numberofstories'] = 1



transactions.loc[

    (transactions['propertylandusetypeid'] == 31) &

    (transactions['finishedfloor1squarefeet'].isnull()),

    'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet']
transactions.loc[

    (transactions['propertylandusetypeid'] == 261) &

    (transactions['numberofstories'].isnull()),

    'numberofstories'] = 1



transactions.loc[

    (transactions['propertylandusetypeid'] == 261) &

    (transactions['finishedfloor1squarefeet'].isnull()),

    'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet']
transactions.loc[

    (transactions['propertylandusetypeid'] == 263) &

    (transactions['numberofstories'].isnull()),

    'numberofstories'] = 1



transactions.loc[

    (transactions['propertylandusetypeid'] == 263) &

    (transactions['finishedfloor1squarefeet'].isnull()),

    'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet']
transactions.loc[

    (transactions['propertylandusetypeid'] == 266) &

    (transactions['numberofstories'].isnull()),

    'numberofstories'] = 1



transactions.loc[

    (transactions['propertylandusetypeid'] == 266) &

    (transactions['finishedfloor1squarefeet'].isnull()),

    'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet']
transactions.loc[

    (transactions['propertylandusetypeid'] == 275) &

    (transactions['numberofstories'].isnull()),

    'numberofstories'] = 1



transactions.loc[

    (transactions['propertylandusetypeid'] == 275) &

    (transactions['finishedfloor1squarefeet'].isnull()),

    'finishedfloor1squarefeet'] = transactions['calculatedfinishedsquarefeet']
# planned development

transactions.loc[

    (transactions['propertylandusetypeid'] == 269) &

    (transactions['numberofstories'].isnull()),

    'numberofstories'] = 0



transactions.loc[

    (transactions['propertylandusetypeid'] == 269) &

    (transactions['finishedfloor1squarefeet'].isnull()),

    'finishedfloor1squarefeet'] = 0
ds = transactions[transactions['propertylandusetypeid'] == 246]

display(ds['finishedfloor1squarefeet'].describe())

display(ds['calculatedfinishedsquarefeet'].describe())

display(ds['numberofstories'].describe())
display(transactions['numberofstories'].describe())

transactions.loc[(transactions['numberofstories'].isnull()), 'numberofstories'] = 1
display(transactions['unitcnt'].describe())

transactions.loc[(transactions['unitcnt'].isnull()), 'unitcnt'] = 1
missing_data = describe_dataset(transactions)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0.0])
from sklearn.decomposition import PCA



pca = PCA(n_components=4)

ds = transactions.select_dtypes([np.number])

ds = ds.dropna()

ds_reduced = pca.fit_transform(ds)



#display(pca.explained_variance_ratio_)



cumsum = np.cumsum(pca.explained_variance_ratio_)

display(cumsum)
def pca_results(good_data, pca):

    '''

    Create a DataFrame of the PCA results

    Includes dimension feature weights and explained variance

    Visualizes the PCA results

    '''



    # Dimension indexing

    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



    # PCA components

    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())

    components.index = dimensions



    # PCA explained variance

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

    variance_ratios.index = dimensions



    # Create a bar plot visualization

    fig, ax = plt.subplots(figsize = (14,8))



    # Plot the feature weights as a function of the components

    components.plot(ax = ax, kind = 'bar');

    ax.set_ylabel("Feature Weights")

    ax.set_xticklabels(dimensions, rotation=0)

    ax.legend().set_visible(False)



    # Display the explained variance ratios

    for i, ev in enumerate(pca.explained_variance_ratio_):

        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))



    # Return a concatenated DataFrame

    return pd.concat([variance_ratios, components], axis = 1)
pca_results = pca_results(ds, pca)
noop
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
def calculate_score(model, X_test, y_test):

    y_predict = model.predict(X_test)

    score = r2_score(y_test, y_predict)

    return score
def create_knn_classification_model(dataset, features, target, neighbours=10):

    all_features = features + [target]

    ds = dataset.dropna(subset=all_features)

    print(ds.shape)



    if ds[target].dtype == float:

        ds[[target]] = ds[[target]].astype(np.int32)



    X_train, X_test, y_train, y_test = train_test_split(

        ds.loc[:, features],

        ds[target],

        test_size=0.20,

        random_state=42)



    model = KNeighborsClassifier(n_neighbors=neighbours, n_jobs=-1)

    model.fit(X_train, y_train)



    score = calculate_score(model, X_test, y_test)

    return (model, score)
features = ['rawcensustractandblock']

target = 'regionidcity'

model, score = create_knn_classification_model(transactions, features, target, neighbours=10)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['rawcensustractandblock']

target = 'regionidzip'

model, score = create_knn_classification_model(transactions, features, target, neighbours=5)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['rawcensustractandblock']

target = 'regionidneighborhood'

model, score = create_knn_classification_model(transactions, features, target, neighbours=1)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['latitude', 'longitude', 'rawcensustractandblock']

target = 'lotsizesquarefeet'

model, score = create_knn_regression_model(transactions, features, target, neighbours=2)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['latitude', 'longitude', 'rawcensustractandblock']

target = 'propertycountylandusecode'

model, score = create_knn_classification_model(transactions, features, target, neighbours=2)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['rawcensustractandblock']

target='propertyzoningdesc'

model, score = create_knn_classification_model(transactions, features, target, neighbours=10)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['yearbuilt', 'taxvaluedollarcnt', 'propertycountylandusecode']

target = 'taxamount'

model, score = create_regression_model(transactions, features, target)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['latitude', 'longitude', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt']

target = 'yearbuilt'

model, score = create_knn_classification_model(transactions, features, target, neighbours=5)

print(score)
noop
fill_missing_values(transactions, features, target, model)
features = ['bedroomcnt', 'bathroomcnt', 'yearbuilt']

target = 'roomcnt'

model, score = create_knn_classification_model(transactions, features, target, neighbours=15)

print(score)
fill_missing_values(transactions, features, target, model)
# todo - check predictions

ds = transactions[

    (transactions['roomcnt'] == 0) &

    (transactions['bathroomcnt'] > 0)]





median = transactions['roomcnt'].median()



transactions.loc[

    (transactions['roomcnt'] == 0) &

    (transactions['bathroomcnt'] > 0),

     'roomcnt'] = median
features = ['bedroomcnt', 'bathroomcnt', 'yearbuilt', 'structuretaxvaluedollarcnt', 'lotsizesquarefeet']

target='calculatedfinishedsquarefeet'

model, score = create_regression_model(transactions, features, target)

print(score)
fill_missing_values(transactions, features, target, model)
features = ['bedroomcnt', 'bathroomcnt', 'numberofstories', 'yearbuilt', 'structuretaxvaluedollarcnt', 'calculatedfinishedsquarefeet']

target='finishedfloor1squarefeet'

model, score = create_regression_model(transactions, features, target)

print(score)
fill_missing_values(transactions, features, target, model)
missing_data = describe_dataset(transactions)

ds = missing_data.sort_values('missing_count', ascending=False)

display(ds[ds['missing_ratio'] > 0.0])
ds = transactions.groupby(['transactiondate'], sort=True)['logerror'].median()



plt.figure(figsize=(18,6))

plt.plot(ds, color='red', alpha=0.8)

plt.show()
transactions['abs_logerror'] = transactions['logerror'].abs()

ds = transactions.groupby(['bedroomcnt'], sort=True)['abs_logerror'].median()



plt.figure(figsize=(18,6))

plt.plot(ds, color='red', alpha=0.8)

plt.show()
ds = transactions.groupby(['roomcnt'], sort=True)['abs_logerror'].median()



plt.figure(figsize=(18,6))

plt.plot(ds, color='red', alpha=0.8)

plt.show()
plt.figure(figsize=(6,6))



plt.scatter(transactions.latitude, transactions.longitude, c=transactions.fips)

plt.xlabel('Latitude', fontsize=12)

plt.ylabel('Longitude', fontsize=12)

plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



#logerror = scaler.fit_transform(transactions.abs_logerror)



plt.figure(figsize=(10,10))



plt.scatter(transactions.latitude, transactions.longitude, c=transactions.abs_logerror)

plt.xlabel('Latitude', fontsize=12)

plt.ylabel('Longitude', fontsize=12)

plt.show()
plt.figure(figsize=(10,10))

sns.jointplot(x=transactions.latitude, y=transactions.longitude, kind='hex')

plt.show()
plt.figure(figsize=(6,6))



plt.scatter(transactions.latitude, transactions.longitude, c=transactions.regionidneighborhood)

plt.xlabel('Latitude', fontsize=12)

plt.ylabel('Longitude', fontsize=12)

plt.show()
plt.figure(figsize=(6,6))



plt.scatter(transactions.latitude, transactions.longitude, c=transactions.regionidzip, alpha=0.2)

plt.xlabel('Latitude', fontsize=12)

plt.ylabel('Longitude', fontsize=12)

plt.show()