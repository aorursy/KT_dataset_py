import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
features_dictionary = pd.read_excel('../input/zillow_data_dictionary.xlsx')

features_dictionary['Feature'] = features_dictionary['Feature'].apply(lambda x: x.replace("'", ""))

features_dictionary.head()
n_features = len(features_dictionary)
properties_df = pd.read_csv('../input/properties_2016.csv')

properties_df.head()
properties_df.shape
missing_df = pd.DataFrame({'Missing': properties_df.isnull().sum()/len(properties_df)})

missing_df.sort_values(by="Missing", ascending=True, inplace=True)



fig,ax = plt.subplots(figsize=(10,15))

ax.barh(np.arange(n_features), missing_df['Missing'])

ax.set_yticks(np.arange(n_features))

ax.set_yticklabels(missing_df.index)

plt.show()
def summary(f):

    print('Description: '+features_dictionary[features_dictionary['Feature'] == f]['Description'].values[0])

    print('Type: {}'.format(properties_df[f].dtype))

    print('Missing: {}%'.format(100*missing_df.loc[f].values[0]))
summary('parcelid')
properties_df['parcelid'].min(), properties_df['parcelid'].mean(), properties_df['parcelid'].max()
fig,ax = plt.subplots()

sns.distplot(np.log10(properties_df['parcelid']), kde=False, ax=ax)
summary('fips')
fig,ax = plt.subplots()

sns.countplot(properties_df['fips'], ax=ax)

ax.set_xticklabels(['6037 (Los Angeles)', '6059 (Orange)', '6111 (Ventura)'])
summary('propertylandusetypeid')
fig,ax = plt.subplots(figsize=(8,3))

sns.countplot(properties_df['propertylandusetypeid'], ax=ax)
summary('rawcensustractandblock')
len(properties_df['rawcensustractandblock'].unique())
summary('regionidcounty')
fig,ax = plt.subplots(figsize=(8,3))

sns.countplot(properties_df['regionidcounty'], ax=ax)
summary('longitude')

summary('latitude')
colors = properties_df['fips'].dropna().map({6037: 'red', 6059: 'blue', 6111: 'green'})

sample = np.random.randint(0, len(properties_df['longitude'].dropna()), size=10000)

plt.scatter(properties_df['longitude'].dropna()[sample], properties_df['latitude'].dropna()[sample], c=colors[sample])
summary('assessmentyear')
(properties_df['assessmentyear'] == 2015).mean()
summary('bedroomcnt')
fig,ax = plt.subplots(figsize=(8,3))

sns.countplot(properties_df['bedroomcnt'], ax=ax)
summary('bathroomcnt')
fig,ax = plt.subplots(figsize=(12,3))

sns.countplot(properties_df['bathroomcnt'], ax=ax)
summary('roomcnt')
fig,ax = plt.subplots(figsize=(12,3))

sns.countplot(properties_df['roomcnt'], ax=ax)
summary('propertycountylandusecode')
len(properties_df['propertycountylandusecode'].unique())
fig,ax = plt.subplots(3, 1, figsize=(12,6))

for i,county in enumerate([6037, 6059, 6111]):

    sns.countplot(properties_df[properties_df['fips'] == county]['propertycountylandusecode'], ax=ax[i])
summary('regionidzip')
properties_df['regionidzip'].max()
(properties_df['regionidzip'] == properties_df['regionidzip'].max()).sum()
sns.distplot(properties_df[properties_df['regionidzip'] < 300000]['regionidzip'].dropna(), kde=False)
summary('taxamount')
properties_df['taxamount'].mean()
sns.distplot(np.log10(properties_df['taxamount'].dropna()), kde=False)
summary('taxvaluedollarcnt')
properties_df['taxvaluedollarcnt'].mean()
sns.distplot(np.log10(properties_df['taxvaluedollarcnt'].dropna()), kde=False)
summary('structuretaxvaluedollarcnt')
properties_df['structuretaxvaluedollarcnt'].mean()
sns.distplot(np.log10(properties_df['structuretaxvaluedollarcnt'].dropna()), kde=False)
summary('calculatedfinishedsquarefeet')
sns.distplot(np.log10(properties_df['calculatedfinishedsquarefeet'].dropna()), kde=False)
summary('yearbuilt')
properties_df['yearbuilt'].min(), properties_df['yearbuilt'].max()
sns.distplot(properties_df['yearbuilt'].dropna(), kde=False)
summary('regionidcity')
fig,ax = plt.subplots(3, 1, figsize=(12,6))

for i,county in enumerate([6037, 6059, 6111]):

    sns.countplot(properties_df[properties_df['fips'] == county]['regionidcity'], ax=ax[i])
summary('landtaxvaluedollarcnt')
sns.distplot(np.log10(properties_df['landtaxvaluedollarcnt'].dropna()), kde=False)
summary('censustractandblock')
properties_df['censustractandblock'][1000], properties_df['rawcensustractandblock'][1000]
summary('fullbathcnt')
sns.countplot(properties_df['fullbathcnt'])
summary('calculatedbathnbr')
sns.countplot(properties_df['calculatedbathnbr'])
summary('finishedsquarefeet12')
sns.distplot(np.log10(properties_df['finishedsquarefeet12'].dropna()), kde=False)
summary('lotsizesquarefeet')
sns.distplot(np.log10(properties_df['lotsizesquarefeet'].dropna()), kde=False)
summary('propertyzoningdesc')
properties_df['propertyzoningdesc'].unique().size
summary('unitcnt')
properties_df['unitcnt'].min(), properties_df['unitcnt'].mean(), properties_df['unitcnt'].max()
sns.countplot(np.clip(properties_df['unitcnt'], 0, 10))
summary('buildingqualitytypeid')
properties_df['buildingqualitytypeid'].mean()
properties_df['buildingqualitytypeid'].unique()
sns.countplot(properties_df['buildingqualitytypeid'])
summary('heatingorsystemtypeid')
sns.countplot(properties_df['heatingorsystemtypeid'])
summary('regionidneighborhood')
properties_df['regionidneighborhood'].unique().size
summary('garagecarcnt')
sns.countplot(np.clip(properties_df['garagecarcnt'], 0, 10))
summary('garagetotalsqft')
sns.distplot(np.clip(properties_df['garagetotalsqft'].dropna(), 0, 2000), kde=False)
summary('airconditioningtypeid')
sns.countplot(properties_df['airconditioningtypeid'])
summary('numberofstories')
sns.countplot(properties_df['numberofstories'])
summary('poolcnt')
properties_df['poolcnt'].unique()
summary('pooltypeid7')
properties_df['pooltypeid7'].unique()
summary('fireplacecnt')
sns.countplot(properties_df['fireplacecnt'].fillna(0.))
summary('threequarterbathnbr')
sns.countplot(properties_df['threequarterbathnbr'].fillna(0.))
summary('finishedfloor1squarefeet')
sns.distplot(np.log10(properties_df['finishedfloor1squarefeet'].dropna()), kde=False)
summary('finishedsquarefeet50')
(np.abs(properties_df['finishedsquarefeet50'].dropna() - properties_df['finishedfloor1squarefeet'].dropna())).mean()
sns.distplot(np.log10(properties_df['finishedsquarefeet50'].dropna()), kde=False)
summary('finishedsquarefeet15')
sns.distplot(np.log10(properties_df['finishedsquarefeet15'].dropna()), kde=False)
summary('yardbuildingsqft17')
sns.distplot(np.log10(properties_df['yardbuildingsqft17'].dropna()), kde=False)
summary('hashottuborspa')
properties_df['hashottuborspa'].unique()
summary('taxdelinquencyyear')
properties_df['taxdelinquencyyear'].unique()
properties_df['taxdelinquencyyear'].apply(lambda x: 1900+x if x>80 else 2000+x).unique()
fig,ax = plt.subplots(figsize=(10,3))

sns.countplot(properties_df['taxdelinquencyyear'].apply(lambda x: 1900+x if x>80 else 2000+x), ax=ax)
summary('taxdelinquencyflag')
properties_df['taxdelinquencyflag'].unique()
summary('pooltypeid10')
properties_df['pooltypeid10'].unique()
summary('pooltypeid2')
properties_df['pooltypeid2'].unique()
summary('poolsizesum')
sns.distplot(np.log10(properties_df['poolsizesum'].dropna()), kde=False)
summary('finishedsquarefeet6')
sns.distplot(np.log10(properties_df['finishedsquarefeet6'].dropna()), kde=False)
summary('decktypeid')
properties_df['decktypeid'].unique()
summary('buildingclasstypeid')
sns.countplot(properties_df['buildingclasstypeid'])
summary('finishedsquarefeet13')
sns.distplot(np.log10(properties_df['finishedsquarefeet13'].dropna()), kde=False)
summary('typeconstructiontypeid')
sns.countplot(properties_df['typeconstructiontypeid'])
summary('architecturalstyletypeid')
properties_df['architecturalstyletypeid'].unique()
sns.countplot(properties_df['architecturalstyletypeid'])
summary('fireplaceflag')
summary('yardbuildingsqft26')
sns.distplot(np.log10(properties_df['yardbuildingsqft26'].dropna()), kde=False)
summary('basementsqft')
sns.distplot(np.log10(properties_df['basementsqft'].dropna()), kde=False)
summary('storytypeid')
properties_df['storytypeid'].unique()