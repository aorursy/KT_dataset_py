### Necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

### Seaborn style

sns.set_style("whitegrid")
## Dictionary of feature dtypes

ints = ['parcelid']



floats = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'finishedfloor1squarefeet', 

          'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13',

          'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt',

          'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'latitude', 'longitude',

          'lotsizesquarefeet', 'poolcnt', 'poolsizesum', 'roomcnt', 'threequarterbathnbr', 'unitcnt',

          'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt', 'numberofstories',

          'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',

          'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyyear']



objects = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',

           'buildingqualitytypeid', 'decktypeid', 'fips', 'hashottuborspa', 'heatingorsystemtypeid',

           'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode',

           'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',

           'regionidcounty', 'regionidneighborhood', 'regionidzip', 'storytypeid',

           'typeconstructiontypeid', 'fireplaceflag', 'taxdelinquencyflag', 'censustractandblock']



feature_dtypes = {col: col_type for type_list, col_type in zip([ints, floats, objects],

                                                               ['int64', 'float64', 'object']) 

                                  for col in type_list}
### Let's import our data

data = pd.read_csv('../input/properties_2016.csv' , dtype = feature_dtypes)

### and test if everything OK

data.head()
### ... check for NaNs

nan = data.isnull().sum()

nan
### Plotting NaN counts

nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()

nan_sorted.columns = ['Column', 'Number of NaNs']



fig, ax = plt.subplots(figsize=(12, 25))

sns.barplot(x="Number of NaNs", y="Column", data=nan_sorted, color='Sienna', ax=ax);

ax.set(xlabel="Number of NaNs", ylabel="", title="Total Nimber of NaNs in each column");
data.dtypes
continuous = ['basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 

              'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',

              'finishedsquarefeet50', 'finishedsquarefeet6', 'garagetotalsqft', 'latitude',

              'longitude', 'lotsizesquarefeet', 'poolsizesum',  'yardbuildingsqft17',

              'yardbuildingsqft26', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',

              'landtaxvaluedollarcnt', 'taxamount']



discrete = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'fireplacecnt', 'fullbathcnt',

            'garagecarcnt', 'poolcnt', 'roomcnt', 'threequarterbathnbr', 'unitcnt',

            'numberofstories', 'assessmentyear', 'taxdelinquencyyear']
### Continuous variable plots

for col in continuous:

    values = data[col].dropna()

    lower = np.percentile(values, 1)

    upper = np.percentile(values, 99)

    fig = plt.figure(figsize=(18,9));

    sns.distplot(values[(values>lower) & (values<upper)], color='Sienna', ax = plt.subplot(121));

    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

    plt.suptitle(col, fontsize=16)       
### Discrete variable plots

NanAsZero = ['fireplacecnt', 'poolcnt', 'threequarterbathnbr']

for col in discrete:

    if col in NanAsZero:

        data[col].fillna(0, inplace=True)

    values = data[col].dropna()   

    fig = plt.figure(figsize=(18,9));

    sns.countplot(x=values, color='Sienna', ax = plt.subplot(121));

    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

    plt.suptitle(col, fontsize=16)
### Categorical variable plots

for col in objects:

    values = data[col].astype('str').value_counts(dropna=False).to_frame().reset_index()

    if len(values) > 30:

        continue

    values.columns = [col, 'counts']

    fig = plt.figure(figsize=(18,9))

    ax = sns.barplot(x=col, y='counts', color='Sienna', data=values, order=values[col]);

    plt.xlabel(col);

    plt.ylabel('Number of occurrences')

    plt.suptitle(col, fontsize=16)



    ### Adding percents over bars

    height = [p.get_height() for p in ax.patches]    

    total = sum(height)

    for i, p in enumerate(ax.patches):    

        ax.text(p.get_x()+p.get_width()/2,

                height[i]+total*0.01,

                '{:1.0%}'.format(height[i]/total),

                ha="center")    
### Reading train file

errors = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'])

errors.head()
#### Merging tables

data_sold = data.merge(errors, how='inner', on='parcelid')

data_sold.head()
### Checking logerror

col = 'logerror'



values = data_sold[col].dropna()

lower = np.percentile(values, 1)

upper = np.percentile(values, 99)

fig = plt.figure(figsize=(18,9));

sns.distplot(values[(values>lower) & (values<upper)], color='Sienna', ax = plt.subplot(121));

sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

plt.suptitle(col, fontsize=16);
### Adding some new features from transactiondate

data_sold['month'] = data_sold['transactiondate'].dt.month

data_sold['day_of_week'] = data_sold['transactiondate'].dt.weekday_name

data_sold['week_number'] = data_sold['transactiondate'].dt.week

data_sold.head()
### Scrutinizing transactiondate

fig = plt.figure(figsize=(18, 18));

sns.countplot(x='transactiondate', color='Sienna', data=data_sold, ax = plt.subplot(221));

sns.countplot(x='month', color='Sienna', data=data_sold, ax = plt.subplot(222));

sns.countplot(x='day_of_week', color='Sienna', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday',

                                                      'Friday', 'Saturday', 'Sunday'], 

              data=data_sold, ax = plt.subplot(223));

sns.countplot(x='week_number', color='Sienna', data=data_sold, ax = plt.subplot(224));

plt.suptitle('Transaction Date', fontsize=20);
### Creating 5 equal size logerror bins 

data_sold['logerror_bin'] = pd.qcut(data_sold['logerror'], 5, 

                                    labels=['Large Negative Error', 'Medium Negative Error',

                                            'Small Error', 'Medium Positive Error',

                                            'Large Positive Error'])

print(data_sold.logerror_bin.value_counts())
### Continuous variable vs logerror plots

for col in continuous:     

    fig = plt.figure(figsize=(18,9));

    sns.barplot(x='logerror_bin', y=col, data=data_sold, ax = plt.subplot(121),

                order=['Large Negative Error', 'Medium Negative Error','Small Error',

                       'Medium Positive Error', 'Large Positive Error']);

    plt.xlabel('LogError Bin');

    plt.ylabel('Average {}'.format(col));

    sns.regplot(x='logerror', y=col, data=data_sold, color='Sienna', ax = plt.subplot(122));

    plt.suptitle('LogError vs {}'.format(col), fontsize=16)   