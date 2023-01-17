import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

color = sns.color_palette()

%matplotlib inline

from sklearn.experimental import enable_iterative_imputer # to impute logerror

from sklearn.impute import IterativeImputer
prop2016 = pd.read_csv("../input/group-9-data-set/2016prop.csv", low_memory = False)

train2016 = pd.read_csv("../input/group-9-data-set/train_2016_v2.csv", parse_dates=["transactiondate"], low_memory = False)

pd.set_option('display.max_columns', None)
prop2016.head()
train2016.head()
# To find the distribution of logerror 

ulimit = np.percentile(train2016.logerror.values, 99)

llimit = np.percentile(train2016.logerror.values, 1)



plt.figure(figsize=(12,4))

sns.distplot(train2016.logerror.values, bins=500, kde=False)

plt.xlabel("Log Error", fontsize=12)

mu, std = norm.fit(train2016.logerror.values)

xmin,xmax=plt.xlim()

x = np.linspace(-5,5,1000)

p = norm.pdf(x, 0, 0.025)

plt.plot(x, p*(x[1]-x[0])*90811, 'r', linewidth=2)

plt.axis([-0.5,0.5,0,20000])

plt.ylabel("Number of Parcels", fontsize=12)

plt.title("Train 2016 Dataset")


id_feature=['airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid',

           'buildingqualitytypeid','decktypeid','hashottuborspa','heatingorsystemtypeid',

           'pooltypeid10','pooltypeid2','pooltypeid7','propertylandusetypeid',

            'storytypeid','typeconstructiontypeid','fireplaceflag','taxdelinquencyflag',

            'taxdelinquencyyear']

cnt_feature=['bathroomcnt','bedroomcnt','calculatedbathnbr','fireplacecnt','fullbathcnt',

            'garagecarcnt','garagetotalsqft','poolcnt','roomcnt','threequarterbathnbr',

            'unitcnt','yearbuilt','numberofstories','assessmentyear']

size_feature=['basementsqft','finishedfloor1squarefeet','calculatedfinishedsquarefeet',

             'finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15',

              'finishedsquarefeet50','finishedsquarefeet6','lotsizesquarefeet',

             'poolsizesum','yardbuildingsqft17','yardbuildingsqft26','structuretaxvaluedollarcnt','taxvaluedollarcnt',

             'landtaxvaluedollarcnt','taxamount','latitude','longitude']

location_feature=['fips','propertycountylandusecode','rawcensustractandblock',

                 'regionidcity','regionidcounty','regionidneighborhood','regionidzip','censustractandblock']

str_feature=['propertyzoningdesc','propertycountylandusecode']
# data type analysis

dtype_df16 = prop2016.dtypes.reset_index()

dtype_df16.columns = ["Feature", "Column Type"]

dtype_df16.groupby("Column Type").aggregate('count').reset_index()
# feature as object 

# look into non-numeric features

dtype_df16[dtype_df16['Column Type']=='object']['Feature']
# values counts for propertycountylandusecode

print(prop2016['propertycountylandusecode'].value_counts())
# values counts for propertyzoningdesc

print(prop2016['propertyzoningdesc'].value_counts())
prop2016['propertytype'] = 'NR'

prop2016.loc[prop2016['propertyzoningdesc'].str.contains('r', na=False, case=False), 'propertytype'] = 'R'
prop2016.head()
plt.pie

prop2016.propertytype.value_counts().plot(kind='pie', autopct='%1.0f%%', fontsize=12)

plt.axis('equal')

plt.title('Residential (R) and NonResidential (NR) for year 2016')
#https://thispointer.com/pandas-count-rows-in-a-dataframe-all-or-those-only-that-satisfy-a-condition/

TotalNumOfRows16 = len(prop2016.index)

seriesObj16 = prop2016.apply(lambda x: True if x['propertytype'] == 'R' else False, axis=1)

numOfRows16 = len(seriesObj16[seriesObj16 == True].index)

print('Property 2016 Dataset')

print('Number of Residential Properties: ', numOfRows16)

print ("Number of Non-Residential Properties: ", TotalNumOfRows16 - numOfRows16)
# value count for object features

print(prop2016['hashottuborspa'].value_counts())

print(prop2016['fireplaceflag'].value_counts())

print(prop2016['taxdelinquencyflag'].value_counts())
# drop the five object columns

prop2016.drop(dtype_df16[dtype_df16['Column Type']=='object']['Feature'].values.tolist(), axis=1,inplace=True)
prop2016.shape
prop2016.head()
# check the missing percentage

missing_df16 = prop2016.isnull().sum(axis=0).reset_index()

missing_df16.columns = ['column_name', 'missing_count']

missing_df16 = missing_df16.ix[missing_df16['missing_count']>0]

missing_df16 = missing_df16.sort_values(by='missing_count')

ind16 = np.arange(missing_df16.shape[0])

width16 = 0.9

fig, ax = plt.subplots(figsize=(12,14))

rects16 = ax.barh(ind16, missing_df16.missing_count.values/2985217*100, color='g')

ax.set_yticks(ind16)

ax.set_yticklabels(missing_df16.column_name.values, rotation='horizontal',fontsize=12)

ax.set_xlabel("Percentage of missing values",fontsize=12)

ax.set_title('Property 2016 Data Set Missing Values')
# missing rate of the data

missing_df16['missing_rate']=missing_df16['missing_count']/2985217

cutoff=0.9

print(missing_df16[missing_df16.missing_rate<cutoff].shape)

print('There are',missing_df16[missing_df16.missing_rate<cutoff].shape[0],'features of which the percentage of missing values is less than',cutoff*100,'% in the 2016 property dataset.')

missing_df16[(missing_df16.missing_rate<cutoff)].column_name.values
# drop feature missing rate>0.9

prop2016.drop(missing_df16[(missing_df16.missing_rate>=cutoff)].column_name.values.tolist(),

                    axis=1,inplace=True)
# fill missing values

# for id_feature, fill the missing values with most frequent value

# for cnt_feature, fill the missing values with median value

# for size_feature, fill the missing values with mean values

# for location_feature, fill the missing values with the nearest values

# categorize the left feature

feature_left16=prop2016.columns.tolist()

id_feature_left16=list()

cnt_feature_left16=list()

size_feature_left16=list()

location_feature_left16=list()

for x in feature_left16:

    if x in id_feature:

        id_feature_left16.append(x)

    elif x in cnt_feature:

        cnt_feature_left16.append(x)

    elif x in size_feature:

        size_feature_left16.append(x)

    elif x in location_feature:

        location_feature_left16.append(x)



# fill missing values

# for id_feature, fill the missing values with most frequent value

# for cnt_feature, fill the missing values with median value

# for size_feature, fill the missing values with mean values

# for location_feature, fill the missing values with the most frequent values

fill_missing_value16=dict()

# for id_feature

for x in id_feature_left16:

    fill_missing_value16[x]=0    # prop2016[x].value_counts().index.tolist()[0]

# for cnt_feature

for x in cnt_feature_left16:

    fill_missing_value16[x]=prop2016[x].median()

# for size_feature

for x in size_feature_left16:

    fill_missing_value16[x]=prop2016[x].mean()

# for size_feature

for x in location_feature_left16:

    fill_missing_value16[x]=0     #prop2016[x].value_counts().index.tolist()[0]

for x in fill_missing_value16:

    prop2016[x].fillna(fill_missing_value16[x],inplace=True)
clean2016 = prop2016[['parcelid', 'calculatedfinishedsquarefeet', 'regionidzip', 'structuretaxvaluedollarcnt', 'yearbuilt']].copy()
clean2016.head()
#https://stackoverflow.com/questions/44593284/python-pandas-dataframe-merge-and-pick-only-few-columns

clean2016=clean2016.merge(train2016[['parcelid', 'logerror']], on = 'parcelid', how='outer')
imp = IterativeImputer(max_iter=5, random_state=0)

imp.fit(clean2016)  

IterativeImputer(add_indicator=False, estimator=None,

                 imputation_order='random', initial_strategy='median',

                 max_iter=5, max_value=0.5, min_value=-0.5,

                 missing_values='nan', n_nearest_features=None,

                 sample_posterior=False, verbose=0)

X_test16 = clean2016

# the model learns that the second feature is double the first

np.round(imp.transform(X_test16))
clean2016i = imp.fit_transform(clean2016)

print(clean2016i)
clean2016i = pd.DataFrame(clean2016i, columns=['parcelid', 'calculatedfinishedsquarefeet', 'regionidzip',

                                               'structuretaxvaluedollarcnt', 'yearbuilt', 'logerror'])
clean2016i.head()
prop2016=prop2016.merge(clean2016i[['parcelid', 'logerror']], on = 'parcelid', how='outer')
prop2016.head()
## Read the distribution of logerror after imputation into prop2016 dataset

ulimit = np.percentile(train2016.logerror.values, 99)

llimit = np.percentile(train2016.logerror.values, 1)



plt.figure(figsize=(12,4))

plt.title("Training 2016 Dataset Logerror")

sns.distplot(train2016.logerror.values, bins=500, kde=False)

plt.xlabel("Log Error", fontsize=12)

mu, std = norm.fit(train2016.logerror.values)

xmin,xmax=plt.xlim()

x = np.linspace(-5,5,1000)

p = norm.pdf(x, 0, 0.025)

plt.plot(x, p*(x[1]-x[0])*90811, 'r', linewidth=2)

plt.axis([-0.5,0.5,0,20000])

plt.ylabel("Number of Parcels", fontsize=12)

ulimit = np.percentile(prop2016.logerror.values, 99)

llimit = np.percentile(prop2016.logerror.values, 1)



plt.figure(figsize=(12,4))

sns.distplot(prop2016.logerror.values, bins=500, kde=False)

plt.title("Imputed logerror into Property 2016 Dataset")

plt.xlabel("Log Error", fontsize=12)

mu, std = norm.fit(prop2016.logerror.values)

xmin,xmax=plt.xlim()

x = np.linspace(-5,5,1000)

p = norm.pdf(x, 0, 0.025)

plt.plot(x, p*(x[1]-x[0])*90811, 'r', linewidth=2)

plt.axis([-0.50,0.50,0.00,30000])

plt.ylabel("Number of Parcels", fontsize=12)
train2016v2=train2016.merge(prop2016[['parcelid', 'propertytype']], on = 'parcelid', how='inner')
train2016v2.head()
sns.barplot(y="logerror", x="propertytype", data=train2016v2, order=["R", "NR"]).set_title('Train 2016')
sns.barplot(y="logerror", x="propertytype", data=prop2016, order=["R", "NR"]).set_title('Prop2016')
sns.stripplot(x='propertytype', y='logerror', data=train2016v2, order=["R", "NR"]).set_title('Train2016')
sns.stripplot(x='propertytype', y='logerror', data=prop2016, order=["R", "NR"]).set_title('Prop2016')
sns.stripplot(x='propertytype', y='logerror', data=train2016v2, order=["R", "NR"]).set_title('Combined 2016')

sns.stripplot(x='propertytype', y='logerror', data=prop2016, order=["R", "NR"])
#Find the mean of NR and R properties, https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas

print("Train2016 dataset mean of logerror is: ", train2016['logerror'].mean(axis=0))

print("Prop2016 dataset mean of logerror is: ", prop2016['logerror'].mean(axis=0))
print("Train2016 dataset mean of logerror for NR and R properties are: ", train2016v2.groupby('propertytype', as_index=False)['logerror'].mean())

print("")

print("Prop2016 dataset mean of logerror for NR and R properties are: ", prop2016.groupby('propertytype', as_index=False)['logerror'].mean())