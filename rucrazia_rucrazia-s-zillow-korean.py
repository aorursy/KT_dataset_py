# For DataFrame

import numpy as np # for linear algebra

import pandas as pd # for data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



# For Analysis

from sklearn import neighbors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder





# For Visualization

import plotly # visualization

from plotly.graph_objs import Scatter, Figure, Layout # visualization

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot # visualization

import plotly.figure_factory as ff # visualization

import plotly.graph_objs as go # visualization

init_notebook_mode(connected=True) # visualization

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns # for data visualization

color = sns.color_palette()

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_2016_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

train_2016_df.shape
train_2016_df.head()
train_2017_df = pd.read_csv("../input/train_2017.csv", parse_dates=["transactiondate"])

train_2017_df.shape
train_2017_df.head()
prop_2016_df = pd.read_csv("../input/properties_2016.csv")

prop_2016_df.shape
prop_2016_df.head()
prop_2017_df = pd.read_csv("../input/properties_2017.csv")

prop_2017_df.shape
prop_2017_df.head()
merge_2016_df = pd.merge(train_2016_df,prop_2016_df,on="parcelid",how="left")

merge_2016_df.shape
merge_2016_df.head()
merge_2017_df = pd.merge(train_2017_df,prop_2017_df,on="parcelid",how="left")

merge_2017_df.shape
missingvalues_prop = (merge_2017_df.isnull().sum()/len(merge_2017_df)).reset_index()

missingvalues_prop.columns = ['field','proportion']

missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)

print(missingvalues_prop)
merge_2016_df.head()
merge_2016_df.dtypes
# Rename columns

merge_2016_df = merge_2016_df.rename(columns = {

  "parcelid" : "id_parcel",

  "yearbuilt":"build_year",

  "basementsqft":"area_basement",

  "yardbuildingsqft17":"area_patio",

  "yardbuildingsqft26":"area_shed",

  "poolsizesum":"area_pool",

  "lotsizesquarefeet":"area_lot",

  "garagetotalsqft":"area_garage",

  "finishedfloor1squarefeet":"area_firstfloor_finished",

  "calculatedfinishedsquarefeet":"area_total_calc",

  "finishedsquarefeet6":"area_base",

  "finishedsquarefeet12":"area_live_finished",

  "finishedsquarefeet13":"area_liveperi_finished",

  "finishedsquarefeet15":"area_total_finished",

  "finishedsquarefeet50":"area_unknown",

  "unitcnt":"num_unit",

  "numberofstories":"num_story",

  "roomcnt":"num_room",

  "bathroomcnt":"num_bathroom",

  "bedroomcnt":"num_bedroom",

  "calculatedbathnbr":"num_bathroom_calc",

  "fullbathcnt":"num_bath",

  "threequarterbathnbr":"num_75_bath",

  "fireplacecnt":"num_fireplace",

  "poolcnt":"num_pool",

  "garagecarcnt":"num_garage",

  "regionidcounty":"region_county",

  "regionidcity":"region_city",

  "regionidzip":"region_zip",

  "regionidneighborhood":"region_neighbor",

  "taxvaluedollarcnt":"tax_total",

  "structuretaxvaluedollarcnt":"tax_building",

  "landtaxvaluedollarcnt":"tax_land",

  "taxamount":"tax_property",

  "assessmentyear":"tax_year",

  "taxdelinquencyflag":"tax_delinquency",

  "taxdelinquencyyear":"tax_delinquency_year",

  "propertyzoningdesc":"zoning_property",

  "propertylandusetypeid":"zoning_landuse",

  "propertycountylandusecode":"zoning_landuse_county",

  "fireplaceflag":"flag_fireplace",

  "hashottuborspa":"flag_tub",

  "buildingqualitytypeid":"quality",

  "buildingclasstypeid":"framing",

  "typeconstructiontypeid":"material",

  "decktypeid":"deck",

  "storytypeid":"story",

  "heatingorsystemtypeid":"heating",

  "airconditioningtypeid":"aircon",

  "architecturalstyletypeid":"architectural_style"

})
merge_2016_df.head()
# Rename columns

merge_2017_df = merge_2017_df.rename(columns = {

  "parcelid" : "id_parcel",

  "yearbuilt":"build_year",

  "basementsqft":"area_basement",

  "yardbuildingsqft17":"area_patio",

  "yardbuildingsqft26":"area_shed",

  "poolsizesum":"area_pool",

  "lotsizesquarefeet":"area_lot",

  "garagetotalsqft":"area_garage",

  "finishedfloor1squarefeet":"area_firstfloor_finished",

  "calculatedfinishedsquarefeet":"area_total_calc",

  "finishedsquarefeet6":"area_base",

  "finishedsquarefeet12":"area_live_finished",

  "finishedsquarefeet13":"area_liveperi_finished",

  "finishedsquarefeet15":"area_total_finished",

  "finishedsquarefeet50":"area_unknown",

  "unitcnt":"num_unit",

  "numberofstories":"num_story",

  "roomcnt":"num_room",

  "bathroomcnt":"num_bathroom",

  "bedroomcnt":"num_bedroom",

  "calculatedbathnbr":"num_bathroom_calc",

  "fullbathcnt":"num_bath",

  "threequarterbathnbr":"num_75_bath",

  "fireplacecnt":"num_fireplace",

  "poolcnt":"num_pool",

  "garagecarcnt":"num_garage",

  "regionidcounty":"region_county",

  "regionidcity":"region_city",

  "regionidzip":"region_zip",

  "regionidneighborhood":"region_neighbor",

  "taxvaluedollarcnt":"tax_total",

  "structuretaxvaluedollarcnt":"tax_building",

  "landtaxvaluedollarcnt":"tax_land",

  "taxamount":"tax_property",

  "assessmentyear":"tax_year",

  "taxdelinquencyflag":"tax_delinquency",

  "taxdelinquencyyear":"tax_delinquency_year",

  "propertyzoningdesc":"zoning_property",

  "propertylandusetypeid":"zoning_landuse",

  "propertycountylandusecode":"zoning_landuse_county",

  "fireplaceflag":"flag_fireplace",

  "hashottuborspa":"flag_tub",

  "buildingqualitytypeid":"quality",

  "buildingclasstypeid":"framing",

  "typeconstructiontypeid":"material",

  "decktypeid":"deck",

  "storytypeid":"story",

  "heatingorsystemtypeid":"heating",

  "airconditioningtypeid":"aircon",

  "architecturalstyletypeid":"architectural_style"

})
merge_2017_df.head()
merge_df_raw = merge_2016_df.append(merge_2017_df, ignore_index=True)

merge_df_raw.shape
merge_df_raw.head()
merge_df_raw[['aircon', 'architectural_style', 'num_bathroom', 'num_bedroom', 'framing', 'quality', 'num_bathroom_calc', 'deck', 'num_fireplace', 'num_bath','num_garage','flag_tub','heating','num_pool','pooltypeid10','pooltypeid2','pooltypeid7','zoning_landuse_county','zoning_landuse','zoning_property','region_city','region_county','region_neighbor','region_zip','num_room','story','num_75_bath','material','num_unit','build_year','num_story','flag_fireplace','tax_year','tax_delinquency_year']] = merge_df_raw[['aircon', 'architectural_style', 'num_bathroom', 'num_bedroom', 'framing', 'quality', 'num_bathroom_calc', 'deck', 'num_fireplace', 'num_bath','num_garage','flag_tub','heating','num_pool','pooltypeid10','pooltypeid2','pooltypeid7','zoning_landuse_county','zoning_landuse','zoning_property','region_city','region_county','region_neighbor','region_zip','num_room','story','num_75_bath','material','num_unit','build_year','num_story','flag_fireplace','tax_year','tax_delinquency_year']].astype(str)
merge_df_raw.dtypes
corrMatt = merge_df_raw.corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True)
# LogErorr

df_train = merge_df_raw.copy()

df_train.loc[:,'abs_logerror'] = df_train['logerror'].abs()

worst_prediction = df_train['abs_logerror'].quantile(q=.95)





trace0 = go.Scatter(

    y = df_train[(df_train['fips']==6037)&(df_train['abs_logerror']>worst_prediction)].\

                groupby('build_year')['abs_logerror'].mean(),

    x = df_train[(df_train['fips']==6037)&(df_train['abs_logerror']>worst_prediction)].\

                groupby('build_year')['abs_logerror'].mean().index,

    mode = 'lines+markers',

    name = "Los Angeles", 

)

trace1 = go.Scatter(

    y = df_train[(df_train['fips']==6059)&(df_train['abs_logerror']>worst_prediction)].\

                groupby('build_year')['abs_logerror'].mean(),

    x = df_train[(df_train['fips']==6059)&(df_train['abs_logerror']>worst_prediction)].\

                groupby('build_year')['abs_logerror'].mean().index,

    mode = 'lines+markers',

    name = "Orange County"

)

trace2 = go.Scatter(

    y = df_train[(df_train['fips']==6111)&(df_train['abs_logerror']>worst_prediction)].\

                groupby('build_year')['abs_logerror'].mean(),

    x = df_train[(df_train['fips']==6111)&(df_train['abs_logerror']>worst_prediction)].\

                groupby('build_year')['abs_logerror'].mean().index,

    mode = 'lines+markers',

    name = "Ventura County"

)

data = [trace0, trace1, trace2]



plotly.offline.iplot(data, filename='line-mode')
merge_df_raw['zoning_landuse'].unique()
cnt_srs = merge_df_raw['zoning_landuse'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('zoning_landuse', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
# 아파트 corr

df = merge_df_raw.loc[merge_df_raw['zoning_landuse']=='266.0']

corrMatt = df.corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True)
# 비 아파트 corr

df = merge_df_raw.loc[merge_df_raw['zoning_landuse']!='266.0']

corrMatt = df.corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True)
continuous = ['area_basement','area_firstfloor_finished','area_total_calc','area_live_finished','area_liveperi_finished','area_total_finished','area_unknown','area_base','area_garage','area_lot','area_pool','area_patio','area_shed']

discrete = ['num_bathroom','num_bedroom','num_bathroom_calc','num_bath','num_garage','num_pool','num_room','num_75_bath','num_unit','num_story']

categories = ['aircon','architectural_style','framing','quality','deck','heating','pooltypeid2','pooltypeid7','story','material','build_year']

### Continuous variable plots

for col in continuous:

    values = merge_df_raw[col].dropna()

    lower = np.percentile(values, 1)

    upper = np.percentile(values, 99)

    fig = plt.figure(figsize=(18,9));

    sns.distplot(values[(values>lower) & (values<upper)], color='Sienna', ax = plt.subplot(121));

    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

    plt.suptitle(col, fontsize=16)
merge_df_raw['aircon'].dropna().head(10)
merge_df_raw[discrete] = merge_df_raw[discrete].astype(float)
### Discrete variable plots

NanAsZero = ['num_pool', 'num_75_bath']

for col in discrete:

    if col in NanAsZero:

        merge_df_raw[col].fillna(0, inplace=True)

    values = merge_df_raw[col].dropna()  

    fig = plt.figure(figsize=(18,9));

    sns.countplot(x=values, color='Sienna', ax = plt.subplot(121));

    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

    plt.suptitle(col, fontsize=16)
### Categorical variable plots

for col in categories:

    values = merge_df_raw[col].astype('str').value_counts(dropna=False).to_frame().reset_index()

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
### Continuous variable vs logerror plots

for col in continuous:     

    fig = plt.figure(figsize=(18,9));

    sns.barplot(x='logerror', y=col, data=merge_df_raw, ax = plt.subplot(121),

                order=['Large Negative Error', 'Medium Negative Error','Small Error',

                       'Medium Positive Error', 'Large Positive Error']);

    plt.xlabel('LogError Bin');

    plt.ylabel('Average {}'.format(col));

    sns.regplot(x='logerror', y=col, data=merge_df_raw, color='Sienna', ax = plt.subplot(122));

    plt.suptitle('LogError vs {}'.format(col), fontsize=16)   
df = merge_df_raw.loc[(merge_df_raw['zoning_landuse']!='266.0')]

fig = plt.figure(figsize=(18,9));

sns.barplot(x='logerror', y='area_total_calc', data=df, ax = plt.subplot(121),

            order=['Large Negative Error', 'Medium Negative Error','Small Error',

                   'Medium Positive Error', 'Large Positive Error']);

plt.xlabel('LogError Bin');

plt.ylabel('Average {}'.format(col));

sns.regplot(x='logerror', y='area_total_calc', data=df, color='Sienna', ax = plt.subplot(122));

plt.suptitle('LogError vs {}'.format(col), fontsize=16)   
continuous = ['area_basement','area_firstfloor_finished','area_total_calc','area_live_finished','area_liveperi_finished','area_total_finished','area_unknown','area_base','area_garage','area_lot','area_pool','area_patio','area_shed']

discrete = ['num_bathroom','num_bedroom','num_bathroom_calc','num_bath','num_garage','num_pool','num_room','num_75_bath','num_unit','num_story']

categories = ['aircon','architectural_style','framing','quality','deck','heating','pooltypeid2','pooltypeid7','story','material','build_year']

plt.figure(figsize=(12,12))

sns.jointplot(x=merge_df_raw.latitude.values, y=merge_df_raw.longitude.values, size=10)

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
discrete = ['num_fireplace']

categories = ['flag_tub','zoning_landuse_county','zoning_landuse','zoning_property','region_city','region_county','region_neighbor','region_zip','flag_fireplace']
### Categorical variable plots

for col in categories:

    values = merge_df_raw[col].astype('str').value_counts(dropna=False).to_frame().reset_index()

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
### Continuous variable vs logerror plots

for col in continuous:     

    fig = plt.figure(figsize=(18,9));

    sns.barplot(x='logerror', y=col, data=merge_df_raw, ax = plt.subplot(121),

                order=['Large Negative Error', 'Medium Negative Error','Small Error',

                       'Medium Positive Error', 'Large Positive Error']);

    plt.xlabel('LogError Bin');

    plt.ylabel('Average {}'.format(col));

    sns.regplot(x='logerror', y=col, data=merge_df_raw, color='Sienna', ax = plt.subplot(122));

    plt.suptitle('LogError vs {}'.format(col), fontsize=16)   
merge_df =  merge_df_raw.copy()
merge_df = merge_df[np.isfinite(merge_df['latitude'])]

merge_df = merge_df[np.isfinite(merge_df['longitude'])]
# NULL Check

missing_df = merge_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
merge_df = merge_df.drop(['framing', 'area_liveperi_finished', 'area_basement','pooltypeid10','story','area_shed','material','area_base','area_total_finished','area_unknown','area_firstfloor_finished'], axis=1)

merge_df = merge_df.drop(['num_bathroom_calc','num_bath'], axis=1)
merge_df = merge_2016_df.append(merge_2017_df, ignore_index=True)

merge_df.shape
index = merge_df.flag_tub.isnull()

merge_df.loc[index,'flag_tub'] = "None"

# pooltypeid10(does home have a Spa or hot tub) seems to be inconcistent with the 'hashottuborspa' field - these two fields should have the same information I assume?

print(merge_df.flag_tub.value_counts())



#Assume if the pooltype id is null then pool/hottub doesnt exist 

index = merge_df.pooltypeid2.isnull()

merge_df.loc[index,'pooltypeid2'] = 0

print(merge_df.pooltypeid2.value_counts())



index = merge_df.pooltypeid7.isnull()

merge_df.loc[index,'pooltypeid7'] = 0

print(merge_df.pooltypeid7.value_counts())



index = merge_df.num_pool.isnull()

merge_df.loc[index,'num_pool'] = 0

print(merge_df.num_pool.value_counts())



index = merge_df.num_fireplace.isnull()

merge_df.loc[index,'num_fireplace'] = 0

print(merge_df.num_fireplace.value_counts())



#Tax deliquency flag - assume if it is null then doesn't exist

index = merge_df.tax_delinquency.isnull()

merge_df.loc[index,'tax_delinquency'] = "None"

print(merge_df.tax_delinquency.value_counts())





#Assume if Null in garage count it means there are no garages

index = merge_df.num_garage.isnull()

merge_df.loc[index,'num_garage'] = 0

print(merge_df.num_garage.value_counts())



#Likewise no garage means the size is 0 by default

index = merge_df.area_garage.isnull()

merge_df.loc[index,'area_garage'] = 0

print(merge_df.area_garage.value_counts())

#There seems to be inconsistency between the fireplaceflag and fireplace cnt - my guess is that these should be the same

print(merge_df.flag_fireplace.isnull().sum())



#There seems to be 80668 properties without fireplace according to the 'fireplacecnt' but the 'fireplace flag' says they are 90053 missing values

#Lets instead create the fireplaceflag from scratch using 'fireplacecnt' as there are less missing values here

merge_df['flag_fireplace']= "No"

merge_df.loc[merge_df['num_fireplace']>0,'flag_fireplace']= "Yes"

print(merge_df.flag_fireplace.isnull().sum())
#Let's fill in some missing values using the most common value for those variables where this might be a sensible approach

#AC Type - Mostly 1's, which corresponds to central AC. Reasonable to assume most other properties are similar.

merge_df['aircon'].value_counts()

index = merge_df.aircon.isnull()

merge_df.loc[index,'aircon'] = 1

print(merge_df.aircon.value_counts())



#heating or system - Mostly 2, which corresponds to central heating so seems reasonable to assume most other properties have central heating  

print(merge_df['heating'].value_counts())

index = merge_df.heating.isnull()

merge_df.loc[index,'heating'] = 2



# 'threequarterbathnbr' - not an important variable according to https://www.kaggle.com/nikunjm88/creating-additional-features, so fill with most common value

print(merge_df['num_75_bath'].value_counts())

index = merge_df.num_75_bath.isnull()

merge_df.loc[index,'num_75_bath'] = 1
merge_df.shape
missingvalues_prop = (merge_df.isnull().sum()/len(merge_df)).reset_index()

missingvalues_prop.columns = ['field','proportion']

missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)

print(missingvalues_prop)

missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()

merge_df = merge_df.drop(missingvaluescols, axis=1)
def fillna_knn2( df, base, target, fraction = 1, threshold = 10, n_neighbors = 5 ):

    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 

    whole = [ target ] + base

    

    miss = df[target].isnull()

    notmiss = ~miss 

    nummiss = miss.sum()

    

    enc = OneHotEncoder()

    X_target = df.loc[ notmiss, whole ].sample( frac = fraction )

    

    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )

    

    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()

    X = X_target[ base  ]

    

    print( 'fitting' )

    n_neighbors = n_neighbors

    clf = neighbors.KNeighborsClassifier( n_neighbors, weights = 'uniform' )

    clf.fit( X, Y )

    

    print( 'the shape of active features: ' ,enc.active_features_.shape )

    

    print( 'predicting' )

    Z = clf.predict(df.loc[miss, base])

    

    numunperdicted = Z[:,0].sum()

    if numunperdicted / nummiss *100 < threshold :

        print( 'writing result to df' )    

        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )

        print( 'num of unperdictable data: ', numunperdicted )

        return enc

    else:

        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )



#function to deal with variables that are actually string/categories

def zoningcode2int( df, target ):

    storenull = df[ target ].isnull()

    enc = LabelEncoder( )

    df[ target ] = df[ target ].astype( str )



    print('fit and transform')

    df[ target ]= enc.fit_transform( df[ target ].values )

    print( 'num of categories: ', enc.classes_.shape  )

    df.loc[ storenull, target ] = np.nan

    print('recover the nan value')

    return enc
#buildingqualitytypeid - assume it is the similar to the nearest property. Probably makes senses if its a property in a block of flats, i.e if block was built all at the same time and therefore all flats will have similar quality 

#Use the same logic for propertycountylandusecode (assume it is same as nearest property i.e two properties right next to each other are likely to have the same code) & propertyzoningdesc. 

#These assumptions are only reasonable if you actually have nearby properties to the one with the missing value

'''

fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'quality', fraction = 0.15, n_neighbors = 1 )





zoningcode2int( df = merge_df,

                            target = 'zoning_landuse_county' )

fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'zoning_landuse_county', fraction = 0.15, n_neighbors = 1 )



zoningcode2int( df = merge_df,

                            target = 'zoning_property' )



fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'zoning_property', fraction = 0.15, n_neighbors = 1 )



#regionidcity, regionidneighborhood & regionidzip - assume it is the same as the nereast property. 

#As mentioned above, this is ok if there's a property very nearby to the one with missing values (I leave it up to the reader to check if this is the case!)

fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'region_city', fraction = 0.15, n_neighbors = 1 )



fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'region_neighbor', fraction = 0.15, n_neighbors = 1 )



fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'region_zip', fraction = 0.15, n_neighbors = 1 )



#unitcnt - the number of structures the unit is built into. Assume it is the same as the nearest properties. If the property with missing values is in a block of flats or in a terrace street then this is probably ok - but again I leave it up to the reader to check if this is the case!

fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'num_unit', fraction = 0.15, n_neighbors = 1 )



#yearbuilt - assume it is the same as the nearest property. This assumes properties all near to each other were built around the same time

fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'build_year', fraction = 0.15, n_neighbors = 1 )



#lot size square feet - not sure what to do about this one. Lets use nearest neighbours. Assume it has same lot size as property closest to it

fillna_knn( df = merge_df,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'area_lot', fraction = 0.15, n_neighbors = 1 )

'''
merge_df.to_csv('merge_df.csv',index=False)