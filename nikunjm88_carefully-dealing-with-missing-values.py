# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import neighbors

import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Start of by reading in the data and merging the datasets

prop = pd.read_csv('../input/properties_2016.csv')

train = pd.read_csv("../input/train_2016_v2.csv")



for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:		

        prop[c] = prop[c].astype(np.float32)



df_train = train.merge(prop, how='left', on='parcelid')

del prop, train

df_train = df_train.drop(['parcelid', 'transactiondate'], axis=1)

#Identify numerical columns to produce a heatmap

catcols = ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid','buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood','regionidzip','storytypeid','typeconstructiontypeid','yearbuilt','taxdelinquencyflag']

numcols = [x for x in df_train.columns if x not in catcols]



#Lets start by plotting a heatmap to determine if any variables are correlated

plt.figure(figsize = (12,8))

sns.heatmap(data=df_train[numcols].corr())

plt.show()

plt.gcf().clear()
missing_df = df_train.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

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



#'calculatedfinishedsquarefeet' has the fewest missing values so lets remove the others, note also that except for 'finishedsquarefeet12' the rest have large amount of missing values anyways. 

#Also if you look at my script in https://www.kaggle.com/nikunjm88/creating-additional-features 'calculatedfinishedsquarefeet' appears to be the most important variable

dropcols = ['finishedsquarefeet12','finishedsquarefeet13', 'finishedsquarefeet15','finishedsquarefeet6']



#finishedsquarefeet50 and finishedfloor1squarefeet are the exactly the same information according to the dictionary descriptions, lets remove finishedsquarefeet50 as it has more missing values

dropcols.append('finishedsquarefeet50')



#'bathroomcnt' and 'calculatedbathnbr' and 'fullbathcnt' seem to be the same information aswell according to the dictionary descriptions. Choose 'bathroomcnt' as has no missing values, so remove the other two

dropcols.append('calculatedbathnbr')

dropcols.append('fullbathcnt')

#The below variables are flags and lets assume if they are NA's it means the object does not exist so lets fix this

index = df_train.hashottuborspa.isnull()

df_train.loc[index,'hashottuborspa'] = "None"



# pooltypeid10(does home have a Spa or hot tub) seems to be inconcistent with the 'hashottuborspa' field - these two fields should have the same information I assume?

print(df_train.hashottuborspa.value_counts())

print(df_train.pooltypeid10.value_counts())



#lets remove 'pooltypeid10' as has more missing values

dropcols.append('pooltypeid10')



#Assume if the pooltype id is null then pool/hottub doesnt exist 

index = df_train.pooltypeid2.isnull()

df_train.loc[index,'pooltypeid2'] = 0



index = df_train.pooltypeid7.isnull()

df_train.loc[index,'pooltypeid7'] = 0



index = df_train.poolcnt.isnull()

df_train.loc[index,'poolcnt'] = 0



#Theres more missing values in the 'poolsizesum' then in 'poolcnt', Let's fill in median values for poolsizesum where pool count is >0 and missing. I think this is sensible assumption as residential pool sizes are fairly standard size I guess in the U.S.

#Also the poolsizesum doesn't seem to be much of an important variable (https://www.kaggle.com/nikunjm88/creating-additional-features) so imputing with the median hopefully won't cause too much of an issue

print(df_train.poolsizesum.isnull().sum())

print(df_train.poolcnt.value_counts())



#Fill in those properties that have a pool with median pool value

poolsizesum_median = df_train.loc[df_train['poolcnt'] > 0, 'poolsizesum'].median()

df_train.loc[(df_train['poolcnt'] > 0) & (df_train['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median



#If it doesn't have a pool then poolsizesum is 0 by default

df_train.loc[(df_train['poolcnt'] == 0), 'poolsizesum'] = 0



#There seems to be inconsistency between the fireplaceflag and fireplace cnt - my guess is that these should be the same

print(df_train.fireplaceflag.isnull().sum())

print(df_train.fireplacecnt.isnull().sum())



#There seems to be 80668 properties without fireplace according to the 'fireplacecnt' but the 'fireplace flag' says they are 90053 missing values

#Lets instead create the fireplaceflag from scratch using 'fireplacecnt' as there are less missing values here

df_train['fireplaceflag']= "No"

df_train.loc[df_train['fireplacecnt']>0,'fireplaceflag']= "Yes"



index = df_train.fireplacecnt.isnull()

df_train.loc[index,'fireplacecnt'] = 0



#Tax deliquency flag - assume if it is null then doesn't exist

index = df_train.taxdelinquencyflag.isnull()

df_train.loc[index,'taxdelinquencyflag'] = "None"



#Same number of missing values between garage count and garage size - assume this is because when there are properties with no garages then both variables are NA

print(df_train.garagecarcnt.isnull().sum())

print(df_train.garagetotalsqft.isnull().sum())



#Assume if Null in garage count it means there are no garages

index = df_train.garagecarcnt.isnull()

df_train.loc[index,'garagecarcnt'] = 0



#Likewise no garage means the size is 0 by default

index = df_train.garagetotalsqft.isnull()

df_train.loc[index,'garagetotalsqft'] = 0



#Let's fill in some missing values using the most common value for those variables where this might be a sensible approach

#AC Type - Mostly 1's, which corresponds to central AC. Reasonable to assume most other properties are similar.

df_train['airconditioningtypeid'].value_counts()

index = df_train.airconditioningtypeid.isnull()

df_train.loc[index,'airconditioningtypeid'] = 1



#heating or system - Mostly 2, which corresponds to central heating so seems reasonable to assume most other properties have central heating  

print(df_train['heatingorsystemtypeid'].value_counts())

index = df_train.heatingorsystemtypeid.isnull()

df_train.loc[index,'heatingorsystemtypeid'] = 2

# 'threequarterbathnbr' - not an important variable according to https://www.kaggle.com/nikunjm88/creating-additional-features, so fill with most common value

print(df_train['threequarterbathnbr'].value_counts())

index = df_train.threequarterbathnbr.isnull()

df_train.loc[index,'threequarterbathnbr'] = 1

missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()

missingvalues_prop.columns = ['field','proportion']

missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)

print(missingvalues_prop)

missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()

dropcols = dropcols + missingvaluescols

df_train = df_train.drop(dropcols, axis=1)

def fillna_knn( df, base, target, fraction = 1, threshold = 10, n_neighbors = 5 ):

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



fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'buildingqualitytypeid', fraction = 0.15, n_neighbors = 1 )





zoningcode2int( df = df_train,

                            target = 'propertycountylandusecode' )

fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'propertycountylandusecode', fraction = 0.15, n_neighbors = 1 )



zoningcode2int( df = df_train,

                            target = 'propertyzoningdesc' )



fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'propertyzoningdesc', fraction = 0.15, n_neighbors = 1 )



#regionidcity, regionidneighborhood & regionidzip - assume it is the same as the nereast property. 

#As mentioned above, this is ok if there's a property very nearby to the one with missing values (I leave it up to the reader to check if this is the case!)

fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'regionidcity', fraction = 0.15, n_neighbors = 1 )



fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'regionidneighborhood', fraction = 0.15, n_neighbors = 1 )



fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'regionidzip', fraction = 0.15, n_neighbors = 1 )



#unitcnt - the number of structures the unit is built into. Assume it is the same as the nearest properties. If the property with missing values is in a block of flats or in a terrace street then this is probably ok - but again I leave it up to the reader to check if this is the case!

fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'unitcnt', fraction = 0.15, n_neighbors = 1 )



#yearbuilt - assume it is the same as the nearest property. This assumes properties all near to each other were built around the same time

fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'yearbuilt', fraction = 0.15, n_neighbors = 1 )



#lot size square feet - not sure what to do about this one. Lets use nearest neighbours. Assume it has same lot size as property closest to it

fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'lotsizesquarefeet', fraction = 0.15, n_neighbors = 1 )

plt.figure(figsize=(12,12))

sns.jointplot(x=df_train.finishedfloor1squarefeet.values, y=df_train.calculatedfinishedsquarefeet.values)

plt.ylabel('calculatedfinishedsquarefeet', fontsize=12)

plt.xlabel('finishedfloor1squarefeet', fontsize=12)

plt.title("finishedfloor1squarefeet Vs calculatedfinishedsquarefeet", fontsize=15)

plt.show()



#There are some properties where finishedfloor1squarefeet and calculatedfinishedsquarefeetare are both exactly the same - probably because its a studio flat of some sort so that the area on the first floor is equivalent to the total area, lets see how many there are

#For now assume if the number of stories is 1 then the finishedfloor1squarefeet is the same as calculatedfinishedsquarefeet

df_train.loc[(df_train['finishedfloor1squarefeet'].isnull()) & (df_train['numberofstories']==1),'finishedfloor1squarefeet'] = df_train.loc[(df_train['finishedfloor1squarefeet'].isnull()) & (df_train['numberofstories']==1),'calculatedfinishedsquarefeet']



#I also discovered that there seems to be two properties that have finishedfloor1squarefeet greater than calculated finishedsquarefeet. Notice also that they have big logerrors aswell - my guess is that the Zillow House price model found it difficult to predict these points due to the fact that they probably had potentially 'incorrect' data input values?

#Discussion point - should we be removing these points or leave them in as they are or 'fix' them? I think it really depends on whether the test data has similar points which may be wrong as we'll want to predict big log errors for these incorrect points aswell I guess...

#For now just remove them.

print(df_train.loc[df_train['calculatedfinishedsquarefeet']<df_train['finishedfloor1squarefeet']])

droprows = df_train.loc[df_train['calculatedfinishedsquarefeet']<df_train['finishedfloor1squarefeet']].index

df_train = df_train.drop(droprows)



#Let's check whats missing still

print(df_train.isnull().sum())

#taxvaluedollarcnt & landtaxvaluedollarcnt - set it equal to the tax amount (most correlated value). Single story property so assume they are all the same

df_train.loc[df_train.taxvaluedollarcnt.isnull(),'taxvaluedollarcnt'] = df_train.loc[df_train.taxvaluedollarcnt.isnull(),'taxamount']

df_train.loc[df_train.landtaxvaluedollarcnt.isnull(),'landtaxvaluedollarcnt'] = df_train.loc[df_train.landtaxvaluedollarcnt.isnull(),'taxamount']



#structure tax value dollar - fill this in using its most correlated variable

x =  df_train.corr()

print(x.structuretaxvaluedollarcnt.sort_values(ascending = False))



#taxvaluedollarcnt is most correlated variable, let's see how they are related 

plt.figure(figsize=(12,12))

sns.jointplot(x=df_train.structuretaxvaluedollarcnt.values, y=df_train.taxvaluedollarcnt.values)

plt.ylabel('taxvaluedollarcnt', fontsize=12)

plt.xlabel('structuretaxvaluedollarcnt', fontsize=12)

plt.title("structuretaxvaluedollarcnt Vs taxvaluedollarcnt", fontsize=15)

plt.show()



#Lets look at the distribution of taxvaluedollar cnt where structuretaxvaluedollarcnt is missing just to make sure we are predicting missing values in the body of the taxvaluedollarcnt distribution

print(df_train.loc[df_train['structuretaxvaluedollarcnt'].isnull(),'taxvaluedollarcnt'].describe())

print(df_train['taxvaluedollarcnt'].describe())



#Slightly amend the k nearest neighbour function so it works on regression

def fillna_knn_reg( df, base, target, n_neighbors = 5 ):

    cols = base + [target]

    X_train = df[cols]

    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train[base].values.reshape(-1, 1))

    rescaledX = scaler.transform(X_train[base].values.reshape(-1, 1))



    X_train = rescaledX[df[target].notnull()]

    Y_train = df.loc[df[target].notnull(),target].values.reshape(-1, 1)



    knn = KNeighborsRegressor(n_neighbors, n_jobs = -1)    

    # fitting the model

    knn.fit(X_train, Y_train)

    # predict the response

    X_test = rescaledX[df[target].isnull()]

    pred = knn.predict(X_test)

    df.loc[df_train[target].isnull(),target] = pred

    return



#fill in structuretaxvaluedollarcnt using taxvaluedollarcnt as per the above

fillna_knn_reg(df = df_train, base = ['taxvaluedollarcnt'], target = 'structuretaxvaluedollarcnt')



#Do the same thing for tax amount, as taxvaluedollarcnt is its most correlated variable

fillna_knn_reg(df = df_train, base = ['taxvaluedollarcnt'], target = 'taxamount')

print(df_train.isnull().sum())
#Let's see whats left

df_train.isnull().sum()
