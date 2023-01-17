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
#df1=pd.read_csv('../input/newdata/1-4-2019_newdataset.csv')

df1=pd.read_csv('../input/finaldata/new dataset.csv')



df1.head()
# feature names as a list

col = df1.columns       # .columns gives columns names in data 

print(col)

print(df1.shape)
"""# y includes our labels and x includes our features

y = df1.is_churn                # 0 or 1 

list = ['is_churn']

x = df1.drop(list,axis = 1 )

x.head()"""
#ax = sns.countplot(y,label="Count")       # 1 = 44231, 0 =217451

#print(y.value_counts())
drop_list1 = ['msno']

x_1 = df1.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

x_1.head()
#correlation map

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
missing_df = x_1.isnull().sum(axis=0).reset_index()

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
#ax = sns.countplot(missing_df,label="Count")

#print(missing_df.value_counts())

#df1.apply(lambda x: x.count(), axis=1)

missing_df.apply(lambda x: x.count(), axis=1)



missing_df
churn=(df1['is_churn'],df1['num_25'],df1['num_50'])
from fancyimpute import KNN 

X_filled_knn = KNN(k=3).fit_transform(churn)    

x_1.info()
"""from fancyimpute import KNN 

#We use the train dataframe from Titanic dataset

#fancy impute removes column names.

train_cols = x_1

# Use 5 nearest rows which have a feature to fill in each row's

# missing features

x_1 = pd.DataFrame(KNN(k=5).complete(x_1))

x_1.columns = train_cols"""
print(x_1.is_cancel.value_counts())

print(x_1.is_cancel.isnull().sum())

missingvalues_prop = (x_1.isnull().sum()/len(x_1)).reset_index()

missingvalues_prop.columns = ['field','proportion']

missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)

print(missingvalues_prop)

#missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()

#dropcols = dropcols + missingvaluescols

#x_1 = x_1.drop(dropcols, axis=1)
df1 = df1.reset_index()

def fillna_knn( df1, base, target, fraction = 1, threshold = 10, n_neighbors = 5 ):

    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 

    whole = [ target ] + base

    

    miss = df1[target].isnull()

    notmiss = ~miss 

    nummiss = miss.sum()

    

    enc = OneHotEncoder()

    X_target = df1.loc[ notmiss, whole ].sample( frac = fraction )

    

    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )

    

    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()

    X = X_target[ base  ]

    

    print( 'fitting' )

    n_neighbors = n_neighbors

    clf = neighbors.KNeighborsClassifier( n_neighbors, weights = 'uniform' )

    clf.fit( X, Y )

    

    print( 'the shape of active features: ' ,enc.active_features_.shape )

    

    print( 'predicting' )

    Z = clf.predict(df1.loc[miss, base])

    

    numunperdicted = Z[:,0].sum()

    if numunperdicted / nummiss *100 < threshold :

        print( 'writing result to df1' )    

        df1.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )

        print( 'num of unperdictable data: ', numunperdicted )

        return enc

    else:

        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )



#function to deal with variables that are actually string/categories

def zoningcode2int( df1, target ):

    storenull = df1[ target ].isnull()

    enc = LabelEncoder( )

    df1[ target ] = df1[ target ].astype( str )



    print('fit and transform')

    df1[ target ]= enc.fit_transform( df1[ target ].values )

    print( 'num of categories: ', enc.classes_.shape  )

    df1.loc[ storenull, target ] = np.nan

    print('recover the nan value')

    return enc
#buildingqualitytypeid - assume it is the similar to the nearest property. Probably makes senses if its a property in a block of flats, i.e if block was built all at the same time and therefore all flats will have similar quality 

#Use the same logic for propertycountylandusecode (assume it is same as nearest property i.e two properties right next to each other are likely to have the same code) & propertyzoningdesc. 

#These assumptions are only reasonable if you actually have nearby properties to the one with the missing value



fillna_knn( df1 = x_1,base = [  'is_churn ' , 'msnoid '] ,target = 'is_cancel', fraction = 0.15, n_neighbors = 1 )





#zoningcode2int( df1 = x_1,target = 'date' )

fillna_knn( df1 = x_1,

                  base = ['is_churn ', 'msnoid ' ] ,

                  target = 'date', fraction = 0.15, n_neighbors = 1 )



#zoningcode2int( df1 = x_1,target = 'num_25' )



fillna_knn( df1 = x_1,

                  base = [ 'is_churn ', 'msnoid ' ] ,

                  target = 'num_25', fraction = 0.15, n_neighbors = 1 )



#zoningcode2int( df1 = x_1,target = 'num_50' )



fillna_knn( df1 = x_1,

                  base = [ 'is_churn ', 'msnoid ' ] ,

                  target = 'num_50', fraction = 0.15, n_neighbors = 1 )



#zoningcode2int( df1 = x_1,target = 'num_75' )



fillna_knn( df1 = x_1,

                  base = [ 'is_churn ', 'msnoid ' ] ,

                  target = 'num_75', fraction = 0.15, n_neighbors = 1 )



#regionidcity, regionidneighborhood & regionidzip - assume it is the same as the nereast property. 

#As mentioned above, this is ok if there's a property very nearby to the one with missing values (I leave it up to the reader to check if this is the case!)

fillna_knn( df1 = x_1,

                  base = ['is_churn ', 'msnoid ' ] ,

                  target = 'payment_plan_days', fraction = 0.15, n_neighbors = 1 )



fillna_knn(df1 = x_1,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'plan_list_price', fraction = 0.15, n_neighbors = 1 )



fillna_knn( df1 = x_1,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'actual_amount_paid', fraction = 0.15, n_neighbors = 1 )



#unitcnt - the number of structures the unit is built into. Assume it is the same as the nearest properties. If the property with missing values is in a block of flats or in a terrace street then this is probably ok - but again I leave it up to the reader to check if this is the case!

fillna_knn( df1 = x_1,

                  base = [ 'is_churn ', 'msnoid '] ,

                  target = 'is_auto_renew', fraction = 0.15, n_neighbors = 1 )



#yearbuilt - assume it is the same as the nearest property. This assumes properties all near to each other were built around the same time

fillna_knn( df1 = x_1,

                  base = ['is_churn ', 'msnoid '] ,

                  target = 'transaction_date', fraction = 0.15, n_neighbors = 1 )



#lot size square feet - not sure what to do about this one. Lets use nearest neighbours. Assume it has same lot size as property closest to it

fillna_knn( df1 = x_1,

                  base = [ 'is_churn ', 'msnoid '] ,

                  target = 'membership_expire_date', fraction = 0.15, n_neighbors = 1 )