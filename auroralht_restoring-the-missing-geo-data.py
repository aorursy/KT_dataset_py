import numpy as np

import pandas as pd

import gc

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(font_scale=1.5)

#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



#from sklearn.gaussian_process import GaussianProcessRegressor

#from sklearn.gaussian_process.kernels import Matern, WhiteKernel

#from scipy.optimize import minimize

#from scipy.stats import norm

prop = pd.read_csv('../input/properties_2016.csv')
prop.columns.tolist() 
geocolumns = [  'latitude', 'longitude'

                            ,'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc'

                            ,'regionidcity','regionidcounty', 'regionidneighborhood', 'regionidzip'

                            ,'censustractandblock', 'rawcensustractandblock']
geoprop = prop[geocolumns]

del prop; gc.collect()
missingcount = geoprop.isnull().sum(axis=0)

plt.figure( figsize = (16,8) )

plot= sns.barplot( x = geocolumns,y = missingcount.values )

plt.setp( plot.get_xticklabels(), rotation = 45 )

missingcount
corr = geoprop.isnull().corr()

sns.heatmap( corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values ) 
# let's clean the row without the latitude and longitude value

geoprop.dropna( axis = 0, subset = [ 'latitude', 'longitude' ], inplace = True )
geoprop.loc[:,'latitude'] = geoprop.loc[:,'latitude']/1e6

geoprop.loc[:,'longitude'] = geoprop.loc[:,'longitude']/1e6



maxlat = (geoprop['latitude']).max()

maxlon = (geoprop['longitude']).max()

minlat = (geoprop['latitude']).min()

minlon = (geoprop['longitude']).min()

print('maxlat {} minlat {} maxlon {} minlon {}'.format(maxlat, minlat, maxlon, minlon))



CAparms = { 'llcrnrlat' : minlat,

                     'urcrnrlat' : maxlat+0.2,

                     'llcrnrlon' : maxlon-2.5,

                     'urcrnrlon' :minlon+2.5 }
from mpl_toolkits.basemap import Basemap, cm
def create_basemap( llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60, figsize=(16,9) ):

    fig=plt.figure( figsize = figsize )

    Bm = Basemap( projection='merc', 

                llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,

                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,

                lat_ts=20,resolution='i' )

    # draw coastlines, state and country boundaries, edge of map.

    Bm.drawcoastlines(); Bm.drawstates(); Bm.drawcountries() 

    return Bm, fig    
Bm, fig = create_basemap()

x,y = Bm( geoprop['longitude'].values, geoprop['latitude'].values)                           

Bm.scatter( x, y, marker = 'D',color = 'm', s = 1 )

plt.show()
citydict = {}

citydict['Los Angle'] = [ 34.088537, -118.249923 ]

citydict['Anaheim'] = [ 33.838199,  -117.924770 ]

citydict['Irvine'] = [ 33.683549,  -117.793723 ]

citydict['Long Beach'] = [ 33.778341,  -118.285261]

citydict['Oxnard'] = [ 34.171196, -119.165045 ]

citydict['Ventura'] = [ 34.283106, -119.225597 ]

citydict['Palmdale'] = [34.612009, -118.127173]

citydict['Lancaster'] = [34.719710, -118.135903]

citydict['Hesperia'] = [34.420196, -117.289121]

citydict['Riverside'] = [33.972528, -117.405517]

def plot_maincities( Bm, citydict ):

    for key, values in citydict.items():

        x , y = Bm( values[1], values[0] )

        Bm.plot( x, y, 'bo', markersize = 5)

        plt.text( x+3000, y+3000, key )    
def view_missing( df, target,see_known=True ,ignorefirst = False ):





    Bm, fig = create_basemap( **CAparms )



    # plot the known data

    if see_known:

        notmiss_df = df.loc[ df[target].notnull() ]

        groupby = notmiss_df.groupby(target)

        groups = [ groupby.get_group(g) for g in groupby.groups ]

        groups = groups[1:] if ignorefirst else groups 

        print( 'num groups:  ', len( groups ) )

        for group in groups:

            x,y = Bm( group['longitude'].values, group['latitude'].values )  

            Bm.scatter( x, y,  marker = 'D', s = 1 )



    # plot the missing data

    missing_target = df[target].isnull()

    if missing_target.any():

        print( '{} missing value at column: {}'.format( missing_target.sum(), target ) )

        missing = df.loc[ missing_target, ['latitude','longitude'] ]

        x,y = Bm( missing['longitude'].values, missing['latitude'].values )  

        Bm.scatter( x, y,  marker='D',s = 3, color = 'yellow', alpha = 0.1 )

    else:

        print('zero missing value at column: ', target )

        

    Bm.drawcounties( color='b', linewidth=0.3 )



    plot_maincities( Bm, citydict )



    plt.show()
Bm, fig = create_basemap( **CAparms )



x , y = Bm( geoprop['longitude'].values, geoprop['latitude'].values ) 



Bm.scatter(x,y , marker='D',color='m',s=1 )

Bm.drawcounties(color='b')



plot_maincities( Bm, citydict )



plt.show()
misscity = geoprop['regionidcity'].isnull()



Bm, fig = create_basemap( **CAparms )



x,y = Bm( geoprop.loc[ misscity, 'longitude' ].values, geoprop.loc[ misscity,'latitude' ].values )                        



#plot the property

Bm.scatter(x, y , marker = 'D',color = 'm',s = 1 )

Bm.drawcounties( color = 'b' )



#plot the location of the main cities

plot_maincities( Bm, citydict )



plt.show()

view_missing( geoprop, 'regionidcity', ignorefirst = False )
from sklearn import neighbors

from sklearn.preprocessing import OneHotEncoder



def fillna_knn( df, base, target, fraction = 1, threshold = 10 ):

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

    n_neighbors = 10

    clf = neighbors.KNeighborsClassifier( n_neighbors, weights = 'uniform' )

    clf.fit( X, Y )

    

    print( 'the shape of active features: ' ,enc.active_features_.shape )

    

    print( 'perdicting' )

    Z = clf.predict(geoprop.loc[ miss, base  ])

    

    numunperdicted = Z[:,0].sum()

    if numunperdicted / nummiss *100 < threshold :

        print( 'writing result to df' )    

        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )

        print( 'num of unperdictable data: ', numunperdicted )

        return enc

    else:

        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )

        
fillna_knn( df = geoprop,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'regionidcity', fraction = 0.15 )
groupscity = geoprop.groupby('regionidcity')

groups =[ groupscity.get_group(x) for x in groupscity.groups ]

print('num groups : ',len(groups))
groups[0].regionidcity.values[0]

del groups, groupscity ; gc.collect()
view_missing( geoprop, 'regionidcity', ignorefirst = True )
missingcount = geoprop.isnull().sum(axis=0)

missingcount[missingcount>0]
fillna_knn( df = geoprop,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'regionidzip', fraction = 0.1 )
missingcount = geoprop.isnull().sum( axis = 0 )

missingcount[ missingcount>0 ]
print('The number of categories : ' ,geoprop.regionidneighborhood.nunique())

print(geoprop.regionidneighborhood.value_counts().head() )
view_missing( geoprop, 'regionidneighborhood', ignorefirst = False )
groupby_county = geoprop.groupby('regionidcounty')

groups = [ groupby_county.get_group(g) for g in groupby_county.groups ]

print('num groups : ', len(groups))

for g in groups:

    print( 'num unique ', g.propertycountylandusecode.nunique() )

    print( g.propertycountylandusecode.unique() )

    print('----------------------------------------------------------------------')
view_missing( geoprop, 'propertycountylandusecode', ignorefirst = False )
from sklearn.preprocessing import LabelEncoder



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

zoningcode2int( df = geoprop,

                            target = 'propertycountylandusecode' )
geoprop.propertycountylandusecode.nunique()
enc=fillna_knn( df = geoprop,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'propertycountylandusecode', fraction = 0.1 )
geoprop.propertycountylandusecode.value_counts().tail(100)
vc = geoprop.propertyzoningdesc.value_counts()

fig = plt.figure(figsize=((16,9)))

sns.barplot( x = vc.index[:10], y = vc.values[:10] )
view_missing(geoprop,'propertyzoningdesc')
view_missing(geoprop,'censustractandblock',see_known=False)