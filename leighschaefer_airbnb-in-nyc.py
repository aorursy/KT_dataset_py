import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O



# plotting

from matplotlib import pyplot as plt

import seaborn as sns



# interactive mapping

import geopandas as gpd

import folium 



# load in datasets

# airbnb data

listings=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

# to create interactive maps

neighborhoods=gpd.read_file('/kaggle/input/new-york-shapefile-16/cb_2016_36_tract_500k.shp')
listings.head()
neighborhoods.head()
# drop any possible duplicates

listings.drop_duplicates(inplace=True)
# use the latitude and longitude info in the airbnb dataframe to copy it to a GeoDataFrame, 

# then merge it with the neighborhoods geodataframe

geolistings = gpd.GeoDataFrame(listings, geometry=gpd.points_from_xy(listings.longitude, listings.latitude))

geolistings.crs = neighborhoods.crs

listings=gpd.sjoin(neighborhoods,geolistings)
listings.describe()
plt.yscale('log')

priceHist = listings.price.hist(bins=20)

plt.xlabel('Price (USD)')

plt.ylabel('Number of NYC listings')
plt.yscale('log')

priceHist = listings.price.hist(bins=20,range=(500,1500))

plt.xlabel('Price (USD)')

plt.ylabel('Number of NYC listings')
plt.yscale('log')

priceHist = listings.price.hist(bins=20,range=(0,700))

plt.xlabel('Price (USD)')

plt.ylabel('Number of NYC listings')
listings = listings.loc[listings.price<700]

listings=listings.reset_index()
plt.yscale('log')

priceHist = listings.minimum_nights.hist(bins=20)

plt.xlabel('Minimum nights')

plt.ylabel('Number of NYC listings')
plt.yscale('log')

priceHist = listings.minimum_nights.hist(bins=20,range=(0,90))

plt.xlabel('Minimum nights')

plt.ylabel('Number of NYC listings')
listings = listings.loc[listings.minimum_nights<30]

listings=listings.reset_index()
listings.info()
listings.groupby([listings.neighbourhood_group]).id.count().plot(kind='bar')

plt.ylabel('Number of listings')
# from https://www.kaggle.com/alexisbcook/interactive-maps

from IPython.display import IFrame



def embed_map(m, file_name):

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
heatmapnyc = folium.Map(location=[40.7, -74],tiles='cartodbpositron', zoom_start=10) 

from folium.plugins import HeatMap

HeatMap(data=listings[['latitude', 'longitude']], radius=8).add_to(heatmapnyc)

embed_map(heatmapnyc, "heatmapnyc.html")
listings.groupby([listings.neighbourhood_group,listings.room_type]).id.count().unstack().plot(kind='bar',stacked=True)

plt.ylabel('Number of listings')
bk = listings.loc[(listings.neighbourhood_group=='Brooklyn')].price.hist(bins=20,label='Brooklyn',histtype='step',linewidth=2)

mh = listings.loc[(listings.neighbourhood_group=='Manhattan')].price.hist(bins=20,label='Manhattan',histtype='step',linewidth=2)

qu = listings.loc[(listings.neighbourhood_group=='Queens')].price.hist(bins=20,label='Queens',histtype='step',linewidth=2)

si = listings.loc[(listings.neighbourhood_group=='Staten Island')].price.hist(bins=20,label='Staten Island',histtype='step',linewidth=2)

bx = listings.loc[(listings.neighbourhood_group=='Bronx')].price.hist(bins=20,label='Bronx',histtype='step',linewidth=2)

plt.legend()

plt.xlabel('Price (USD)')

plt.ylabel('Number of listings')
bk = listings.loc[(listings.neighbourhood_group=='Brooklyn')].price.hist(bins=20,label='Brooklyn',density=True,histtype='step',linewidth=2)

mh = listings.loc[(listings.neighbourhood_group=='Manhattan')].price.hist(bins=20,label='Manhattan',density=True,histtype='step',linewidth=2)

qu = listings.loc[(listings.neighbourhood_group=='Queens')].price.hist(bins=20,label='Queens',density=True,histtype='step',linewidth=2)

si = listings.loc[(listings.neighbourhood_group=='Staten Island')].price.hist(bins=20,label='Staten Island',density=True,histtype='step',linewidth=2)

bx = listings.loc[(listings.neighbourhood_group=='Bronx')].price.hist(bins=20,label='Bronx',density=True,histtype='step',linewidth=2)

plt.legend()

plt.xlabel('Price (USD)')

plt.ylabel('Number of listings')
sns.catplot(x="neighbourhood_group", y="price",

            kind="box", data=listings)
sns.catplot(x="neighbourhood_group", y="price",

            kind="violin", data=listings,cut=0)
sns.catplot(x="neighbourhood_group", y="price",hue='room_type',

            kind="violin", data=listings,cut=0,figsize=(50,50))
mapnyc = folium.Map(location=[40.7, -74],tiles='cartodbpositron', zoom_start=10)

meanPrice=listings.groupby('NAME').price.mean()



folium.Choropleth(geo_data=listings.set_index("NAME").__geo_interface__, 

           data=meanPrice,

           key_on="feature.id", 

           fill_color='YlGnBu', 

           line_color='none',

           legend_name='Mean Airbnb price (USD)'

          ).add_to(mapnyc)



# Display the map (commented for now)

# embed_map(mapnyc, 'mapnyc.html')
listings.dtypes
# first, remove the listings with nan reviews_per_month (nonzero as seen from `listings.info` above)

listings.dropna(subset=['reviews_per_month'],inplace=True)

listings.reset_index(drop=True,inplace=True)
listings['reviewsPerMonthScaled']=listings.reviews_per_month*listings.minimum_nights
sns.relplot(x="price",

            y="reviewsPerMonthScaled",

            data=listings,

            hue=listings.neighbourhood_group,

            kind='line',

            ci=None,

         )
# make a coarsePrice column which bins in multiples of 10.

listings['coarsePrice']=(listings.price/10).round()*10 # divide then multiply because round function is designed for decimals
listings[['price','coarsePrice']].head(10)
sns.relplot(x="coarsePrice",

            y="reviewsPerMonthScaled",

            data=listings,

            hue=listings.neighbourhood_group,

            kind='line',

            ci=None,

         )
# since I'm quickly checking 3 distributions, make each one a bit smaller with `height`

sns.relplot(x="coarsePrice",

            y="reviewsPerMonthScaled",

            data=listings.loc[(listings.room_type=="Entire home/apt")],

            hue=listings.neighbourhood_group,

            kind='line',

            ci=None,

            height=3

         )

sns.relplot(x="coarsePrice",

            y="reviewsPerMonthScaled",

            data=listings.loc[(listings.room_type=="Private room")],

            hue=listings.neighbourhood_group,

            kind='line',

            ci=None,

            height=3

         )

sns.relplot(x="coarsePrice",

            y="reviewsPerMonthScaled",

            data=listings.loc[(listings.room_type=="Shared room")],

            hue=listings.neighbourhood_group,

            kind='line',

            ci=None,

            height=3

         )
listings['estMonthlyIncome']=listings.price*listings.reviewsPerMonthScaled



sns.relplot(x="coarsePrice",

            y='estMonthlyIncome',

            data=listings,

            hue=listings.neighbourhood_group,

            kind='line',

            ci=None

         )
sns.relplot(x="coarsePrice",

            y='estMonthlyIncome',

            data=listings[(listings.price<400) & (listings.neighbourhood_group=='Manhattan') & ((listings.room_type=='Entire home/apt') | (listings.room_type=='Private room'))],

            hue=listings.neighbourhood_group,

            style=listings.room_type,

            kind='line',

            ci=None,

            height=3

         )

sns.relplot(x="coarsePrice",

            y='estMonthlyIncome',

            data=listings[(listings.price<400) & (listings.neighbourhood_group=='Brooklyn') & ((listings.room_type=='Entire home/apt') | (listings.room_type=='Private room'))],

            hue=listings.neighbourhood_group,

            style=listings.room_type,

            kind='line',

            ci=None,

            height=3

         )

# Import necessary scikit-learn packages

from sklearn import linear_model

from sklearn.metrics import r2_score



# Split the data into training/testing sets

from sklearn.model_selection import train_test_split



# to fit a polynomial

from sklearn.preprocessing import PolynomialFeatures



# First create a function since we'll be repeating these few lines several times

def fitPredictPlot__incomeVsPrice(data, model='linear'):



    X,Y=data[['price']],data.estMonthlyIncome

    

    if model=='poly': # if fitting a 2nd order polynomial

    # https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions

        poly = PolynomialFeatures(degree=2)

        X=poly.fit_transform(data[['price']])

    

    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)



    # Create linear regression object

    fitter=linear_model.LinearRegression()

    

    # Train the model using the training sets

    fitter.fit(X_train,y_train)

    y_pred = fitter.predict(X_test)

    

    # Variance score: 1 is perfect prediction

    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    

    # Plot the output

    sns.relplot(x="coarsePrice",

                y='estMonthlyIncome',

                data=data,

                hue=listings.neighbourhood_group,

                style=listings.room_type,

                kind='line',

                ci=None,

                height=5

             )

    if model=='linear':

        xaxis=X_test

    elif model=='poly':

        xaxis=X_test[:,1]

        

    plt.scatter(xaxis, y_pred, color='blue', linewidth=3)



    # return the regression object

    return fitter

fitPredictPlot__incomeVsPrice(listings.loc[(listings.room_type=='Entire home/apt') & (listings.price<400)])

# now try a 2nd order polynomial

# first do it just for manhattan private rooms to see how it goes



fitPredictPlot__incomeVsPrice(listings.loc[(listings.room_type=='Private room') & (listings.neighbourhood_group=='Manhattan')],

                              model='poly')

import itertools

for borough,roomType in itertools.product(['Manhattan','Queens','Bronx','Staten Island','Brooklyn'],['Entire home/apt','Private room','Shared room']):

    fitter=fitPredictPlot__incomeVsPrice(listings.loc[(listings.room_type==roomType) & (listings.neighbourhood_group==borough)],

                                         model='poly')



    # print ('parameters: ',regr.coef_)

    # location of max/min for f(x)=ax^2+bx+c is -b/(2a)

    priceMax=-fitter.coef_[1]/(2*fitter.coef_[2])

    print ('The optimal price for this %s listing in %s is $%d' %(roomType,borough,priceMax))
# First drop categorial data and features that I've added

listings.dtypes
# also drop borough information, this can be picked up via latitude and longitude

listings_numer=listings.drop(columns=['index','STATEFP','COUNTYFP','TRACTCE','AFFGEOID','GEOID','NAME','LSAD','ALAND','AWATER','geometry','index_right','id','name','host_id','host_name','neighbourhood_group','neighbourhood','estMonthlyIncome','coarsePrice','reviewsPerMonthScaled','last_review'])
for roomType in ['Entire home/apt','Private room','Shared room']:

    thisListings = listings_numer.loc[(listings_numer.room_type==roomType)].drop(columns=['room_type'])

    all_X = thisListings.drop(columns=['price'])

    all_y = thisListings['price']

    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2)



    fitter=linear_model.LinearRegression()

    fitter.fit(X_train,y_train)

    y_pred = fitter.predict(X_test)

    

    # Variance score: 1 is perfect prediction

    print('Variance score: %.2f' % r2_score(y_test, y_pred))



    d = pd.DataFrame({'actual %s' %roomType: y_test, 'predicted %s' %roomType: y_pred})

    sns.relplot(x="predicted %s" %roomType,

                y="actual %s" %roomType,

                data=d,

                color="red")

    

    plt.plot(y_pred, y_pred, color='blue', linewidth=3)

corr = listings.corr()

sns.heatmap(corr)