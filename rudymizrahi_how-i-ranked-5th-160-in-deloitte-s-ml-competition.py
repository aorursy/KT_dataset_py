#load packages
import sys
print("Python version: {}". format(sys.version))
import pandas as pd
print("pandas version: {}". format(pd.__version__))
import numpy as np
print("NumPy version: {}". format(np.__version__))
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import sklearn
print("scikit-learn version: {}". format(sklearn.__version__))
from sklearn.metrics import make_scorer, mean_squared_error
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
import os
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
%matplotlib inline


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

#checking the data size
print("\nThe train data size is : {} ".format(data_train.shape))
print("The test data size is : {} ".format(data_test.shape))


print(data_train.info())
data_train.sample(10)
import matplotlib.pyplot as plt
sns.distplot(data_train['log_price'] , fit=norm);
# Get the fitted parameters
(mu, sigma) = norm.fit(data_train['log_price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('LogPrice distribution')
#the QQ-plot
fig = plt.figure()
res = stats.probplot(data_train['log_price'], plot=plt)
plt.show()
corr_mx = data_train.corr()
corr_mx["log_price"].sort_values(ascending=False)
g = sns.PairGrid(data_train, hue="city", vars=["log_price", "accommodates", "bathrooms","bedrooms", "beds" ])
g = g.map(plt.scatter)
g = g.add_legend()
g = sns.PairGrid(data_train, hue="city", vars=["log_price", "number_of_reviews", "review_scores_rating", "cleaning_fee"])
g = g.map(plt.scatter)
g = g.add_legend()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

data_train['city'].value_counts().plot(kind='bar', ax=ax1)
plt.title('Number of Listings per City - Train')
plt.xlabel('City')
plt.ylabel('Count')


data_test['city'].value_counts().plot(kind='bar', ax=ax2)
plt.title('Number of Listings per City - Test')
plt.xlabel('City')
plt.ylabel('Count')
plt.ylim(0,33000 )
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(data_train.groupby([
        'city', 'room_type']).log_price.mean().unstack(),annot=True, fmt=".0f")

plt.figure(figsize=(10,10))
sns.heatmap(data_train.groupby(['property_type', 'bedrooms']).log_price.mean().unstack(), annot=True, fmt=".0f")

plt.figure(figsize=(10,10))
sns.heatmap(data_train.groupby(['beds', 'bedrooms']).log_price.mean().unstack(), annot=True, fmt=".0f")
g = sns.lmplot(x="log_price", y="number_of_reviews", col="city", col_wrap=3, hue='cancellation_policy', data=data_train)
g.set(ylim=(0, 700))

import folium

MAPdata = data_train.copy()
MAPdata = MAPdata.sample(1000) #nlargest(100, 'log_price')
MAPdataloc = MAPdata[['latitude','longitude']]
MAPdatalist = MAPdataloc.values.tolist()

MAPdatat = data_test.copy()
MAPdatat = MAPdatat.sample(1000)
MAPdataloct = MAPdatat[['latitude','longitude']]
MAPdatalistt = MAPdataloct.values.tolist()



mapall = folium.Map(location=[42.3601, -71.0589], zoom_start=12)

for point in range(0, len(MAPdatalist)):
    folium.Marker(MAPdatalist[point], icon=folium.Icon(color='red', icon='cloud')).add_to(mapall)
for point in range(0, len(MAPdatalistt)):
    folium.Marker(MAPdatalistt[point], icon=folium.Icon(color='blue', icon='cloud')).add_to(mapall)    
    
mapall
NEIdata = data_train.copy()
plt.figure(figsize = (12, 100))
sns.boxplot(y = 'neighbourhood', x = 'log_price',  data = NEIdata, orient="h")
xt = plt.xticks()
def lat_citycenter(row):
    if (row['city']=='Boston'):
        return 42.3601
    elif (row['city']=='NYC'):
        return 40.7128
    elif (row['city']=='LA'):
        return 34.0522
    elif (row['city']=='SF'):
        return 37.7749
    elif (row['city']=='Chicago'):
        return 41.8781
    elif (row['city']=='DC'):
        return 38.9072

   
def long_citycenter(row):
    if (row['city']=='Boston'):
        return -71.0589
    elif (row['city']=='NYC'):
        return -74.0060
    elif (row['city']=='LA'):
        return -118.2437    
    elif (row['city']=='SF'):
        return -122.4194     
    elif (row['city']=='Chicago'):
        return -87.6298    
    elif (row['city']=='DC'):
        return -77.0369 

###########
def lat_attr1(row):
    if (row['city']=='Boston'):
        return 42.3602#FanueilHall
    elif (row['city']=='NYC'):
        return 40.7589
    elif (row['city']=='LA'):
        return 34.0928
    elif (row['city']=='SF'):
        return 37.8199
    elif (row['city']=='Chicago'):
        return 41.8918
    elif (row['city']=='DC'):
        return 38.8973
   
def long_attr1(row):
    if (row['city']=='Boston'):
        return -71.0548
    elif (row['city']=='NYC'):
        return -73.9851
    elif (row['city']=='LA'):
        return -118.329    
    elif (row['city']=='SF'):
        return -122.478     
    elif (row['city']=='Chicago'):
        return -87.6052  
    elif (row['city']=='DC'):
        return -77.0063

def lat_attr2(row):
    if (row['city']=='Boston'):
        return 42.3467#FenwayPark
    elif (row['city']=='NYC'):
        return 40.7484
    elif (row['city']=='LA'):
        return 34.0195
    elif (row['city']=='SF'):
        return 37.8087
    elif (row['city']=='Chicago'):
        return 41.8827
    elif (row['city']=='DC'):
        return 38.886
   
def long_attr2(row):
    if (row['city']=='Boston'):
        return -71.0972
    elif (row['city']=='NYC'):
        return -73.9857
    elif (row['city']=='LA'):
        return -118.491    
    elif (row['city']=='SF'):
        return -122.41     
    elif (row['city']=='Chicago'):
        return -87.6233 
    elif (row['city']=='DC'):
        return -77.0213

def lat_attr3(row):
    if (row['city']=='Boston'):
        return 42.377#Harvard
    elif (row['city']=='NYC'):
        return 40.6892
    elif (row['city']=='LA'):
        return 33.8121
    elif (row['city']=='SF'):
        return 37.788
    elif (row['city']=='Chicago'):
        return 41.8789
    elif (row['city']=='DC'):
        return 38.9097
   
def long_attr3(row):
    if (row['city']=='Boston'):
        return -71.1167
    elif (row['city']=='NYC'):
        return -74.0445
    elif (row['city']=='LA'):
        return -117.919    
    elif (row['city']=='SF'):
        return -122.408     
    elif (row['city']=='Chicago'):
        return -87.6359 
    elif (row['city']=='DC'):
        return -77.0654

##########
def lat_attr4(row):
    if (row['city']=='Boston'):
        return 42.3601#MIT
    elif (row['city']=='NYC'):
        return 40.7829
    elif (row['city']=='LA'):
        return 34.1362
    elif (row['city']=='SF'):
        return 37.7694
    elif (row['city']=='Chicago'):
        return 41.8676
    elif (row['city']=='DC'):
        return 38.8899
 
def long_attr4(row):
    if (row['city']=='Boston'):
        return -71.0942
    elif (row['city']=='NYC'):
        return -73.9654
    elif (row['city']=='LA'):
        return -118.3514    
    elif (row['city']=='SF'):
        return -122.4862    
    elif (row['city']=='Chicago'):
        return -87.6140 
    elif (row['city']=='DC'):
        return -77.0091


##########
def lat_attr5(row):
    if (row['city']=='Boston'):
        return 42.3663#Old North Church
    elif (row['city']=='NYC'):
        return 40.7587#Rockefeller Center
    elif (row['city']=='LA'):
        return 34.0692#Rodeo Drive
    elif (row['city']=='SF'):
        return 37.7599#Mission District
    elif (row['city']=='Chicago'):
        return 41.9484#wrigley Field
    elif (row['city']=='DC'):
        return 39.9288#Columbia Heights
 
def long_attr5(row):
    if (row['city']=='Boston'):
        return -71.0544
    elif (row['city']=='NYC'):
        return -73.9787
    elif (row['city']=='LA'):
        return -118.4029    
    elif (row['city']=='SF'):
        return -122.4148    
    elif (row['city']=='Chicago'):
        return -87.6553
    elif (row['city']=='DC'):
        return -77.0305

##########
def lat_attr6(row):
    if (row['city']=='Boston'):
        return 42.3340#City Point
    elif (row['city']=='NYC'):
        return 40.7230#Lower Manhattan
    elif (row['city']=='LA'):
        return 34.0900#WEHO
    elif (row['city']=='SF'):
        return 37.7775#Alamo Square
    elif (row['city']=='Chicago'):
        return 41.9077#old town
    elif (row['city']=='DC'):
        return 39.9096#Logan Circle
 
def long_attr6(row):
    if (row['city']=='Boston'):
        return -71.0275
    elif (row['city']=='NYC'):
        return -74.0006
    elif (row['city']=='LA'):
        return -118.3617   
    elif (row['city']=='SF'):
        return -122.4333   
    elif (row['city']=='Chicago'):
        return -87.6374
    elif (row['city']=='DC'):
        return -77.0296

##########
'''def lat_attr7(row):
    if (row['city']=='Boston'):
        return 42.377#Harvard
    elif (row['city']=='NYC'):
        return 40.6892
    elif (row['city']=='LA'):
        return 33.8121
    elif (row['city']=='SF'):
        return 37.788
    elif (row['city']=='Chicago'):
        return 41.8789
    elif (row['city']=='DC'):
        return 38.9097
   
def long_attr7(row):
    if (row['city']=='Boston'):
        return -71.1167
    elif (row['city']=='NYC'):
        return -74.0445
    elif (row['city']=='LA'):
        return -117.919    
    elif (row['city']=='SF'):
        return -122.408     
    elif (row['city']=='Chicago'):
        return -87.6359 
    elif (row['city']=='DC'):
        return -77.0654'''
    
###########
        
    
data_train['lat_citycenter']=data_train.apply(lambda row: lat_citycenter(row), axis=1)
data_train['long_citycenter']=data_train.apply(lambda row: long_citycenter(row), axis=1)

data_train['lat_attr1']=data_train.apply(lambda row: lat_attr1(row), axis=1)
data_train['long_attr1']=data_train.apply(lambda row: long_attr1(row), axis=1)
data_train['lat_attr2']=data_train.apply(lambda row: lat_attr2(row), axis=1)
data_train['long_attr2']=data_train.apply(lambda row: long_attr2(row), axis=1)
data_train['lat_attr3']=data_train.apply(lambda row: lat_attr3(row), axis=1)
data_train['long_attr3']=data_train.apply(lambda row: long_attr3(row), axis=1)
data_train['lat_attr4']=data_train.apply(lambda row: lat_attr4(row), axis=1)
data_train['long_attr4']=data_train.apply(lambda row: long_attr4(row), axis=1)
data_train['lat_attr5']=data_train.apply(lambda row: lat_attr5(row), axis=1)
data_train['long_attr5']=data_train.apply(lambda row: long_attr5(row), axis=1)
data_train['lat_attr6']=data_train.apply(lambda row: lat_attr6(row), axis=1)
data_train['long_attr6']=data_train.apply(lambda row: long_attr6(row), axis=1)
#data_train['lat_attr7']=data_train.apply(lambda row: lat_attr7(row), axis=1)
#data_train['long_attr7']=data_train.apply(lambda row: long_attr7(row), axis=1)

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 3956 Radius of earth in miles.
    distance = c * r
    return distance
data_train['dist_to_citycenter'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_citycenter'], row['lat_citycenter']), axis=1)

data_train['dist_to_attr1'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr1'], row['lat_attr1']), axis=1)
data_train['dist_to_attr2'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr2'], row['lat_attr2']), axis=1)
data_train['dist_to_attr3'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr3'], row['lat_attr3']), axis=1)
data_train['dist_to_attr4'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr4'], row['lat_attr4']), axis=1)
data_train['dist_to_attr5'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr5'], row['lat_attr5']), axis=1)
data_train['dist_to_attr6'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr6'], row['lat_attr6']), axis=1)
#data_train['dist_to_attr7'] = data_train.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr7'], row['lat_attr7']), axis=1)

#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data_train)
def jan_occupancy(row):
    if (row['city']=='Boston'):
        return 0.425806452
    elif (row['city']=='NYC'):
        return 0.490322581
    elif (row['city']=='LA'):
        return 0.591290323
    elif (row['city']=='SF'):
        return 0.591612903
    elif (row['city']=='Chicago'):
        return 0.38
    elif (row['city']=='DC'):
        return 0.47233871
    
def feb_occupancy(row):
    if (row['city']=='Boston'):
        return 0.436785714
    elif (row['city']=='NYC'):
        return 0.5125
    elif (row['city']=='LA'):
        return 0.623214286
    elif (row['city']=='SF'):
        return 0.655
    elif (row['city']=='Chicago'):
        return 0.373928571
    elif (row['city']=='DC'):
        return 0.514375

def mar_occupancy(row):
    if (row['city']=='Boston'):
        return 0.513225806
    elif (row['city']=='NYC'):
        return 0.606129032
    elif (row['city']=='LA'):
        return 0.642580645
    elif (row['city']=='SF'):
        return 0.690967742
    elif (row['city']=='Chicago'):
        return 0.471935484
    elif (row['city']=='DC'):
        return 0.654919355

def apr_occupancy(row):
    if (row['city']=='Boston'):
        return 0.661333333
    elif (row['city']=='NYC'):
        return 0.737333333
    elif (row['city']=='LA'):
        return 0.635333333
    elif (row['city']=='SF'):
        return 0.724333333
    elif (row['city']=='Chicago'):
        return 0.579333333
    elif (row['city']=='DC'):
        return 0.6975

def may_occupancy(row):
    if (row['city']=='Boston'):
        return 0.692258065
    elif (row['city']=='NYC'):
        return 0.785483871
    elif (row['city']=='LA'):
        return 0.618387097
    elif (row['city']=='SF'):
        return 0.789677419
    elif (row['city']=='Chicago'):
        return 0.714516129
    elif (row['city']=='DC'):
        return 0.716612903

def jun_occupancy(row):
    if (row['city']=='Boston'):
        return 0.688333333
    elif (row['city']=='NYC'):
        return 0.747
    elif (row['city']=='LA'):
        return 0.684666667
    elif (row['city']=='SF'):
        return 0.841666667
    elif (row['city']=='Chicago'):
        return 0.708333333
    elif (row['city']=='DC'):
        return 0.736

def jul_occupancy(row):
    if (row['city']=='Boston'):
        return 0.738064516
    elif (row['city']=='NYC'):
        return 0.675806452
    elif (row['city']=='LA'):
        return 0.715483871
    elif (row['city']=='SF'):
        return 0.85483871
    elif (row['city']=='Chicago'):
        return 0.733548387
    elif (row['city']=='DC'):
        return 0.736370968

def aug_occupancy(row):
    if (row['city']=='Boston'):
        return 0.739677419
    elif (row['city']=='NYC'):
        return 0.700322581
    elif (row['city']=='LA'):
        return 0.72516129
    elif (row['city']=='SF'):
        return 0.884516129
    elif (row['city']=='Chicago'):
        return 0.683225806
    elif (row['city']=='DC'):
        return 0.580887097

def sep_occupancy(row):
    if (row['city']=='Boston'):
        return 0.728666667
    elif (row['city']=='NYC'):
        return 0.774
    elif (row['city']=='LA'):
        return 0.574333333
    elif (row['city']=='SF'):
        return 0.843
    elif (row['city']=='Chicago'):
        return 0.702
    elif (row['city']=='DC'):
        return 0.5295
    
def oct_occupancy(row):
    if (row['city']=='Boston'):
        return 0.691290323
    elif (row['city']=='NYC'):
        return 0.758064516
    elif (row['city']=='LA'):
        return 0.579677419
    elif (row['city']=='SF'):
        return 0.797096774
    elif (row['city']=='Chicago'):
        return 0.677419355
    elif (row['city']=='DC'):
        return 0.505322581

def nov_occupancy(row):
    if (row['city']=='Boston'):
        return 0.592333333
    elif (row['city']=='NYC'):
        return 0.634666667
    elif (row['city']=='LA'):
        return 0.559666667
    elif (row['city']=='SF'):
        return 0.644
    elif (row['city']=='Chicago'):
        return 0.543
    elif (row['city']=='DC'):
        return 0.62875

def dec_occupancy(row):
    if (row['city']=='Boston'):
        return 0.44
    elif (row['city']=='NYC'):
        return 0.611290323
    elif (row['city']=='LA'):
        return 0.532580645
    elif (row['city']=='SF'):
        return 0.633870968
    elif (row['city']=='Chicago'):
        return 0.476451613
    elif (row['city']=='DC'):
        return 0.508870968

    
data_train['jan_occupancy']=data_train.apply(lambda row: jan_occupancy(row), axis=1)
data_train['feb_occupancy']=data_train.apply(lambda row: feb_occupancy(row), axis=1)
data_train['mar_occupancy']=data_train.apply(lambda row: mar_occupancy(row), axis=1)
data_train['apr_occupancy']=data_train.apply(lambda row: apr_occupancy(row), axis=1)
data_train['may_occupancy']=data_train.apply(lambda row: may_occupancy(row), axis=1)
data_train['jun_occupancy']=data_train.apply(lambda row: jun_occupancy(row), axis=1)
data_train['jul_occupancy']=data_train.apply(lambda row: jul_occupancy(row), axis=1)
data_train['aug_occupancy']=data_train.apply(lambda row: aug_occupancy(row), axis=1)
data_train['sep_occupancy']=data_train.apply(lambda row: sep_occupancy(row), axis=1)
data_train['oct_occupancy']=data_train.apply(lambda row: oct_occupancy(row), axis=1)
data_train['nov_occupancy']=data_train.apply(lambda row: nov_occupancy(row), axis=1)
data_train['dec_occupancy']=data_train.apply(lambda row: dec_occupancy(row), axis=1)

def walkscore(row):
    if (row['neighbourhood']=='Back Bay'):
        return 96
    elif (row['neighbourhood']=='Beacon Hill'): 
        return 98
    elif (row['neighbourhood']=='Roslindale'): 
        return 86
    elif (row['neighbourhood']=='East Boston'): 
        return 94
    elif (row['neighbourhood']=='West End'): 
        return 95
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Allston-Brighton'): 
        return 84
    elif (row['neighbourhood']=='South Boston'): 
        return 93
    elif (row['neighbourhood']=='Roxbury'): 
        return 82
    elif (row['neighbourhood']=='North End'): 
        return 98
    elif (row['neighbourhood']=='Charlestown'): 
        return 89
    elif (row['neighbourhood']=='Fenway/Kenmore'): 
        return 94
    elif (row['neighbourhood']=='Revere'): 
        return 63
    elif (row['neighbourhood']=='Dorchester'): 
        return 87
    elif (row['neighbourhood']=='Jamaica Plain'): 
        return 72
    elif (row['neighbourhood']=='Mission Hill'): 
        return 88
    elif (row['neighbourhood']=='Hyde Park'): 
        return 86
    elif (row['neighbourhood']=='South End'): 
        return 96
    elif (row['neighbourhood']=='Leather District'): 
        return 94
    elif (row['neighbourhood']=='Financial District'): 
        return 100
    elif (row['neighbourhood']=='Theater District'): 
        return 99
    elif (row['neighbourhood']=='Mattapan'): 
        return 82
    elif (row['neighbourhood']=='Government Center'): 
        return 97
    elif (row['neighbourhood']=='Downtown'): 
        return 97
    elif (row['neighbourhood']=='Downtown Crossing'): 
        return 99
    elif (row['neighbourhood']=='West Roxbury'): 
        return 78
    elif (row['neighbourhood']=='Somerville'): 
        return 86
    elif (row['neighbourhood']=='Brookline'): 
        return 78
    elif (row['neighbourhood']=='Chelsea'): 
        return 99
    elif (row['neighbourhood']=='Chestnut Hill'): 
        return 59
    elif (row['neighbourhood']=='Cambridge'): 
        return 87
    elif (row['neighbourhood']=='Pilsen'): 
        return 88
    elif (row['neighbourhood']=='North Center'): 
        return 88
    elif (row['neighbourhood']=='Ukrainian Village'): 
        return 94
    elif (row['neighbourhood']=='Wicker Park'): 
        return 94
    elif (row['neighbourhood']=='Irving Park'): 
        return 85
    elif (row['neighbourhood']=='Bronzeville'): 
        return 74
    elif (row['neighbourhood']=='River North'): 
        return 97
    elif (row['neighbourhood']=='Edgewater'): 
        return 89
    elif (row['neighbourhood']=='Lakeview'): 
        return 91
    elif (row['neighbourhood']=='Norwood Park'): 
        return 62
    elif (row['neighbourhood']=='Back of the Yards'): 
        return 76
    elif (row['neighbourhood']=='Uptown'): 
        return 91
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Roscoe Village'):
        return 88
    elif (row['neighbourhood']=='Avondale'): 
        return 84
    elif (row['neighbourhood']=='Portage Park'): 
        return 74
    elif (row['neighbourhood']=='Lincoln Square'): 
        return 82
    elif (row['neighbourhood']=='Humboldt Park'): 
        return 84
    elif (row['neighbourhood']=='Logan Square'): 
        return 88
    elif (row['neighbourhood']=='Hyde Park'): 
        return 86
    elif (row['neighbourhood']=='West Town/Noble Square'): 
        return 96
    elif (row['neighbourhood']=='Bridgeport'): 
        return 81
    elif (row['neighbourhood']=='West Ridge'): 
        return 81
    elif (row['neighbourhood']=='West Town'): 
        return 91
    elif (row['neighbourhood']=='Woodlawn'): 
        return 69
    elif (row['neighbourhood']=='Little Italy/UIC'):
        return 93
    elif (row['neighbourhood']=='Lincoln Park'):
        return 94
    elif (row['neighbourhood']=='Loop'): 
        return 98
    elif (row['neighbourhood']=='Andersonville'): 
        return 95
    elif (row['neighbourhood']=='South Loop/Printers Row'): 
        return 97
    elif (row['neighbourhood']=='Near North Side'): 
        return 96
    elif (row['neighbourhood']=='Old Town'): 
        return 93
    elif (row['neighbourhood']=='Wrigleyville'): 
        return 93
    elif (row['neighbourhood']=='River West'): 
        return 84
    elif (row['neighbourhood']=='Rogers Park'): 
        return 86
    elif (row['neighbourhood']=='Kenwood'): 
        return 79
    elif (row['neighbourhood']=='Little Village'): 
        return 82
    elif (row['neighbourhood']=='Gold Coast'): 
        return 99
    elif (row['neighbourhood']=='West Loop/Greektown'): 
        return 94
    elif (row['neighbourhood']=='Streeterville'): 
        return 97
    elif (row['neighbourhood']=='Archer Heights'): 
        return 74
    elif (row['neighbourhood']=='Near West Side'): 
        return 86
    elif (row['neighbourhood']=='Oakland'): 
        return 55
    elif (row['neighbourhood']=='North Park'): 
        return 73
    elif (row['neighbourhood']=='Boystown'): 
        return 95
    elif (row['neighbourhood']=='Albany Park'): 
        return 87
    elif (row['neighbourhood']=='Garfield Park'): 
        return 84
    elif (row['neighbourhood']=='Grand Crossing'): 
        return 76
    elif (row['neighbourhood']=='Bucktown'): 
        return 91
    elif (row['neighbourhood']=='Pullman'): 
        return 49
    elif (row['neighbourhood']=='Belmont Cragin'): 
        return 82
    elif (row['neighbourhood']=='Jefferson Park'): 
        return 72
    elif (row['neighbourhood']=='South Chicago'): 
        return 70
    elif (row['neighbourhood']=='Armour Square'): 
        return 88
    elif (row['neighbourhood']=='Calumet Heights'): 
        return 63
    elif (row['neighbourhood']=='West Elsdon'): 
        return 70
    elif (row['neighbourhood']=='Dunning'): 
        return 68
    elif (row['neighbourhood']=='West Lawn'): 
        return 70
    elif (row['neighbourhood']=='Beverly'): 
        return 64
    elif (row['neighbourhood']=='Washington Park'): 
        return 79
    elif (row['neighbourhood']=='South Deering'): 
        return 49
    elif (row['neighbourhood']=='South Shore'): 
        return 75
    elif (row['neighbourhood']=='Chatham'): 
        return 82
    elif (row['neighbourhood']=='North Lawndale'): 
        return 72
    elif (row['neighbourhood']=='Englewood'): 
        return 70
    elif (row['neighbourhood']=='Morgan Park'): 
        return 66
    elif (row['neighbourhood']=='Sauganash'): 
        return 55
    elif (row['neighbourhood']=='Magnificent Mile'): 
        return 99
    elif (row['neighbourhood']=='McKinley Park'): 
        return 75
    elif (row['neighbourhood']=='Friendship Heights'): 
        return 92
    elif (row['neighbourhood']=='Kingman Park'): 
        return 82
    elif (row['neighbourhood']=='Southwest Waterfront'): 
        return 81
    elif (row['neighbourhood']=='Capitol Hill'): 
        return 86
    elif (row['neighbourhood']=='Columbia Heights'): 
        return 94
    elif (row['neighbourhood']=='Burleith'): 
        return 85
    elif (row['neighbourhood']=='Adams Morgan'): 
        return 95
    elif (row['neighbourhood']=='Mount Pleasant'): 
        return 90
    elif (row['neighbourhood']=='16th Street Heights'): 
        return 84
    elif (row['neighbourhood']=='Dupont Circle'): 
        return 98
    elif (row['neighbourhood']=='Georgetown'): 
        return 39
    elif (row['neighbourhood']=='Near Northeast/H Street Corridor'): 
        return 95
    elif (row['neighbourhood']=='Carver Langston'): 
        return 82
    elif (row['neighbourhood']=='Downtown/Penn Quarter'): 
        return 98
    elif (row['neighbourhood']=='U Street Corridor'): 
        return 99
    elif (row['neighbourhood']=='Petworth'): 
        return 85
    elif (row['neighbourhood']=='Bloomingdale'): 
        return 92
    elif (row['neighbourhood']=='Takoma Park, MD'): 
        return 81
    elif (row['neighbourhood']=='LeDroit Park'): 
        return 93
    elif (row['neighbourhood']=='Pleasant Hill'): 
        return 67
    elif (row['neighbourhood']=='Logan Circle'): 
        return 98
    elif (row['neighbourhood']=='Eastland Gardens'): 
        return 49
    elif (row['neighbourhood']=='Benning Ridge'): 
        return 58
    elif (row['neighbourhood']=='Mount Vernon Square'): 
        return 97
    elif (row['neighbourhood']=='Bellevue'): 
        return 69
    elif (row['neighbourhood']=='Kalorama'): 
        return 88
    elif (row['neighbourhood']=='Edgewood'): 
        return 79
    elif (row['neighbourhood']=='Barney Circle'): 
        return 69
    elif (row['neighbourhood']=='Eckington'): 
        return 84
    elif (row['neighbourhood']=='Glover Park'): 
        return 78
    elif (row['neighbourhood']=='Brookland'): 
        return 75
    elif (row['neighbourhood']=='Park View'): 
        return 92
    elif (row['neighbourhood']=='Michigan Park'): 
        return 58
    elif (row['neighbourhood']=='Cathedral Heights'): 
        return 78
    elif (row['neighbourhood']=='Shaw'): 
        return 98
    elif (row['neighbourhood']=='Fairlawn'): 
        return 73
    elif (row['neighbourhood']=='Foggy Bottom'): 
        return 91
    elif (row['neighbourhood']=='Washington Highlands'): 
        return 63
    elif (row['neighbourhood']=='Deanwood'): 
        return 51
    elif (row['neighbourhood']=='Cleveland Park'): 
        return 73
    elif (row['neighbourhood']=='Shipley Terrace'): 
        return 53
    elif (row['neighbourhood']=='West End'): 
        return 95
    elif (row['neighbourhood']=='Brentwood'): 
        return 54
    elif (row['neighbourhood']=='Judiciary Square'): 
        return 97
    elif (row['neighbourhood']=='Randle Highlands'): 
        return 69
    elif (row['neighbourhood']=='Chevy Chase'): 
        return 64
    elif (row['neighbourhood']=='Pleasant Plains'): 
        return 58
    elif (row['neighbourhood']=='Trinidad'): 
        return 80
    elif (row['neighbourhood']=='Woodridge'): 
        return 82
    elif (row['neighbourhood']=='Anacostia'): 
        return 64
    elif (row['neighbourhood']=='Palisades'): 
        return 56
    elif (row['neighbourhood']=='Garfield Heights'): 
        return 41
    elif (row['neighbourhood']=='Massachusetts Heights'): 
        return 60
    elif (row['neighbourhood']=='Truxton Circle'): 
        return 92
    elif (row['neighbourhood']=='Navy Yard'): 
        return 88
    elif (row['neighbourhood']=='Brightwood'): 
        return 90
    elif (row['neighbourhood']=='Shepherd Park'): 
        return 85
    elif (row['neighbourhood']=='Buena Vista'): 
        return 92
    elif (row['neighbourhood']=='Manor Park'): 
        return 77
    elif (row['neighbourhood']=='Stronghold'): 
        return 73
    elif (row['neighbourhood']=='American University Park'): 
        return 82
    elif (row['neighbourhood']=='North Cleveland Park'): 
        return 83
    elif (row['neighbourhood']=='Lamond Riggs'): 
        return 75
    elif (row['neighbourhood']=='Fort Lincoln'): 
        return 53
    elif (row['neighbourhood']=='Forest Hills'): 
        return 91
    elif (row['neighbourhood']=='Central Northeast/Mahaning Heights'): 
        return 99
    elif (row['neighbourhood']=='Langdon'): 
        return 74
    elif (row['neighbourhood']=='Good Hope'): 
        return 68
    elif (row['neighbourhood']=='Lincoln Heights'): 
        return 78
    elif (row['neighbourhood']=='Takoma'): 
        return 83
    elif (row['neighbourhood']=='Woodley Park'): 
        return 72
    elif (row['neighbourhood']=='Congress Heights'): 
        return 63
    elif (row['neighbourhood']=='Benning'): 
        return 71
    elif (row['neighbourhood']=='Marshall Heights'): 
        return 67
    elif (row['neighbourhood']=='Kent'): 
        return 47
    elif (row['neighbourhood']=='Colonial Village'): 
        return 44
    elif (row['neighbourhood']=='Fort Davis'): 
        return 53
    elif (row['neighbourhood']=='Ivy City'): 
        return 77
    elif (row['neighbourhood']=='River Terrace'): 
        return 57
    elif (row['neighbourhood']=='Crestwood'): 
        return 55
    elif (row['neighbourhood']=='Greenway'): 
        return 34
    elif (row['neighbourhood']=='Fort Dupont'): 
        return 58
    elif (row['neighbourhood']=='Knox Hill'): 
        return 36
    elif (row['neighbourhood']=='Douglass'): 
        return 67
    elif (row['neighbourhood']=='North Michigan Park'): 
        return 58
    elif (row['neighbourhood']=='Bethesda, MD'): 
        return 46
    elif (row['neighbourhood']=='Hillbrook'): 
        return 39
    elif (row['neighbourhood']=='Twining'): 
        return 71
    elif (row['neighbourhood']=='Gallaudet'): 
        return 83
    elif (row['neighbourhood']=='Foxhall'): 
        return 63
    elif (row['neighbourhood']=='Mt Rainier/Brentwood, MD'): 
        return 80
    elif (row['neighbourhood']=='Wesley Heights'): 
        return 50
    elif (row['neighbourhood']=='Santa Monica'): 
        return 83
    elif (row['neighbourhood']=='Marina Del Rey'): 
        return 64
    elif (row['neighbourhood']=='Palms'): 
        return 87
    elif (row['neighbourhood']=='Westlake'): 
        return 91
    elif (row['neighbourhood']=='Lawndale'): 
        return 72
    elif (row['neighbourhood']=='Mid-Wilshire'): 
        return 96
    elif (row['neighbourhood']=='San Pedro'): 
        return 86
    elif (row['neighbourhood']=='East Hollywood'): 
        return 89
    elif (row['neighbourhood']=='Los Feliz'): 
        return 80
    elif (row['neighbourhood']=='West Los Angeles'): 
        return 87
    elif (row['neighbourhood']=='Hollywood'): 
        return 90
    elif (row['neighbourhood']=='Long Beach'): 
        return 70
    elif (row['neighbourhood']=='Echo Park'): 
        return 85
    elif (row['neighbourhood']=='Venice'): 
        return 82
    elif (row['neighbourhood']=='Culver City'): 
        return 73
    elif (row['neighbourhood']=='Highland Park'): 
        return 75
    elif (row['neighbourhood']=='Woodland Hills/Warner Center'): 
        return 46
    elif (row['neighbourhood']=='El Segundo'): 
        return 69
    elif (row['neighbourhood']=='Tarzana'): 
        return 53
    elif (row['neighbourhood']=='Arcadia'): 
        return 36
    elif (row['neighbourhood']=='La Crescenta-Montrose'): 
        return 52
    elif (row['neighbourhood']=='Monrovia'): 
        return 60
    elif (row['neighbourhood']=='Encino'): 
        return 47
    elif (row['neighbourhood']=='Pacific Palisades'): 
        return 36
    elif (row['neighbourhood']=='Bell'): 
        return 88
    elif (row['neighbourhood']=='Hermosa Beach'): 
        return 84
    elif (row['neighbourhood']=='Valley Village'): 
        return 72
    elif (row['neighbourhood']=='Downtown'): 
        return 97
    elif (row['neighbourhood']=='North Hollywood'): 
        return 83
    elif (row['neighbourhood']=='Del Rey'): 
        return 73
    elif (row['neighbourhood']=='Eagle Rock'): 
        return 70
    elif (row['neighbourhood']=='Malibu'): 
        return 17
    elif (row['neighbourhood']=='Glendale'): 
        return 84
    elif (row['neighbourhood']=='West Adams'): 
        return 72
    elif (row['neighbourhood']=='West Hills'): 
        return 41
    elif (row['neighbourhood']=='South LA'): 
        return 84
    elif (row['neighbourhood']=='Bradbury'): 
        return 28
    elif (row['neighbourhood']=='San Marino'): 
        return 70
    elif (row['neighbourhood']=='Hollywood Hills'): 
        return 76
    elif (row['neighbourhood']=='Westwood'): 
        return 69
    elif (row['neighbourhood']=='West Hollywood'): 
        return 96
    elif (row['neighbourhood']=='Mar Vista'): 
        return 70
    elif (row['neighbourhood']=='Hawthorne'): 
        return 69
    elif (row['neighbourhood']=='Alhambra'): 
        return 81
    elif (row['neighbourhood']=='Redondo Beach'): 
        return 79
    elif (row['neighbourhood']=='Silver Lake'): 
        return 61
    elif (row['neighbourhood']=='Mid-City'): 
        return 72
    elif (row['neighbourhood']=='Brentwood'): 
        return 54
    elif (row['neighbourhood']=='Laurel Canyon'): 
        return 86
    elif (row['neighbourhood']=='Cahuenga Pass'): 
        return 39
    elif (row['neighbourhood']=='Sherman Oaks'): 
        return 62
    elif (row['neighbourhood']=='Lomita'): 
        return 69
    elif (row['neighbourhood']=='Boyle Heights'): 
        return 81
    elif (row['neighbourhood']=='Valley Glen'): 
        return 58
    elif (row['neighbourhood']=='South Pasadena'): 
        return 65
    elif (row['neighbourhood']=='Inglewood'): 
        return 69
    elif (row['neighbourhood']=='Beverly Hills'): 
        return 78
    elif (row['neighbourhood']=='Burbank'): 
        return 69
    elif (row['neighbourhood']=='Westchester/Playa Del Rey'): 
        return 59
    elif (row['neighbourhood']=='Toluca Lake'): 
        return 78
    elif (row['neighbourhood']=='Altadena'): 
        return 82
    elif (row['neighbourhood']=='Irwindale'): 
        return 52
    elif (row['neighbourhood']=='South Robertson'): 
        return 82
    elif (row['neighbourhood']=='Bel Air/Beverly Crest'): 
        return 24
    elif (row['neighbourhood']=='Westside'): 
        return 75
    elif (row['neighbourhood']=='Arts District'): 
        return 87
    elif (row['neighbourhood']=='Rosemead'): 
        return 61
    elif (row['neighbourhood']=='Pasadena'): 
        return 66
    elif (row['neighbourhood']=='Glassell Park'): 
        return 62
    elif (row['neighbourhood']=='Whittier'): 
        return 86
    elif (row['neighbourhood']=='Montebello'): 
        return 64
    elif (row['neighbourhood']=='Atwater Village'): 
        return 74
    elif (row['neighbourhood']=='Lynwood'): 
        return 65
    elif (row['neighbourhood']=='Mission Hills'): 
        return 46
    elif (row['neighbourhood']=='Lenox'): 
        return 78
    elif (row['neighbourhood']=='Hermon'): 
        return 62
    elif (row['neighbourhood']=='Monterey Park'): 
        return 61
    elif (row['neighbourhood']=='San Gabriel'): 
        return 69
    elif (row['neighbourhood']=='Montecito Heights'): 
        return 41
    elif (row['neighbourhood']=='Temple City'): 
        return 55
    elif (row['neighbourhood']=='Canoga Park'): 
        return 66
    elif (row['neighbourhood']=='Van Nuys'): 
        return 70
    elif (row['neighbourhood']=='Northridge'): 
        return 48
    elif (row['neighbourhood']=='Topanga'): 
        return 50
    elif (row['neighbourhood']=='West Covina'): 
        return 45
    elif (row['neighbourhood']=='Harbor City'): 
        return 68
    elif (row['neighbourhood']=='Studio City'): 
        return 63
    elif (row['neighbourhood']=='Manhattan Beach'): 
        return 78
    elif (row['neighbourhood']=='Reseda'): 
        return 60
    elif (row['neighbourhood']=='Mount Washington'): 
        return 51
    elif (row['neighbourhood']=='Lincoln Heights'): 
        return 78
    elif (row['neighbourhood']=='La Canada Flintridge'): 
        return 31
    elif (row['neighbourhood']=='Sunland/Tujunga'): 
        return 56
    elif (row['neighbourhood']=='Glendora'): 
        return 44
    elif (row['neighbourhood']=='Granada Hills North'): 
        return 48
    elif (row['neighbourhood']=='Norwalk'): 
        return 58
    elif (row['neighbourhood']=='Paramount'): 
        return 63
    elif (row['neighbourhood']=='Rancho Palos Verdes'): 
        return 21
    elif (row['neighbourhood']=='Gardena'): 
        return 74
    elif (row['neighbourhood']=='Signal Hill'): 
        return 77
    elif (row['neighbourhood']=='Carson'): 
        return 57
    elif (row['neighbourhood']=='Torrance'): 
        return 64
    elif (row['neighbourhood']=='Baldwin Hills'): 
        return 59
    elif (row['neighbourhood']=='Pico Rivera'): 
        return 56
    elif (row['neighbourhood']=='La Mirada'): 
        return 47
    elif (row['neighbourhood']=='Porter Ranch'): 
        return 22
    elif (row['neighbourhood']=='El Monte'): 
        return 60
    elif (row['neighbourhood']=='Chatsworth'): 
        return 48
    elif (row['neighbourhood']=='West Rancho Dominguez'): 
        return 36
    elif (row['neighbourhood']=='Elysian Valley'): 
        return 42
    elif (row['neighbourhood']=='Azusa'): 
        return 66
    elif (row['neighbourhood']=='El Sereno'): 
        return 39
    elif (row['neighbourhood']=='Skid Row'): 
        return 93
    elif (row['neighbourhood']=='Harbor Gateway'): 
        return 61
    elif (row['neighbourhood']=='Cerritos'): 
        return 54
    elif (row['neighbourhood']=='East Los Angeles'): 
        return 73
    elif (row['neighbourhood']=='South San Gabriel'): 
        return 44
    elif (row['neighbourhood']=='Compton'): 
        return 63
    elif (row['neighbourhood']=='East San Gabriel'): 
        return 79
    elif (row['neighbourhood']=='Sylmar'): 
        return 45
    elif (row['neighbourhood']=='Bellflower'): 
        return 62
    elif (row['neighbourhood']=='Winnetka'): 
        return 57
    elif (row['neighbourhood']=='Lakewood'): 
        return 53
    elif (row['neighbourhood']=='Watts'): 
        return 62
    elif (row['neighbourhood']=='Baldwin Park'): 
        return 86
    elif (row['neighbourhood']=='Panorama City'): 
        return 65
    elif (row['neighbourhood']=='Pacoima'): 
        return 58
    elif (row['neighbourhood']=='Huntington Park'): 
        return 80
    elif (row['neighbourhood']=='Monterey Hills'): 
        return 9
    elif (row['neighbourhood']=='Sierra Madre'): 
        return 81
    elif (row['neighbourhood']=='Lake Balboa'): 
        return 58
    elif (row['neighbourhood']=='Alondra Park'): 
        return 66
    elif (row['neighbourhood']=='South El Monte'): 
        return 64
    elif (row['neighbourhood']=='Cypress Park'): 
        return 73
    elif (row['neighbourhood']=='Westmont'): 
        return 61
    elif (row['neighbourhood']=='Duarte'): 
        return 47
    elif (row['neighbourhood']=='Palos Verdes'): 
        return 21
    elif (row['neighbourhood']=='Downey'): 
        return 59
    elif (row['neighbourhood']=='North Hills West'): 
        return 46
    elif (row['neighbourhood']=='South Whittier'): 
        return 47
    elif (row['neighbourhood']=='Sun Valley'): 
        return 54
    elif (row['neighbourhood']=='Rolling Hills Estates'): 
        return 18
    elif (row['neighbourhood']=='Florence-Graham'): 
        return 73
    elif (row['neighbourhood']=='Wilmington'): 
        return 68
    elif (row['neighbourhood']=='Williamsburg'): 
        return 96
    elif (row['neighbourhood']=='West Village'): 
        return 100
    elif (row['neighbourhood']=='Washington Heights'): 
        return 97
    elif (row['neighbourhood']=='Midtown East'): 
        return 99
    elif (row['neighbourhood']=="Hell's Kitchen"): 
        return 98
    elif (row['neighbourhood']=='Woodside'): 
        return 94
    elif (row['neighbourhood']=='Bushwick'): 
        return 95
    elif (row['neighbourhood']=='Meatpacking District'): 
        return 99
    elif (row['neighbourhood']=='Upper West Side'): 
        return 98
    elif (row['neighbourhood']=='East New York'): 
        return 86
    elif (row['neighbourhood']=='Ridgewood'): 
        return 95
    elif (row['neighbourhood']=='Graniteville'): 
        return 68
    elif (row['neighbourhood']=='Alphabet City'): 
        return 97
    elif (row['neighbourhood']=='Lower East Side'): 
        return 96
    elif (row['neighbourhood']=='Carroll Gardens'): 
        return 97
    elif (row['neighbourhood']=='Midtown'): 
        return 99
    elif (row['neighbourhood']=='Hamilton Heights'): 
        return 98
    elif (row['neighbourhood']=='Greenpoint'): 
        return 96
    elif (row['neighbourhood']=='Chelsea'): 
        return 99
    elif (row['neighbourhood']=='Upper East Side'): 
        return 99
    elif (row['neighbourhood']=='Kensington'): 
        return 95
    elif (row['neighbourhood']=='Crown Heights'): 
        return 95
    elif (row['neighbourhood']=='Bedford-Stuyvesant'): 
        return 94
    elif (row['neighbourhood']=='Coney Island'): 
        return 82
    elif (row['neighbourhood']=='Soho'): 
        return 100
    elif (row['neighbourhood']=='Rego Park'): 
        return 92
    elif (row['neighbourhood']=='Williamsbridge'): 
        return 87
    elif (row['neighbourhood']=='Sunnyside'): 
        return 69
    elif (row['neighbourhood']=='Harlem'): 
        return 98
    elif (row['neighbourhood']=='East Harlem'): 
        return 96
    elif (row['neighbourhood']=='Fort Greene'): 
        return 97
    elif (row['neighbourhood']=='Lefferts Garden'): 
        return 96
    elif (row['neighbourhood']=='Kew Garden Hills'): 
        return 90
    elif (row['neighbourhood']=='Long Island City'): 
        return 95
    elif (row['neighbourhood']=='Financial District'): 
        return 100
    elif (row['neighbourhood']=='Boerum Hill'): 
        return 98
    elif (row['neighbourhood']=='Astoria'): 
        return 92
    elif (row['neighbourhood']=='Flatbush'): 
        return 94
    elif (row['neighbourhood']=='The Rockaways'): 
        return 79
    elif (row['neighbourhood']=='East Village'): 
        return 98
    elif (row['neighbourhood']=='Battery Park City'): 
        return 97
    elif (row['neighbourhood']=='Flushing'): 
        return 89
    elif (row['neighbourhood']=='Greenwood Heights'): 
        return 91
    elif (row['neighbourhood']=='Gowanus'): 
        return 97
    elif (row['neighbourhood']=='Kips Bay'): 
        return 99
    elif (row['neighbourhood']=='Jackson Heights'): 
        return 93
    elif (row['neighbourhood']=='Times Square/Theatre District'): 
        return 99
    elif (row['neighbourhood']=='Roosevelt Island'): 
        return 77
    elif (row['neighbourhood']=='Wakefield'): 
        return 87
    elif (row['neighbourhood']=='Clinton Hill'): 
        return 96
    elif (row['neighbourhood']=='Brooklyn Navy Yard'): 
        return 65
    elif (row['neighbourhood']=='Jamaica'): 
        return 88
    elif (row['neighbourhood']=='Corona'): 
        return 93
    elif (row['neighbourhood']=='Morningside Heights'): 
        return 96
    elif (row['neighbourhood']=='Midwood'): 
        return 91
    elif (row['neighbourhood']=='Murray Hill'): 
        return 99
    elif (row['neighbourhood']=='Maspeth'): 
        return 84
    elif (row['neighbourhood']=='DUMBO'): 
        return 98
    elif (row['neighbourhood']=='Flatiron District'): 
        return 100
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Brooklyn Heights'): 
        return 98
    elif (row['neighbourhood']=='Windsor Terrace'): 
        return 90
    elif (row['neighbourhood']=='Union Square'): 
        return 100
    elif (row['neighbourhood']=='Tompkinsville'): 
        return 78
    elif (row['neighbourhood']=='Gramercy Park'): 
        return 100
    elif (row['neighbourhood']=='Howard Beach'): 
        return 69
    elif (row['neighbourhood']=='Fort Wadsworth'): 
        return 58
    elif (row['neighbourhood']=='Highbridge'): 
        return 93
    elif (row['neighbourhood']=='New Brighton'): 
        return 69
    elif (row['neighbourhood']=='Crotona'): 
        return 91
    elif (row['neighbourhood']=='Woodhaven'): 
        return 88
    elif (row['neighbourhood']=='Park Slope'): 
        return 97
    elif (row['neighbourhood']=='Sunset Park'): 
        return 95
    elif (row['neighbourhood']=='Ozone Park'): 
        return 88
    elif (row['neighbourhood']=='Greenwich Village'): 
        return 100
    elif (row['neighbourhood']=='East Flatbush'): 
        return 91
    elif (row['neighbourhood']=='Brighton Beach'): 
        return 96
    elif (row['neighbourhood']=='Stapleton'): 
        return 81
    elif (row['neighbourhood']=='Bay Ridge'): 
        return 91
    elif (row['neighbourhood']=='Sheepshead Bay'): 
        return 92
    elif (row['neighbourhood']=='Mott Haven'): 
        return 96
    elif (row['neighbourhood']=='Tremont'): 
        return 96
    elif (row['neighbourhood']=='Tribeca'): 
        return 99
    elif (row['neighbourhood']=='Nolita'): 
        return 100
    elif (row['neighbourhood']=='Downtown Brooklyn'): 
        return 97
    elif (row['neighbourhood']=='Pelham Bay'): 
        return 87
    elif (row['neighbourhood']=='Gravesend'): 
        return 91
    elif (row['neighbourhood']=='Prospect Heights'): 
        return 97
    elif (row['neighbourhood']=='Inwood'): 
        return 96
    elif (row['neighbourhood']=='Bensonhurst'): 
        return 93
    elif (row['neighbourhood']=='Elmhurst'): 
        return 95
    elif (row['neighbourhood']=='Columbia Street Waterfront'): 
        return 92
    elif (row['neighbourhood']=='Marble Hill'): 
        return 92
    elif (row['neighbourhood']=='Claremont'): 
        return 0
    elif (row['neighbourhood']=='Bath Beach'): 
        return 88
    elif (row['neighbourhood']=='Concourse Village'): 
        return 96
    elif (row['neighbourhood']=='Morrisania'): 
        return 94
    elif (row['neighbourhood']=='Flatlands'): 
        return 90
    elif (row['neighbourhood']=='Bronxdale'): 
        return 93
    elif (row['neighbourhood']=='Forest Hills'): 
        return 91
    elif (row['neighbourhood']=='Riverdale'): 
        return 80
    elif (row['neighbourhood']=='Red Hook'): 
        return 91
    elif (row['neighbourhood']=='Allerton'): 
        return 90
    elif (row['neighbourhood']=='Grymes Hill'): 
        return 57
    elif (row['neighbourhood']=='Eastchester'): 
        return 77
    elif (row['neighbourhood']=='Cobble Hill'): 
        return 97
    elif (row['neighbourhood']=='Hudson Square'): 
        return 100
    elif (row['neighbourhood']=='Mount Eden'): 
        return 96
    elif (row['neighbourhood']=='Canarsie'): 
        return 84
    elif (row['neighbourhood']=='Little Italy'): 
        return 100
    elif (row['neighbourhood']=='Civic Center'): 
        return 99
    elif (row['neighbourhood']=='Hunts Point'): 
        return 91
    elif (row['neighbourhood']=='University Heights'): 
        return 93
    elif (row['neighbourhood']=='Soundview'): 
        return 84
    elif (row['neighbourhood']=='Concourse'): 
        return 96
    elif (row['neighbourhood']=='East Elmhurst'): 
        return 87
    elif (row['neighbourhood']=='Bedford Park'): 
        return 93
    elif (row['neighbourhood']=='Parkchester'): 
        return 94
    elif (row['neighbourhood']=='Hillcrest'): 
        return 81
    elif (row['neighbourhood']=='Borough Park'): 
        return 95
    elif (row['neighbourhood']=='Mariners Harbor'): 
        return 62
    elif (row['neighbourhood']=='Richmond Hill'): 
        return 90
    elif (row['neighbourhood']=='Brownsville'): 
        return 92
    elif (row['neighbourhood']=='Clifton'): 
        return 71
    elif (row['neighbourhood']=='Randall Manor'): 
        return 77
    elif (row['neighbourhood']=='Spuyten Duyvil'): 
        return 75
    elif (row['neighbourhood']=='West Brighton'): 
        return 79
    elif (row['neighbourhood']=='Kingsbridge'): 
        return 92
    elif (row['neighbourhood']=='New Springville'): 
        return 55
    elif (row['neighbourhood']=='Glendale'): 
        return 84
    elif (row['neighbourhood']=='Midland Beach'): 
        return 73
    elif (row['neighbourhood']=='Port Morris'): 
        return 84
    elif (row['neighbourhood']=='Park Versailles'): 
        return 89
    elif (row['neighbourhood']=='St. George'): 
        return 84
    elif (row['neighbourhood']=='Ditmars / Steinway'): 
        return 96
    elif (row['neighbourhood']=='Baychester'): 
        return 84
    elif (row['neighbourhood']=='South Ozone Park'): 
        return 83
    elif (row['neighbourhood']=='Fordham'): 
        return 98
    elif (row['neighbourhood']=='Middle Village'): 
        return 80
    elif (row['neighbourhood']=='Bayside'): 
        return 73
    elif (row['neighbourhood']=='Kingsbridge Heights'): 
        return 93
    elif (row['neighbourhood']=='City Island'): 
        return 68
    elif (row['neighbourhood']=='Todt Hill'): 
        return 48
    elif (row['neighbourhood']=='Manhattan Beach'): 
        return 78
    elif (row['neighbourhood']=='Norwood'): 
        return 92
    elif (row['neighbourhood']=='Rosebank'): 
        return 76
    elif (row['neighbourhood']=='Whitestone'): 
        return 70
    elif (row['neighbourhood']=='Noho'): 
        return 100
    elif (row['neighbourhood']=='Morris Heights'): 
        return 91
    elif (row['neighbourhood']=='Throgs Neck'): 
        return 69
    elif (row['neighbourhood']=='Grasmere'): 
        return 67
    elif (row['neighbourhood']=='Woodlawn'): 
        return 69
    elif (row['neighbourhood']=='Eltingville'): 
        return 66
    elif (row['neighbourhood']=='Dongan Hills'): 
        return 75
    elif (row['neighbourhood']=='College Point'): 
        return 78
    elif (row['neighbourhood']=='Utopia'): 
        return 88
    elif (row['neighbourhood']=='Melrose'): 
        return 97
    elif (row['neighbourhood']=='Brooklyn'): 
        return 96
    elif (row['neighbourhood']=='South Street Seaport'): 
        return 97
    elif (row['neighbourhood']=='Fresh Meadows'): 
        return 76
    elif (row['neighbourhood']=='Van Nest'): 
        return 91
    elif (row['neighbourhood']=='Manhattan'): 
        return 97
    elif (row['neighbourhood']=='Longwood'): 
        return 96
    elif (row['neighbourhood']=='Dyker Heights'): 
        return 90
    elif (row['neighbourhood']=='Concord'): 
        return 63
    elif (row['neighbourhood']=='Great Kills'): 
        return 66
    elif (row['neighbourhood']=='Belmont'): 
        return 96
    elif (row['neighbourhood']=='New Dorp'): 
        return 77
    elif (row['neighbourhood']=='South Beach'): 
        return 72
    elif (row['neighbourhood']=='Port Richmond'): 
        return 83
    elif (row['neighbourhood']=='Vinegar Hill'): 
        return 94
    elif (row['neighbourhood']=='West Farms'): 
        return 93
    elif (row['neighbourhood']=='Lindenwood'): 
        return 72
    elif (row['neighbourhood']=='Meiers Corners'): 
        return 85
    elif (row['neighbourhood']=='Bergen Beach'): 
        return 75
    elif (row['neighbourhood']=='Queens'): 
        return 73
    elif (row['neighbourhood']=='Westchester Village'): 
        return 94
    elif (row['neighbourhood']=='Sea Gate'): 
        return 50
    elif (row['neighbourhood']=='Richmond District'): 
        return 97
    elif (row['neighbourhood']=='Glen Park'): 
        return 78
    elif (row['neighbourhood']=='Western Addition/NOPA'): 
        return 96
    elif (row['neighbourhood']=='Mission District'): 
        return 97
    elif (row['neighbourhood']=='Union Square'): 
        return 100
    elif (row['neighbourhood']=='Outer Sunset'): 
        return 78
    elif (row['neighbourhood']=='Nob Hill'): 
        return 98
    elif (row['neighbourhood']=='SoMa'): 
        return 95
    elif (row['neighbourhood']=='The Castro'): 
        return 95
    elif (row['neighbourhood']=='Haight-Ashbury'): 
        return 96
    elif (row['neighbourhood']=='Parkside'): 
        return 79
    elif (row['neighbourhood']=='Bernal Heights'): 
        return 88
    elif (row['neighbourhood']=='Presidio Heights'): 
        return 90
    elif (row['neighbourhood']=='Duboce Triangle'): 
        return 98
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Cow Hollow'): 
        return 93
    elif (row['neighbourhood']=='Downtown'): 
        return 97
    elif (row['neighbourhood']=='Marina'): 
        return 94
    elif (row['neighbourhood']=='Cole Valley'): 
        return 96
    elif (row['neighbourhood']=='Twin Peaks'): 
        return 58
    elif (row['neighbourhood']=='Hayes Valley'): 
        return 97
    elif (row['neighbourhood']=='Pacific Heights'): 
        return 96
    elif (row['neighbourhood']=='Financial District'): 
        return 100
    elif (row['neighbourhood']=='Lower Haight'): 
        return 96
    elif (row['neighbourhood']=='Noe Valley'): 
        return 91
    elif (row['neighbourhood']=='North Beach'): 
        return 99
    elif (row['neighbourhood']=='Sunnyside'): 
        return 69
    elif (row['neighbourhood']=='Russian Hill'): 
        return 96
    elif (row['neighbourhood']=='Dogpatch'): 
        return 91
    elif (row['neighbourhood']=='Tenderloin'): 
        return 99
    elif (row['neighbourhood']=='Excelsior'): 
        return 79
    elif (row['neighbourhood']=='Potrero Hill'): 
        return 87
    elif (row['neighbourhood']=='Ingleside'): 
        return 78
    elif (row['neighbourhood']=='Balboa Terrace'): 
        return 76
    elif (row['neighbourhood']=='Oceanview'): 
        return 70
    elif (row['neighbourhood']=="Fisherman's Wharf"): 
        return 98
    elif (row['neighbourhood']=='Lakeshore'): 
        return 39
    elif (row['neighbourhood']=='Daly City'): 
        return 63
    elif (row['neighbourhood']=='Inner Sunset'): 
        return 94
    elif (row['neighbourhood']=='South Beach'): 
        return 72
    elif (row['neighbourhood']=='Forest Hill'): 
        return 60
    elif (row['neighbourhood']=='Bayview'): 
        return 82
    elif (row['neighbourhood']=='Alamo Square'): 
        return 96
    elif (row['neighbourhood']=='Portola'): 
        return 76
    elif (row['neighbourhood']=='Mission Terrace'): 
        return 80
    elif (row['neighbourhood']=='Telegraph Hill'): 
        return 88
    elif (row['neighbourhood']=='Visitacion Valley'): 
        return 68
    elif (row['neighbourhood']=='Civic Center'): 
        return 99
    elif (row['neighbourhood']=='Mission Bay'): 
        return 89
    elif (row['neighbourhood']=='West Portal'): 
        return 91
    elif (row['neighbourhood']=='Crocker Amazon'): 
        return 71
    elif (row['neighbourhood']=='Diamond Heights'): 
        return 69
    elif (row['neighbourhood']=='Japantown'): 
        return 98
    elif (row['neighbourhood']=='Sea Cliff'): 
        return 81

data_train['walkscore']=data_train.apply(lambda row: walkscore(row), axis=1)

def transitscore(row):
    if (row['neighbourhood']=='Back Bay'):
        return 97
    elif (row['neighbourhood']=='Beacon Hill'):
        return 100
    elif (row['neighbourhood']=='Roslindale'):
        return 65
    elif (row['neighbourhood']=='East Boston'):
        return 67
    elif (row['neighbourhood']=='West End'):
        return 100
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Allston-Brighton'):
        return 66
    elif (row['neighbourhood']=='South Boston'):
        return 72
    elif (row['neighbourhood']=='Roxbury'):
        return 73
    elif (row['neighbourhood']=='North End'):
        return 100
    elif (row['neighbourhood']=='Charlestown'):
        return 68
    elif (row['neighbourhood']=='Fenway/Kenmore'):
        return 95
    elif (row['neighbourhood']=='Revere'):
        return 54
    elif (row['neighbourhood']=='Dorchester'):
        return 72
    elif (row['neighbourhood']=='Jamaica Plain'):
        return 80
    elif (row['neighbourhood']=='Mission Hill'):
        return 91
    elif (row['neighbourhood']=='Hyde Park'):
        return 66
    elif (row['neighbourhood']=='South End'):
        return 94
    elif (row['neighbourhood']=='Leather District'):
        return 100
    elif (row['neighbourhood']=='Financial District'):
        return 100
    elif (row['neighbourhood']=='Theater District'):
        return 100
    elif (row['neighbourhood']=='Mattapan'):
        return 69
    elif (row['neighbourhood']=='Government Center'):
        return 100
    elif (row['neighbourhood']=='Downtown'):
        return 100
    elif (row['neighbourhood']=='Downtown Crossing'):
        return 100
    elif (row['neighbourhood']=='West Roxbury'):
        return 44
    elif (row['neighbourhood']=='Somerville'):
        return 63
    elif (row['neighbourhood']=='Brookline'):
        return 68
    elif (row['neighbourhood']=='Chelsea'):
        return 100
    elif (row['neighbourhood']=='Chestnut Hill'):
        return 60
    elif (row['neighbourhood']=='Cambridge'):
        return 72
    elif (row['neighbourhood']=='Pilsen'):
        return 66
    elif (row['neighbourhood']=='North Center'):
        return 66
    elif (row['neighbourhood']=='Ukrainian Village'):
        return 70
    elif (row['neighbourhood']=='Wicker Park'):
        return 75
    elif (row['neighbourhood']=='Irving Park'):
        return 66
    elif (row['neighbourhood']=='Bronzeville'):
        return 67
    elif (row['neighbourhood']=='River North'):
        return 100
    elif (row['neighbourhood']=='Edgewater'):
        return 72
    elif (row['neighbourhood']=='Lakeview'):
        return 79
    elif (row['neighbourhood']=='Norwood Park'):
        return 51
    elif (row['neighbourhood']=='Back of the Yards'):
        return 61
    elif (row['neighbourhood']=='Uptown'):
        return 79
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Roscoe Village'):
        return 66
    elif (row['neighbourhood']=='Avondale'):
        return 68
    elif (row['neighbourhood']=='Portage Park'):
        return 59
    elif (row['neighbourhood']=='Lincoln Square'):
        return 60
    elif (row['neighbourhood']=='Humboldt Park'):
        return 67
    elif (row['neighbourhood']=='Logan Square'):
        return 68
    elif (row['neighbourhood']=='Hyde Park'):
        return 66
    elif (row['neighbourhood']=='West Town/Noble Square'):
        return 75
    elif (row['neighbourhood']=='Bridgeport'):
        return 61
    elif (row['neighbourhood']=='West Ridge'):
        return 56
    elif (row['neighbourhood']=='West Town'):
        return 75
    elif (row['neighbourhood']=='Woodlawn'):
        return 73
    elif (row['neighbourhood']=='Little Italy/UIC'):
        return 75
    elif (row['neighbourhood']=='Lincoln Park'):
        return 79
    elif (row['neighbourhood']=='Loop'):
        return 100
    elif (row['neighbourhood']=='Andersonville'):
        return 68
    elif (row['neighbourhood']=='South Loop/Printers Row'):
        return 100
    elif (row['neighbourhood']=='Near North Side'):
        return 90
    elif (row['neighbourhood']=='Old Town'):
        return 84
    elif (row['neighbourhood']=='Wrigleyville'):
        return 80
    elif (row['neighbourhood']=='River West'):
        return 80
    elif (row['neighbourhood']=='Rogers Park'):
        return 74
    elif (row['neighbourhood']=='Kenwood'):
        return 66
    elif (row['neighbourhood']=='Little Village'):
        return 62
    elif (row['neighbourhood']=='Gold Coast'):
        return 91
    elif (row['neighbourhood']=='West Loop/Greektown'):
        return 95
    elif (row['neighbourhood']=='Streeterville'):
        return 100
    elif (row['neighbourhood']=='Archer Heights'):
        return 57
    elif (row['neighbourhood']=='Near West Side'):
        return 82
    elif (row['neighbourhood']=='Oakland'):
        return 49
    elif (row['neighbourhood']=='North Park'):
        return 54
    elif (row['neighbourhood']=='Boystown'):
        return 84
    elif (row['neighbourhood']=='Albany Park'):
        return 62
    elif (row['neighbourhood']=='Garfield Park'):
        return 74
    elif (row['neighbourhood']=='Grand Crossing'):
        return 66
    elif (row['neighbourhood']=='Bucktown'):
        return 73
    elif (row['neighbourhood']=='Pullman'):
        return 62
    elif (row['neighbourhood']=='Belmont Cragin'):
        return 61
    elif (row['neighbourhood']=='Jefferson Park'):
        return 64
    elif (row['neighbourhood']=='South Chicago'):
        return 63
    elif (row['neighbourhood']=='Armour Square'):
        return 75
    elif (row['neighbourhood']=='Calumet Heights'):
        return 58
    elif (row['neighbourhood']=='West Elsdon'):
        return 62
    elif (row['neighbourhood']=='Dunning'):
        return 53
    elif (row['neighbourhood']=='West Lawn'):
        return 57
    elif (row['neighbourhood']=='Beverly'):
        return 50
    elif (row['neighbourhood']=='Washington Park'):
        return 75
    elif (row['neighbourhood']=='South Deering'):
        return 51
    elif (row['neighbourhood']=='South Shore'):
        return 68
    elif (row['neighbourhood']=='Chatham'):
        return 67
    elif (row['neighbourhood']=='North Lawndale'):
        return 62
    elif (row['neighbourhood']=='Englewood'):
        return 65
    elif (row['neighbourhood']=='Morgan Park'):
        return 50
    elif (row['neighbourhood']=='Sauganash'):
        return 41
    elif (row['neighbourhood']=='Magnificent Mile'):
        return 100
    elif (row['neighbourhood']=='McKinley Park'):
        return 64
    elif (row['neighbourhood']=='Friendship Heights'):
        return 73
    elif (row['neighbourhood']=='Kingman Park'):
        return 65
    elif (row['neighbourhood']=='Southwest Waterfront'):
        return 82
    elif (row['neighbourhood']=='Capitol Hill'):
        return 76
    elif (row['neighbourhood']=='Columbia Heights'):
        return 79
    elif (row['neighbourhood']=='Burleith'):
        return 54
    elif (row['neighbourhood']=='Adams Morgan'):
        return 78
    elif (row['neighbourhood']=='Mount Pleasant'):
        return 76
    elif (row['neighbourhood']=='16th Street Heights'):
        return 64
    elif (row['neighbourhood']=='Dupont Circle'):
        return 87
    elif (row['neighbourhood']=='Georgetown'):
        return 47
    elif (row['neighbourhood']=='Near Northeast/H Street Corridor'):
        return 73
    elif (row['neighbourhood']=='Carver Langston'):
        return 64
    elif (row['neighbourhood']=='Downtown/Penn Quarter'):
        return 100
    elif (row['neighbourhood']=='U Street Corridor'):
        return 77
    elif (row['neighbourhood']=='Petworth'):
        return 69
    elif (row['neighbourhood']=='Bloomingdale'):
        return 76
    elif (row['neighbourhood']=='Takoma Park, MD'):
        return 68
    elif (row['neighbourhood']=='LeDroit Park'):
        return 77
    elif (row['neighbourhood']=='Pleasant Hill'):
        return 75
    elif (row['neighbourhood']=='Logan Circle'):
        return 87
    elif (row['neighbourhood']=='Eastland Gardens'):
        return 66
    elif (row['neighbourhood']=='Benning Ridge'):
        return 64
    elif (row['neighbourhood']=='Mount Vernon Square'):
        return 98
    elif (row['neighbourhood']=='Bellevue'):
        return 57
    elif (row['neighbourhood']=='Kalorama'):
        return 74
    elif (row['neighbourhood']=='Edgewood'):
        return 71
    elif (row['neighbourhood']=='Barney Circle'):
        return 78
    elif (row['neighbourhood']=='Eckington'):
        return 71
    elif (row['neighbourhood']=='Glover Park'):
        return 51
    elif (row['neighbourhood']=='Brookland'):
        return 68
    elif (row['neighbourhood']=='Park View'):
        return 75
    elif (row['neighbourhood']=='Michigan Park'):
        return 64
    elif (row['neighbourhood']=='Cathedral Heights'):
        return 66
    elif (row['neighbourhood']=='Shaw'):
        return 78
    elif (row['neighbourhood']=='Fairlawn'):
        return 63
    elif (row['neighbourhood']=='Foggy Bottom'):
        return 87
    elif (row['neighbourhood']=='Washington Highlands'):
        return 60
    elif (row['neighbourhood']=='Deanwood'):
        return 61
    elif (row['neighbourhood']=='Cleveland Park'):
        return 63
    elif (row['neighbourhood']=='Shipley Terrace'):
        return 69
    elif (row['neighbourhood']=='West End'):
        return 100
    elif (row['neighbourhood']=='Brentwood'):
        return 43
    elif (row['neighbourhood']=='Judiciary Square'):
        return 98
    elif (row['neighbourhood']=='Randle Highlands'):
        return 62
    elif (row['neighbourhood']=='Chevy Chase'):
        return 51
    elif (row['neighbourhood']=='Pleasant Plains'):
        return 51
    elif (row['neighbourhood']=='Trinidad'):
        return 59
    elif (row['neighbourhood']=='Woodridge'):
        return 56
    elif (row['neighbourhood']=='Anacostia'):
        return 67
    elif (row['neighbourhood']=='Palisades'):
        return 40
    elif (row['neighbourhood']=='Garfield Heights'):
        return 60
    elif (row['neighbourhood']=='Massachusetts Heights'):
        return 58
    elif (row['neighbourhood']=='Truxton Circle'):
        return 77
    elif (row['neighbourhood']=='Navy Yard'):
        return 69
    elif (row['neighbourhood']=='Brightwood'):
        return 67
    elif (row['neighbourhood']=='Shepherd Park'):
        return 68
    elif (row['neighbourhood']=='Buena Vista'):
        return 89
    elif (row['neighbourhood']=='Manor Park'):
        return 60
    elif (row['neighbourhood']=='Stronghold'):
        return 66
    elif (row['neighbourhood']=='American University Park'):
        return 65
    elif (row['neighbourhood']=='North Cleveland Park'):
        return 67
    elif (row['neighbourhood']=='Lamond Riggs'):
        return 55
    elif (row['neighbourhood']=='Fort Lincoln'):
        return 43
    elif (row['neighbourhood']=='Forest Hills'):
        return 91
    elif (row['neighbourhood']=='Central Northeast/Mahaning Heights'):
        return 100
    elif (row['neighbourhood']=='Langdon'):
        return 56
    elif (row['neighbourhood']=='Good Hope'):
        return 60
    elif (row['neighbourhood']=='Lincoln Heights'):
        return 59
    elif (row['neighbourhood']=='Takoma'):
        return 78
    elif (row['neighbourhood']=='Woodley Park'):
        return 65
    elif (row['neighbourhood']=='Congress Heights'):
        return 62
    elif (row['neighbourhood']=='Benning'):
        return 70
    elif (row['neighbourhood']=='Marshall Heights'):
        return 63
    elif (row['neighbourhood']=='Kent'):
        return 41
    elif (row['neighbourhood']=='Colonial Village'):
        return 68
    elif (row['neighbourhood']=='Fort Davis'):
        return 46
    elif (row['neighbourhood']=='Ivy City'):
        return 64
    elif (row['neighbourhood']=='River Terrace'):
        return 63
    elif (row['neighbourhood']=='Crestwood'):
        return 55
    elif (row['neighbourhood']=='Greenway'):
        return 54
    elif (row['neighbourhood']=='Fort Dupont'):
        return 56
    elif (row['neighbourhood']=='Knox Hill'):
        return 64
    elif (row['neighbourhood']=='Douglass'):
        return 65
    elif (row['neighbourhood']=='North Michigan Park'):
        return 51
    elif (row['neighbourhood']=='Bethesda, MD'):
        return 45
    elif (row['neighbourhood']=='Hillbrook'):
        return 43
    elif (row['neighbourhood']=='Twining'):
        return 60
    elif (row['neighbourhood']=='Gallaudet'):
        return 62
    elif (row['neighbourhood']=='Foxhall'):
        return 39
    elif (row['neighbourhood']=='Mt Rainier/Brentwood, MD'):
        return 72
    elif (row['neighbourhood']=='Wesley Heights'):
        return 44
    elif (row['neighbourhood']=='Santa Monica'):
        return 63
    elif (row['neighbourhood']=='Marina Del Rey'):
        return 48
    elif (row['neighbourhood']=='Palms'):
        return 61
    elif (row['neighbourhood']=='Westlake'):
        return 83
    elif (row['neighbourhood']=='Lawndale'):
        return 66
    elif (row['neighbourhood']=='Mid-Wilshire'):
        return 79
    elif (row['neighbourhood']=='San Pedro'):
        return 46
    elif (row['neighbourhood']=='East Hollywood'):
        return 66
    elif (row['neighbourhood']=='Los Feliz'):
        return 60
    elif (row['neighbourhood']=='West Los Angeles'):
        return 66
    elif (row['neighbourhood']=='Hollywood'):
        return 62
    elif (row['neighbourhood']=='Long Beach'):
        return 51
    elif (row['neighbourhood']=='Echo Park'):
        return 63
    elif (row['neighbourhood']=='Venice'):
        return 53
    elif (row['neighbourhood']=='Culver City'):
        return 51
    elif (row['neighbourhood']=='Highland Park'):
        return 87
    elif (row['neighbourhood']=='Woodland Hills/Warner Center'):
        return 40
    elif (row['neighbourhood']=='El Segundo'):
        return 0
    elif (row['neighbourhood']=='Tarzana'):
        return 40
    elif (row['neighbourhood']=='Arcadia'):
        return 0
    elif (row['neighbourhood']=='La Crescenta-Montrose'):
        return 0
    elif (row['neighbourhood']=='Monrovia'):
        return 32
    elif (row['neighbourhood']=='Encino'):
        return 35
    elif (row['neighbourhood']=='Pacific Palisades'):
        return 27
    elif (row['neighbourhood']=='Bell'):
        return 0
    elif (row['neighbourhood']=='Hermosa Beach'):
        return 0
    elif (row['neighbourhood']=='Valley Village'):
        return 45
    elif (row['neighbourhood']=='Downtown'):
        return 100
    elif (row['neighbourhood']=='North Hollywood'):
        return 51
    elif (row['neighbourhood']=='Del Rey'):
        return 45
    elif (row['neighbourhood']=='Eagle Rock'):
        return 45
    elif (row['neighbourhood']=='Malibu'):
        return 0
    elif (row['neighbourhood']=='Glendale'):
        return 70
    elif (row['neighbourhood']=='West Adams'):
        return 63
    elif (row['neighbourhood']=='West Hills'):
        return 34
    elif (row['neighbourhood']=='South LA'):
        return 67
    elif (row['neighbourhood']=='Bradbury'):
        return 0
    elif (row['neighbourhood']=='San Marino'):
        return 0
    elif (row['neighbourhood']=='Hollywood Hills'):
        return 53
    elif (row['neighbourhood']=='Westwood'):
        return 66
    elif (row['neighbourhood']=='West Hollywood'):
        return 59
    elif (row['neighbourhood']=='Mar Vista'):
        return 51
    elif (row['neighbourhood']=='Hawthorne'):
        return 44
    elif (row['neighbourhood']=='Alhambra'):
        return 81
    elif (row['neighbourhood']=='Redondo Beach'):
        return 0
    elif (row['neighbourhood']=='Silver Lake'):
        return 60
    elif (row['neighbourhood']=='Mid-City'):
        return 60
    elif (row['neighbourhood']=='Brentwood'):
        return 43
    elif (row['neighbourhood']=='Laurel Canyon'):
        return 45
    elif (row['neighbourhood']=='Cahuenga Pass'):
        return 30
    elif (row['neighbourhood']=='Sherman Oaks'):
        return 43
    elif (row['neighbourhood']=='Lomita'):
        return 34
    elif (row['neighbourhood']=='Boyle Heights'):
        return 65
    elif (row['neighbourhood']=='Valley Glen'):
        return 39
    elif (row['neighbourhood']=='South Pasadena'):
        return 0
    elif (row['neighbourhood']=='Inglewood'):
        return 50
    elif (row['neighbourhood']=='Beverly Hills'):
        return 0
    elif (row['neighbourhood']=='Burbank'):
        return 39
    elif (row['neighbourhood']=='Westchester/Playa Del Rey'):
        return 44
    elif (row['neighbourhood']=='Toluca Lake'):
        return 39
    elif (row['neighbourhood']=='Altadena'):
        return 0
    elif (row['neighbourhood']=='Irwindale'):
        return 0
    elif (row['neighbourhood']=='South Robertson'):
        return 60
    elif (row['neighbourhood']=='Bel Air/Beverly Crest'):
        return 49
    elif (row['neighbourhood']=='Westside'):
        return 63
    elif (row['neighbourhood']=='Arts District'):
        return 64
    elif (row['neighbourhood']=='Rosemead'):
        return 40
    elif (row['neighbourhood']=='Pasadena'):
        return 0
    elif (row['neighbourhood']=='Glassell Park'):
        return 44
    elif (row['neighbourhood']=='Whittier'):
        return 0
    elif (row['neighbourhood']=='Montebello'):
        return 0
    elif (row['neighbourhood']=='Atwater Village'):
        return 53
    elif (row['neighbourhood']=='Lynwood'):
        return 0
    elif (row['neighbourhood']=='Mission Hills'):
        return 39
    elif (row['neighbourhood']=='Lenox'):
        return 59
    elif (row['neighbourhood']=='Hermon'):
        return 49
    elif (row['neighbourhood']=='Monterey Park'):
        return 0
    elif (row['neighbourhood']=='San Gabriel'):
        return 37
    elif (row['neighbourhood']=='Montecito Heights'):
        return 42
    elif (row['neighbourhood']=='Temple City'):
        return 40
    elif (row['neighbourhood']=='Canoga Park'):
        return 49
    elif (row['neighbourhood']=='Van Nuys'):
        return 50
    elif (row['neighbourhood']=='Northridge'):
        return 37
    elif (row['neighbourhood']=='Topanga'):
        return 0
    elif (row['neighbourhood']=='West Covina'):
        return 0
    elif (row['neighbourhood']=='Harbor City'):
        return 35
    elif (row['neighbourhood']=='Studio City'):
        return 43
    elif (row['neighbourhood']=='Manhattan Beach'):
        return 70
    elif (row['neighbourhood']=='Reseda'):
        return 43
    elif (row['neighbourhood']=='Mount Washington'):
        return 52
    elif (row['neighbourhood']=='Lincoln Heights'):
        return 59
    elif (row['neighbourhood']=='La Canada Flintridge'):
        return 0
    elif (row['neighbourhood']=='Sunland/Tujunga'):
        return 28
    elif (row['neighbourhood']=='Glendora'):
        return 31
    elif (row['neighbourhood']=='Granada Hills North'):
        return 27
    elif (row['neighbourhood']=='Norwalk'):
        return 43
    elif (row['neighbourhood']=='Paramount'):
        return 0
    elif (row['neighbourhood']=='Rancho Palos Verdes'):
        return 0
    elif (row['neighbourhood']=='Gardena'):
        return 0
    elif (row['neighbourhood']=='Signal Hill'):
        return 0
    elif (row['neighbourhood']=='Carson'):
        return 0
    elif (row['neighbourhood']=='Torrance'):
        return 37
    elif (row['neighbourhood']=='Baldwin Hills'):
        return 58
    elif (row['neighbourhood']=='Pico Rivera'):
        return 0
    elif (row['neighbourhood']=='La Mirada'):
        return 0
    elif (row['neighbourhood']=='Porter Ranch'):
        return 8
    elif (row['neighbourhood']=='El Monte'):
        return 0
    elif (row['neighbourhood']=='Chatsworth'):
        return 36
    elif (row['neighbourhood']=='West Rancho Dominguez'):
        return 0
    elif (row['neighbourhood']=='Elysian Valley'):
        return 50
    elif (row['neighbourhood']=='Azusa'):
        return 54
    elif (row['neighbourhood']=='El Sereno'):
        return 36
    elif (row['neighbourhood']=='Skid Row'):
        return 77
    elif (row['neighbourhood']=='Harbor Gateway'):
        return 53
    elif (row['neighbourhood']=='Cerritos'):
        return 0
    elif (row['neighbourhood']=='East Los Angeles'):
        return 54
    elif (row['neighbourhood']=='South San Gabriel'):
        return 25
    elif (row['neighbourhood']=='Compton'):
        return 0
    elif (row['neighbourhood']=='East San Gabriel'):
        return 0
    elif (row['neighbourhood']=='Sylmar'):
        return 39
    elif (row['neighbourhood']=='Bellflower'):
        return 0
    elif (row['neighbourhood']=='Winnetka'):
        return 40
    elif (row['neighbourhood']=='Lakewood'):
        return 0
    elif (row['neighbourhood']=='Watts'):
        return 56
    elif (row['neighbourhood']=='Baldwin Park'):
        return 0
    elif (row['neighbourhood']=='Panorama City'):
        return 47
    elif (row['neighbourhood']=='Pacoima'):
        return 46
    elif (row['neighbourhood']=='Huntington Park'):
        return 53
    elif (row['neighbourhood']=='Monterey Hills'):
        return 23
    elif (row['neighbourhood']=='Sierra Madre'):
        return 0
    elif (row['neighbourhood']=='Lake Balboa'):
        return 42
    elif (row['neighbourhood']=='Alondra Park'):
        return 0
    elif (row['neighbourhood']=='South El Monte'):
        return 0
    elif (row['neighbourhood']=='Cypress Park'):
        return 60
    elif (row['neighbourhood']=='Westmont'):
        return 55
    elif (row['neighbourhood']=='Duarte'):
        return 0
    elif (row['neighbourhood']=='Palos Verdes'):
        return 0
    elif (row['neighbourhood']=='Downey'):
        return 0
    elif (row['neighbourhood']=='North Hills West'):
        return 36
    elif (row['neighbourhood']=='South Whittier'):
        return 0
    elif (row['neighbourhood']=='Sun Valley'):
        return 44
    elif (row['neighbourhood']=='Rolling Hills Estates'):
        return 0
    elif (row['neighbourhood']=='Florence-Graham'):
        return 60
    elif (row['neighbourhood']=='Wilmington'):
        return 39
    elif (row['neighbourhood']=='Williamsburg'):
        return 93
    elif (row['neighbourhood']=='West Village'):
        return 100
    elif (row['neighbourhood']=='Washington Heights'):
        return 97
    elif (row['neighbourhood']=='Midtown East'):
        return 100
    elif (row['neighbourhood']=="Hell's Kitchen"):
        return 100
    elif (row['neighbourhood']=='Woodside'):
        return 95
    elif (row['neighbourhood']=='Bushwick'):
        return 95
    elif (row['neighbourhood']=='Meatpacking District'):
        return 100
    elif (row['neighbourhood']=='Upper West Side'):
        return 99
    elif (row['neighbourhood']=='East New York'):
        return 90
    elif (row['neighbourhood']=='Ridgewood'):
        return 90
    elif (row['neighbourhood']=='Graniteville'):
        return 60
    elif (row['neighbourhood']=='Alphabet City'):
        return 100
    elif (row['neighbourhood']=='Lower East Side'):
        return 93
    elif (row['neighbourhood']=='Carroll Gardens'):
        return 89
    elif (row['neighbourhood']=='Midtown'):
        return 100
    elif (row['neighbourhood']=='Hamilton Heights'):
        return 100
    elif (row['neighbourhood']=='Greenpoint'):
        return 71
    elif (row['neighbourhood']=='Chelsea'):
        return 100
    elif (row['neighbourhood']=='Upper East Side'):
        return 99
    elif (row['neighbourhood']=='Kensington'):
        return 86
    elif (row['neighbourhood']=='Crown Heights'):
        return 100
    elif (row['neighbourhood']=='Bedford-Stuyvesant'):
        return 96
    elif (row['neighbourhood']=='Coney Island'):
        return 77
    elif (row['neighbourhood']=='Soho'):
        return 100
    elif (row['neighbourhood']=='Rego Park'):
        return 91
    elif (row['neighbourhood']=='Williamsbridge'):
        return 80
    elif (row['neighbourhood']=='Sunnyside'):
        return 61
    elif (row['neighbourhood']=='Harlem'):
        return 100
    elif (row['neighbourhood']=='East Harlem'):
        return 99
    elif (row['neighbourhood']=='Fort Greene'):
        return 100
    elif (row['neighbourhood']=='Lefferts Garden'):
        return 96
    elif (row['neighbourhood']=='Kew Garden Hills'):
        return 68
    elif (row['neighbourhood']=='Long Island City'):
        return 87
    elif (row['neighbourhood']=='Financial District'):
        return 100
    elif (row['neighbourhood']=='Boerum Hill'):
        return 100
    elif (row['neighbourhood']=='Astoria'):
        return 78
    elif (row['neighbourhood']=='Flatbush'):
        return 95
    elif (row['neighbourhood']=='The Rockaways'):
        return 63
    elif (row['neighbourhood']=='East Village'):
        return 96
    elif (row['neighbourhood']=='Battery Park City'):
        return 100
    elif (row['neighbourhood']=='Flushing'):
        return 83
    elif (row['neighbourhood']=='Greenwood Heights'):
        return 90
    elif (row['neighbourhood']=='Gowanus'):
        return 97
    elif (row['neighbourhood']=='Kips Bay'):
        return 100
    elif (row['neighbourhood']=='Jackson Heights'):
        return 86
    elif (row['neighbourhood']=='Times Square/Theatre District'):
        return 100
    elif (row['neighbourhood']=='Roosevelt Island'):
        return 90
    elif (row['neighbourhood']=='Wakefield'):
        return 77
    elif (row['neighbourhood']=='Clinton Hill'):
        return 99
    elif (row['neighbourhood']=='Brooklyn Navy Yard'):
        return 73
    elif (row['neighbourhood']=='Jamaica'):
        return 94
    elif (row['neighbourhood']=='Corona'):
        return 79
    elif (row['neighbourhood']=='Morningside Heights'):
        return 99
    elif (row['neighbourhood']=='Midwood'):
        return 87
    elif (row['neighbourhood']=='Murray Hill'):
        return 100
    elif (row['neighbourhood']=='Maspeth'):
        return 75
    elif (row['neighbourhood']=='DUMBO'):
        return 100
    elif (row['neighbourhood']=='Flatiron District'):
        return 100
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Brooklyn Heights'):
        return 100
    elif (row['neighbourhood']=='Windsor Terrace'):
        return 86
    elif (row['neighbourhood']=='Union Square'):
        return 100
    elif (row['neighbourhood']=='Tompkinsville'):
        return 70
    elif (row['neighbourhood']=='Gramercy Park'):
        return 100
    elif (row['neighbourhood']=='Howard Beach'):
        return 66
    elif (row['neighbourhood']=='Fort Wadsworth'):
        return 64
    elif (row['neighbourhood']=='Highbridge'):
        return 94
    elif (row['neighbourhood']=='New Brighton'):
        return 65
    elif (row['neighbourhood']=='Crotona'):
        return 91
    elif (row['neighbourhood']=='Woodhaven'):
        return 72
    elif (row['neighbourhood']=='Park Slope'):
        return 97
    elif (row['neighbourhood']=='Sunset Park'):
        return 85
    elif (row['neighbourhood']=='Ozone Park'):
        return 77
    elif (row['neighbourhood']=='Greenwich Village'):
        return 100
    elif (row['neighbourhood']=='East Flatbush'):
        return 90
    elif (row['neighbourhood']=='Brighton Beach'):
        return 78
    elif (row['neighbourhood']=='Stapleton'):
        return 66
    elif (row['neighbourhood']=='Bay Ridge'):
        return 84
    elif (row['neighbourhood']=='Sheepshead Bay'):
        return 79
    elif (row['neighbourhood']=='Mott Haven'):
        return 99
    elif (row['neighbourhood']=='Tremont'):
        return 93
    elif (row['neighbourhood']=='Tribeca'):
        return 100
    elif (row['neighbourhood']=='Nolita'):
        return 100
    elif (row['neighbourhood']=='Downtown Brooklyn'):
        return 100
    elif (row['neighbourhood']=='Pelham Bay'):
        return 80
    elif (row['neighbourhood']=='Gravesend'):
        return 83
    elif (row['neighbourhood']=='Prospect Heights'):
        return 100
    elif (row['neighbourhood']=='Inwood'):
        return 88
    elif (row['neighbourhood']=='Bensonhurst'):
        return 77
    elif (row['neighbourhood']=='Elmhurst'):
        return 97
    elif (row['neighbourhood']=='Columbia Street Waterfront'):
        return 63
    elif (row['neighbourhood']=='Marble Hill'):
        return 87
    elif (row['neighbourhood']=='Claremont'):
        return 0
    elif (row['neighbourhood']=='Bath Beach'):
        return 74
    elif (row['neighbourhood']=='Concourse Village'):
        return 98
    elif (row['neighbourhood']=='Morrisania'):
        return 92
    elif (row['neighbourhood']=='Flatlands'):
        return 88
    elif (row['neighbourhood']=='Bronxdale'):
        return 80
    elif (row['neighbourhood']=='Forest Hills'):
        return 91
    elif (row['neighbourhood']=='Riverdale'):
        return 72
    elif (row['neighbourhood']=='Red Hook'):
        return 68
    elif (row['neighbourhood']=='Allerton'):
        return 81
    elif (row['neighbourhood']=='Grymes Hill'):
        return 63
    elif (row['neighbourhood']=='Eastchester'):
        return 73
    elif (row['neighbourhood']=='Cobble Hill'):
        return 99
    elif (row['neighbourhood']=='Hudson Square'):
        return 100
    elif (row['neighbourhood']=='Mount Eden'):
        return 93
    elif (row['neighbourhood']=='Canarsie'):
        return 78
    elif (row['neighbourhood']=='Little Italy'):
        return 100
    elif (row['neighbourhood']=='Civic Center'):
        return 100
    elif (row['neighbourhood']=='Hunts Point'):
        return 82
    elif (row['neighbourhood']=='University Heights'):
        return 98
    elif (row['neighbourhood']=='Soundview'):
        return 77
    elif (row['neighbourhood']=='Concourse'):
        return 93
    elif (row['neighbourhood']=='East Elmhurst'):
        return 74
    elif (row['neighbourhood']=='Bedford Park'):
        return 98
    elif (row['neighbourhood']=='Parkchester'):
        return 86
    elif (row['neighbourhood']=='Hillcrest'):
        return 75
    elif (row['neighbourhood']=='Borough Park'):
        return 79
    elif (row['neighbourhood']=='Mariners Harbor'):
        return 55
    elif (row['neighbourhood']=='Richmond Hill'):
        return 81
    elif (row['neighbourhood']=='Brownsville'):
        return 99
    elif (row['neighbourhood']=='Clifton'):
        return 65
    elif (row['neighbourhood']=='Randall Manor'):
        return 58
    elif (row['neighbourhood']=='Spuyten Duyvil'):
        return 78
    elif (row['neighbourhood']=='West Brighton'):
        return 62
    elif (row['neighbourhood']=='Kingsbridge'):
        return 93
    elif (row['neighbourhood']=='New Springville'):
        return 57
    elif (row['neighbourhood']=='Glendale'):
        return 70
    elif (row['neighbourhood']=='Midland Beach'):
        return 61
    elif (row['neighbourhood']=='Port Morris'):
        return 72
    elif (row['neighbourhood']=='Park Versailles'):
        return 93
    elif (row['neighbourhood']=='St. George'):
        return 74
    elif (row['neighbourhood']=='Ditmars / Steinway'):
        return 72
    elif (row['neighbourhood']=='Baychester'):
        return 76
    elif (row['neighbourhood']=='South Ozone Park'):
        return 69
    elif (row['neighbourhood']=='Fordham'):
        return 100
    elif (row['neighbourhood']=='Middle Village'):
        return 70
    elif (row['neighbourhood']=='Bayside'):
        return 63
    elif (row['neighbourhood']=='Kingsbridge Heights'):
        return 100
    elif (row['neighbourhood']=='City Island'):
        return 37
    elif (row['neighbourhood']=='Todt Hill'):
        return 53
    elif (row['neighbourhood']=='Manhattan Beach'):
        return 70
    elif (row['neighbourhood']=='Norwood'):
        return 95
    elif (row['neighbourhood']=='Rosebank'):
        return 64
    elif (row['neighbourhood']=='Whitestone'):
        return 56
    elif (row['neighbourhood']=='Noho'):
        return 100
    elif (row['neighbourhood']=='Morris Heights'):
        return 94
    elif (row['neighbourhood']=='Throgs Neck'):
        return 53
    elif (row['neighbourhood']=='Grasmere'):
        return 70
    elif (row['neighbourhood']=='Woodlawn'):
        return 73
    elif (row['neighbourhood']=='Eltingville'):
        return 61
    elif (row['neighbourhood']=='Dongan Hills'):
        return 65
    elif (row['neighbourhood']=='College Point'):
        return 58
    elif (row['neighbourhood']=='Utopia'):
        return 70
    elif (row['neighbourhood']=='Melrose'):
        return 100
    elif (row['neighbourhood']=='Brooklyn'):
        return 100
    elif (row['neighbourhood']=='South Street Seaport'):
        return 100
    elif (row['neighbourhood']=='Fresh Meadows'):
        return 68
    elif (row['neighbourhood']=='Van Nest'):
        return 85
    elif (row['neighbourhood']=='Manhattan'):
        return 100
    elif (row['neighbourhood']=='Longwood'):
        return 93
    elif (row['neighbourhood']=='Dyker Heights'):
        return 77
    elif (row['neighbourhood']=='Concord'):
        return 67
    elif (row['neighbourhood']=='Great Kills'):
        return 56
    elif (row['neighbourhood']=='Belmont'):
        return 92
    elif (row['neighbourhood']=='New Dorp'):
        return 65
    elif (row['neighbourhood']=='South Beach'):
        return 64
    elif (row['neighbourhood']=='Port Richmond'):
        return 65
    elif (row['neighbourhood']=='Vinegar Hill'):
        return 99
    elif (row['neighbourhood']=='West Farms'):
        return 90
    elif (row['neighbourhood']=='Lindenwood'):
        return 65
    elif (row['neighbourhood']=='Meiers Corners'):
        return 60
    elif (row['neighbourhood']=='Bergen Beach'):
        return 78
    elif (row['neighbourhood']=='Queens'):
        return 75
    elif (row['neighbourhood']=='Westchester Village'):
        return 81
    elif (row['neighbourhood']=='Sea Gate'):
        return 50
    elif (row['neighbourhood']=='Richmond District'):
        return 79
    elif (row['neighbourhood']=='Glen Park'):
        return 81
    elif (row['neighbourhood']=='Western Addition/NOPA'):
        return 90
    elif (row['neighbourhood']=='Mission District'):
        return 87
    elif (row['neighbourhood']=='Union Square'):
        return 100
    elif (row['neighbourhood']=='Outer Sunset'):
        return 62
    elif (row['neighbourhood']=='Nob Hill'):
        return 100
    elif (row['neighbourhood']=='SoMa'):
        return 100
    elif (row['neighbourhood']=='The Castro'):
        return 95
    elif (row['neighbourhood']=='Haight-Ashbury'):
        return 81
    elif (row['neighbourhood']=='Parkside'):
        return 66
    elif (row['neighbourhood']=='Bernal Heights'):
        return 77
    elif (row['neighbourhood']=='Presidio Heights'):
        return 76
    elif (row['neighbourhood']=='Duboce Triangle'):
        return 99
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Cow Hollow'):
        return 77
    elif (row['neighbourhood']=='Downtown'):
        return 100
    elif (row['neighbourhood']=='Marina'):
        return 77
    elif (row['neighbourhood']=='Cole Valley'):
        return 77
    elif (row['neighbourhood']=='Twin Peaks'):
        return 65
    elif (row['neighbourhood']=='Hayes Valley'):
        return 99
    elif (row['neighbourhood']=='Pacific Heights'):
        return 89
    elif (row['neighbourhood']=='Financial District'):
        return 100
    elif (row['neighbourhood']=='Lower Haight'):
        return 98
    elif (row['neighbourhood']=='Noe Valley'):
        return 74
    elif (row['neighbourhood']=='North Beach'):
        return 96
    elif (row['neighbourhood']=='Sunnyside'):
        return 61
    elif (row['neighbourhood']=='Russian Hill'):
        return 94
    elif (row['neighbourhood']=='Dogpatch'):
        return 74
    elif (row['neighbourhood']=='Tenderloin'):
        return 100
    elif (row['neighbourhood']=='Excelsior'):
        return 78
    elif (row['neighbourhood']=='Potrero Hill'):
        return 74
    elif (row['neighbourhood']=='Ingleside'):
        return 80
    elif (row['neighbourhood']=='Balboa Terrace'):
        return 72
    elif (row['neighbourhood']=='Oceanview'):
        return 82
    elif (row['neighbourhood']=="Fisherman's Wharf"):
        return 89
    elif (row['neighbourhood']=='Lakeshore'):
        return 50
    elif (row['neighbourhood']=='Daly City'):
        return 0
    elif (row['neighbourhood']=='Inner Sunset'):
        return 72
    elif (row['neighbourhood']=='South Beach'):
        return 64
    elif (row['neighbourhood']=='Forest Hill'):
        return 80
    elif (row['neighbourhood']=='Bayview'):
        return 68
    elif (row['neighbourhood']=='Alamo Square'):
        return 90
    elif (row['neighbourhood']=='Portola'):
        return 69
    elif (row['neighbourhood']=='Mission Terrace'):
        return 85
    elif (row['neighbourhood']=='Telegraph Hill'):
        return 56
    elif (row['neighbourhood']=='Visitacion Valley'):
        return 67
    elif (row['neighbourhood']=='Civic Center'):
        return 100
    elif (row['neighbourhood']=='Mission Bay'):
        return 88
    elif (row['neighbourhood']=='West Portal'):
        return 80
    elif (row['neighbourhood']=='Crocker Amazon'):
        return 69
    elif (row['neighbourhood']=='Diamond Heights'):
        return 67
    elif (row['neighbourhood']=='Japantown'):
        return 90
    elif (row['neighbourhood']=='Sea Cliff'):
        return 68


data_train['transitscore']=data_train.apply(lambda row: transitscore(row), axis=1)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for sentence in data_train['name'].values[:2]:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
'''
from nltk.corpus import stopwords   # stopwords to detect language
from nltk import wordpunct_tokenize # function to split up our words

def get_language_likelihood(input_text):
    """Return a dictionary of languages and their likelihood of being the 
    natural language of the input text
    """
 
    input_text = input_text.lower()
    input_words = wordpunct_tokenize(input_text)
 
    language_likelihood = {}
    total_matches = 0
    for language in stopwords._fileids:
        language_likelihood[language] = len(set(input_words) &
                set(stopwords.words(language)))
 
    return language_likelihood
 
def get_language(input_text):
    """Return the most likely language of the given text
    """ 
    likelihoods = get_language_likelihood(input_text)
    return sorted(likelihoods, key=likelihoods.get, reverse=True)[0]
'''
name_f = [r for r in data_train['name']]
description_f = [r for r in data_train['description']]

pscores = [sid.polarity_scores(comment) for comment in name_f]
pscoresd = [sid.polarity_scores(comment) for comment in description_f]

data_train['name_compound'] = [score['compound'] for score in pscores]
data_train['name_negativity'] = [score['neg'] for score in pscores]
data_train['name_neutrality'] = [score['neu'] for score in pscores]
data_train['name_positivity'] = [score['pos'] for score in pscores]

data_train['desc_compound'] = [score['compound'] for score in pscoresd]
data_train['desc_negativity'] = [score['neg'] for score in pscoresd]
data_train['desc_neutrality'] = [score['neu'] for score in pscoresd]
data_train['desc_positivity'] = [score['pos'] for score in pscoresd]


from datetime import datetime
from dateutil.parser import parse
from datetime import timedelta


def data_as_of(row):
    if (row['city']=='Boston'):
        return datetime.strptime('2017-10-06', '%Y-%m-%d').date()
    elif (row['city']=='NYC'):
        return datetime.strptime('2017-10-02', '%Y-%m-%d').date()
    elif (row['city']=='LA'):
        return datetime.strptime('2017-05-02', '%Y-%m-%d').date()   
    elif (row['city']=='SF'):
        return datetime.strptime('2017-10-02', '%Y-%m-%d').date()     
    elif (row['city']=='Chicago'):
        return datetime.strptime('2017-05-10', '%Y-%m-%d').date()    
    elif (row['city']=='DC'):
        return datetime.strptime('2017-05-10', '%Y-%m-%d').date() 

data_train['data_as_of']=data_train.apply(lambda row: data_as_of(row), axis=1)

#data_train['data_as_of']
data_train['first_review'] = pd.to_datetime(data_train['first_review'])
data_train['host_since'] = pd.to_datetime(data_train['host_since'])
data_train['last_review'] = pd.to_datetime(data_train['last_review'])

data_train['first_review'] = data_train['first_review'].apply(lambda x: x.date())
data_train['host_since'] = data_train['host_since'].apply(lambda x: x.date())
data_train['last_review'] = data_train['last_review'].apply(lambda x: x.date())

data_train['DateDiffFirstReview'] = (data_train.data_as_of - data_train.first_review)/ np.timedelta64(1, 'D')
data_train['DateDiffHostSince'] = (data_train.data_as_of - data_train.host_since)/ np.timedelta64(1, 'D')
data_train['DateDiffLastReview'] = (data_train.data_as_of - data_train.last_review)/ np.timedelta64(1, 'D')

data_train['DateDiffFirstReview'].fillna(0, inplace=True)
data_train['DateDiffHostSince'].fillna(0, inplace=True)
data_train['DateDiffLastReview'].fillna(0, inplace=True)

data_train['thumbnail_url'].fillna(0, inplace=True)

def picture(row):
    if (row['thumbnail_url']==0):
        return 0
    else:
        return 1
    
data_train['picture']=data_train.apply(lambda row: picture(row), axis=1)

g = sns.PairGrid(data_train, hue="city", vars=["log_price", "DateDiffFirstReview", "DateDiffHostSince", "DateDiffLastReview", "walkscore", "transitscore"])
g = g.map(plt.scatter)
g = g.add_legend()
data_train['amenities'] = data_train['amenities'].map(
    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")\
                           for amn in amns.split(",")]))

np.concatenate(data_train['amenities'].map(lambda amns: amns.split("|")).values)

amenities = np.unique(np.concatenate(data_train['amenities'].map(lambda amns: amns.split("|")).values))
amenities_matrix = np.array([data_train['amenities'].map(lambda amns: amn in amns).values for amn in amenities])

data_train['amenities'].head()
data_train['amenities'].map(lambda amns: amns.split("|")).head()

np.unique(np.concatenate(data_train['amenities'].map(lambda amns: amns.split("|"))))

amenities = np.unique(np.concatenate(data_train['amenities'].map(lambda amns: amns.split("|"))))[1:]
amenity_arr = np.array([data_train['amenities'].map(lambda amns: amn in amns) for amn in amenities])
amenity_arr

features = data_train[['accommodates', 'host_response_rate','city', 'bathrooms', 'cleaning_fee', 'bedrooms', 'beds', 'log_price', 'number_of_reviews',
                     'review_scores_rating','cancellation_policy', 'property_type', 'room_type', 'bed_type', 'host_identity_verified', 'host_has_profile_pic', 'instant_bookable','picture','jan_occupancy','feb_occupancy','mar_occupancy','apr_occupancy','may_occupancy','jun_occupancy','jul_occupancy','aug_occupancy','sep_occupancy','oct_occupancy','nov_occupancy','dec_occupancy','walkscore','transitscore', 'dist_to_citycenter', 'dist_to_attr1','dist_to_attr2','dist_to_attr3', 'dist_to_attr4','dist_to_attr5','dist_to_attr6', 'name_positivity','desc_positivity','DateDiffFirstReview','DateDiffHostSince','DateDiffLastReview']]
features.head()
for tf_feature in ['host_identity_verified', 'host_has_profile_pic', 'instant_bookable']:
    features[tf_feature] = features[tf_feature].map(lambda s: False if s == "f" else True)

list(features.columns.values)
for categorical_feature in ['cancellation_policy', 'property_type', 'room_type', 'bed_type']:
    features = pd.concat([features, pd.get_dummies(features[categorical_feature])], axis=1)

host_response_r = []
for i in features['host_response_rate']:
    i = str(i)
    i = i.strip('%')
    i = float(i)
    host_response_r.append(i)

features['host_response_rate'] = host_response_r
features.drop(['host_response_r'], axis = 1, inplace = True)
features.head()
features = pd.concat([features, pd.get_dummies(features['city'])], axis=1)
features.drop(['city','cancellation_policy', 'property_type', 'room_type', 'bed_type'], axis = 1, inplace = True)
features.head()
for col in features.columns[features.isnull().any()]:
    print(col)
features.drop([ 'Bath towel', 'Body soap', 'Grab-rails for shower and toilet', 'Hand or paper towel', 'Hand soap', 'Toilet paper', 'Casa particular', 'Hut', 'Island', 'Lighthouse', 'Parking Space', 'Annadale', 'Arboretum', 'Arleta', 'Arrochar', 'Artesia', 'Ashburn', 'Auburn Gresham', 'Austin', 'Barry Farm', 'Benning Heights', 'Berkley', 'Brighton Park', 'Castle Hill ', 'Castleton Corners', 'Chevy Chase, MD', 'Chillum, MD', 'Clearing', 'Co-op City', 'Commerce', 'Coolidge Corner', 'Country Club', 'Covina', 'Dupont Park', 'East Corner', 'Edenwald', 'Edison Park', 'Elm Park', 'Emerson Hill', 'Fort Totten', 'Galewood', 'Garfield Ridge', 'Gateway', 'Gerritsen Beach', 'Grant City', 'Harvard Square', 'Hawaiian Gardens', 'Hermosa', 'Hilcrest Heights/Marlow Heights, MD', 'Huguenot', 'La Habra', 'La Puente', 'Lighthouse HIll', 'Marine Park', 'Mill Basin', 'Montclare', 'Morris Park', 'Mt. Pleasant', 'Mt. Vernon Square', 'Naylor Gardens', 'Near Northeast', 'New Dorp Beach', 'Newton', 'North Hills East', "O'Hare", 'Oakwood', 'Observatory Circle', "Old Soldiers' Home", 'Presidio', 'Printers Row', 'Rolling Hills', 'Roseland', 'Rossville', 'Santa Fe Springs', 'Silver Spring, MD', 'Skyland', 'South Gate', 'Spring Valley', 'St. Elizabeths', 'Suitland-Silver Hill, MD', 'The Bronx', 'Tottenville', 'Vernon', 'Watertown', 'West Athens', 'West Puente Valley', 'Westerleigh', 'Willowbrook', 'Winthrop', 'Woodland'], axis = 1, inplace = True)

for col in features.columns[features.isnull().any()]:
    features[col] = features[col].fillna(features[col].median())

features.sort_index(axis=1, inplace=True)
fig, ax = plt.subplots()
ax.scatter(x = features['log_price'], y = features['accommodates'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Accommodates', fontsize=13)
plt.show()
features = features[features['log_price'] >= 2]
fig, ax = plt.subplots()
ax.scatter(x = features['log_price'], y = features['accommodates'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Accommodates', fontsize=13)
plt.show()
featuresy = features
Target = featuresy['log_price']
X = features
X.drop(['log_price'], axis = 1, inplace = True)
data_test.head()
def lat_citycentertest(row):
    if (row['city']=='Boston'):
        return 42.3601
    elif (row['city']=='NYC'):
        return 40.7128
    elif (row['city']=='LA'):
        return 34.0522
    elif (row['city']=='SF'):
        return 37.7749
    elif (row['city']=='Chicago'):
        return 41.8781
    elif (row['city']=='DC'):
        return 38.9072

   
def long_citycentertest(row):
    if (row['city']=='Boston'):
        return -71.0589
    elif (row['city']=='NYC'):
        return -74.0060
    elif (row['city']=='LA'):
        return -118.2437    
    elif (row['city']=='SF'):
        return -122.4194     
    elif (row['city']=='Chicago'):
        return -87.6298    
    elif (row['city']=='DC'):
        return -77.0369 

###########
def lat_attr1t(row):
    if (row['city']=='Boston'):
        return 42.3602
    elif (row['city']=='NYC'):
        return 40.7589
    elif (row['city']=='LA'):
        return 34.0928
    elif (row['city']=='SF'):
        return 37.8199
    elif (row['city']=='Chicago'):
        return 41.8918
    elif (row['city']=='DC'):
        return 38.8973
   
def long_attr1t(row):
    if (row['city']=='Boston'):
        return -71.0548
    elif (row['city']=='NYC'):
        return -73.9851
    elif (row['city']=='LA'):
        return -118.329    
    elif (row['city']=='SF'):
        return -122.478     
    elif (row['city']=='Chicago'):
        return -87.6052  
    elif (row['city']=='DC'):
        return -77.0063

def lat_attr2t(row):
    if (row['city']=='Boston'):
        return 42.3467
    elif (row['city']=='NYC'):
        return 40.7484
    elif (row['city']=='LA'):
        return 34.0195
    elif (row['city']=='SF'):
        return 37.8087
    elif (row['city']=='Chicago'):
        return 41.8827
    elif (row['city']=='DC'):
        return 38.886
   
def long_attr2t(row):
    if (row['city']=='Boston'):
        return -71.0972
    elif (row['city']=='NYC'):
        return -73.9857
    elif (row['city']=='LA'):
        return -118.491    
    elif (row['city']=='SF'):
        return -122.41     
    elif (row['city']=='Chicago'):
        return -87.6233 
    elif (row['city']=='DC'):
        return -77.0213

def lat_attr3t(row):
    if (row['city']=='Boston'):
        return 42.377
    elif (row['city']=='NYC'):
        return 40.6892
    elif (row['city']=='LA'):
        return 33.8121
    elif (row['city']=='SF'):
        return 37.788
    elif (row['city']=='Chicago'):
        return 41.8789
    elif (row['city']=='DC'):
        return 38.9097
   
def long_attr3t(row):
    if (row['city']=='Boston'):
        return -71.1167
    elif (row['city']=='NYC'):
        return -74.0445
    elif (row['city']=='LA'):
        return -117.919    
    elif (row['city']=='SF'):
        return -122.408     
    elif (row['city']=='Chicago'):
        return -87.6359 
    elif (row['city']=='DC'):
        return -77.0654

##########
def lat_attr4t(row):
    if (row['city']=='Boston'):
        return 42.3601#MIT
    elif (row['city']=='NYC'):
        return 40.7829
    elif (row['city']=='LA'):
        return 34.1362
    elif (row['city']=='SF'):
        return 37.7694
    elif (row['city']=='Chicago'):
        return 41.8676
    elif (row['city']=='DC'):
        return 38.8899
 
def long_attr4t(row):
    if (row['city']=='Boston'):
        return -71.0942
    elif (row['city']=='NYC'):
        return -73.9654
    elif (row['city']=='LA'):
        return -118.3514    
    elif (row['city']=='SF'):
        return -122.4862    
    elif (row['city']=='Chicago'):
        return -87.6140 
    elif (row['city']=='DC'):
        return -77.0091

##########
def lat_attr5t(row):
    if (row['city']=='Boston'):
        return 42.3663#Old North Church
    elif (row['city']=='NYC'):
        return 40.7587#Rockefeller Center
    elif (row['city']=='LA'):
        return 34.0692#Rodeo Drive
    elif (row['city']=='SF'):
        return 37.7599#Mission District
    elif (row['city']=='Chicago'):
        return 41.9484#wrigley Field
    elif (row['city']=='DC'):
        return 39.9288#Columbia Heights
 
def long_attr5t(row):
    if (row['city']=='Boston'):
        return -71.0544
    elif (row['city']=='NYC'):
        return -73.9787
    elif (row['city']=='LA'):
        return -118.4029    
    elif (row['city']=='SF'):
        return -122.4148    
    elif (row['city']=='Chicago'):
        return -87.6553
    elif (row['city']=='DC'):
        return -77.0305

##########
def lat_attr6t(row):
    if (row['city']=='Boston'):
        return 42.3340#City Point
    elif (row['city']=='NYC'):
        return 40.7230#Lower Manhattan
    elif (row['city']=='LA'):
        return 34.0900#WEHO
    elif (row['city']=='SF'):
        return 37.7775#Alamo Square
    elif (row['city']=='Chicago'):
        return 41.9077#old town
    elif (row['city']=='DC'):
        return 39.9096#Logan Circle
 
def long_attr6t(row):
    if (row['city']=='Boston'):
        return -71.0275
    elif (row['city']=='NYC'):
        return -74.0006
    elif (row['city']=='LA'):
        return -118.3617   
    elif (row['city']=='SF'):
        return -122.4333   
    elif (row['city']=='Chicago'):
        return -87.6374
    elif (row['city']=='DC'):
        return -77.0296

##########
'''
def lat_attr7t(row):
    if (row['city']=='Boston'):
        return 42.377
    elif (row['city']=='NYC'):
        return 40.6892
    elif (row['city']=='LA'):
        return 33.8121
    elif (row['city']=='SF'):
        return 37.788
    elif (row['city']=='Chicago'):
        return 41.8789
    elif (row['city']=='DC'):
        return 38.9097
   
def long_attr7t(row):
    if (row['city']=='Boston'):
        return -71.1167
    elif (row['city']=='NYC'):
        return -74.0445
    elif (row['city']=='LA'):
        return -117.919    
    elif (row['city']=='SF'):
        return -122.408     
    elif (row['city']=='Chicago'):
        return -87.6359 
    elif (row['city']=='DC'):
        return -77.0654
'''   

###########


data_test['lat_citycenter']=data_test.apply(lambda row: lat_citycentertest(row), axis=1)
data_test['long_citycenter']=data_test.apply(lambda row: long_citycentertest(row), axis=1)

data_test['lat_attr1']=data_test.apply(lambda row: lat_attr1t(row), axis=1)
data_test['long_attr1']=data_test.apply(lambda row: long_attr1t(row), axis=1)
data_test['lat_attr2']=data_test.apply(lambda row: lat_attr2t(row), axis=1)
data_test['long_attr2']=data_test.apply(lambda row: long_attr2t(row), axis=1)
data_test['lat_attr3']=data_test.apply(lambda row: lat_attr3t(row), axis=1)
data_test['long_attr3']=data_test.apply(lambda row: long_attr3t(row), axis=1)
data_test['lat_attr4']=data_test.apply(lambda row: lat_attr4t(row), axis=1)
data_test['long_attr4']=data_test.apply(lambda row: long_attr4t(row), axis=1)
data_test['lat_attr5']=data_test.apply(lambda row: lat_attr5t(row), axis=1)
data_test['long_attr5']=data_test.apply(lambda row: long_attr5t(row), axis=1)
data_test['lat_attr6']=data_test.apply(lambda row: lat_attr6t(row), axis=1)
data_test['long_attr6']=data_test.apply(lambda row: long_attr6t(row), axis=1)
#data_test['lat_attr7']=data_test.apply(lambda row: lat_attr7t(row), axis=1)
#data_test['long_attr7']=data_test.apply(lambda row: long_attr7t(row), axis=1)

data_test['dist_to_citycenter'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_citycenter'], row['lat_citycenter']), axis=1)

data_test['dist_to_attr1'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr1'], row['lat_attr1']), axis=1)
data_test['dist_to_attr2'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr2'], row['lat_attr2']), axis=1)
data_test['dist_to_attr3'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr3'], row['lat_attr3']), axis=1)
data_test['dist_to_attr4'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr4'], row['lat_attr4']), axis=1)
data_test['dist_to_attr5'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr5'], row['lat_attr5']), axis=1)
data_test['dist_to_attr6'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr6'], row['lat_attr6']), axis=1)
#data_test['dist_to_attr7'] = data_test.apply(lambda row: haversine(row['longitude'], row['latitude'], row['long_attr7'], row['lat_attr7']), axis=1)

def jant_occupancy(row):
    if (row['city']=='Boston'):
        return 0.425806452
    elif (row['city']=='NYC'):
        return 0.490322581
    elif (row['city']=='LA'):
        return 0.591290323
    elif (row['city']=='SF'):
        return 0.591612903
    elif (row['city']=='Chicago'):
        return 0.38
    elif (row['city']=='DC'):
        return 0.47233871
    
def febt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.436785714
    elif (row['city']=='NYC'):
        return 0.5125
    elif (row['city']=='LA'):
        return 0.623214286
    elif (row['city']=='SF'):
        return 0.655
    elif (row['city']=='Chicago'):
        return 0.373928571
    elif (row['city']=='DC'):
        return 0.514375

def mart_occupancy(row):
    if (row['city']=='Boston'):
        return 0.513225806
    elif (row['city']=='NYC'):
        return 0.606129032
    elif (row['city']=='LA'):
        return 0.642580645
    elif (row['city']=='SF'):
        return 0.690967742
    elif (row['city']=='Chicago'):
        return 0.471935484
    elif (row['city']=='DC'):
        return 0.654919355

def aprt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.661333333
    elif (row['city']=='NYC'):
        return 0.737333333
    elif (row['city']=='LA'):
        return 0.635333333
    elif (row['city']=='SF'):
        return 0.724333333
    elif (row['city']=='Chicago'):
        return 0.579333333
    elif (row['city']=='DC'):
        return 0.6975

def mayt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.692258065
    elif (row['city']=='NYC'):
        return 0.785483871
    elif (row['city']=='LA'):
        return 0.618387097
    elif (row['city']=='SF'):
        return 0.789677419
    elif (row['city']=='Chicago'):
        return 0.714516129
    elif (row['city']=='DC'):
        return 0.716612903

def junt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.688333333
    elif (row['city']=='NYC'):
        return 0.747
    elif (row['city']=='LA'):
        return 0.684666667
    elif (row['city']=='SF'):
        return 0.841666667
    elif (row['city']=='Chicago'):
        return 0.708333333
    elif (row['city']=='DC'):
        return 0.736

def jult_occupancy(row):
    if (row['city']=='Boston'):
        return 0.738064516
    elif (row['city']=='NYC'):
        return 0.675806452
    elif (row['city']=='LA'):
        return 0.715483871
    elif (row['city']=='SF'):
        return 0.85483871
    elif (row['city']=='Chicago'):
        return 0.733548387
    elif (row['city']=='DC'):
        return 0.736370968

def augt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.739677419
    elif (row['city']=='NYC'):
        return 0.700322581
    elif (row['city']=='LA'):
        return 0.72516129
    elif (row['city']=='SF'):
        return 0.884516129
    elif (row['city']=='Chicago'):
        return 0.683225806
    elif (row['city']=='DC'):
        return 0.580887097

def sept_occupancy(row):
    if (row['city']=='Boston'):
        return 0.728666667
    elif (row['city']=='NYC'):
        return 0.774
    elif (row['city']=='LA'):
        return 0.574333333
    elif (row['city']=='SF'):
        return 0.843
    elif (row['city']=='Chicago'):
        return 0.702
    elif (row['city']=='DC'):
        return 0.5295
    
def octt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.691290323
    elif (row['city']=='NYC'):
        return 0.758064516
    elif (row['city']=='LA'):
        return 0.579677419
    elif (row['city']=='SF'):
        return 0.797096774
    elif (row['city']=='Chicago'):
        return 0.677419355
    elif (row['city']=='DC'):
        return 0.505322581

def novt_occupancy(row):
    if (row['city']=='Boston'):
        return 0.592333333
    elif (row['city']=='NYC'):
        return 0.634666667
    elif (row['city']=='LA'):
        return 0.559666667
    elif (row['city']=='SF'):
        return 0.644
    elif (row['city']=='Chicago'):
        return 0.543
    elif (row['city']=='DC'):
        return 0.62875

def dect_occupancy(row):
    if (row['city']=='Boston'):
        return 0.44
    elif (row['city']=='NYC'):
        return 0.611290323
    elif (row['city']=='LA'):
        return 0.532580645
    elif (row['city']=='SF'):
        return 0.633870968
    elif (row['city']=='Chicago'):
        return 0.476451613
    elif (row['city']=='DC'):
        return 0.508870968

    
data_test['jan_occupancy']=data_test.apply(lambda row: jant_occupancy(row), axis=1)
data_test['feb_occupancy']=data_test.apply(lambda row: febt_occupancy(row), axis=1)
data_test['mar_occupancy']=data_test.apply(lambda row: mart_occupancy(row), axis=1)
data_test['apr_occupancy']=data_test.apply(lambda row: aprt_occupancy(row), axis=1)
data_test['may_occupancy']=data_test.apply(lambda row: mayt_occupancy(row), axis=1)
data_test['jun_occupancy']=data_test.apply(lambda row: junt_occupancy(row), axis=1)
data_test['jul_occupancy']=data_test.apply(lambda row: jult_occupancy(row), axis=1)
data_test['aug_occupancy']=data_test.apply(lambda row: augt_occupancy(row), axis=1)
data_test['sep_occupancy']=data_test.apply(lambda row: sept_occupancy(row), axis=1)
data_test['oct_occupancy']=data_test.apply(lambda row: octt_occupancy(row), axis=1)
data_test['nov_occupancy']=data_test.apply(lambda row: novt_occupancy(row), axis=1)
data_test['dec_occupancy']=data_test.apply(lambda row: dect_occupancy(row), axis=1)

def walkscoret(row):
    if (row['neighbourhood']=='Back Bay'):
        return 96
    elif (row['neighbourhood']=='Beacon Hill'): 
        return 98
    elif (row['neighbourhood']=='Roslindale'): 
        return 86
    elif (row['neighbourhood']=='East Boston'): 
        return 94
    elif (row['neighbourhood']=='West End'): 
        return 95
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Allston-Brighton'): 
        return 84
    elif (row['neighbourhood']=='South Boston'): 
        return 93
    elif (row['neighbourhood']=='Roxbury'): 
        return 82
    elif (row['neighbourhood']=='North End'): 
        return 98
    elif (row['neighbourhood']=='Charlestown'): 
        return 89
    elif (row['neighbourhood']=='Fenway/Kenmore'): 
        return 94
    elif (row['neighbourhood']=='Revere'): 
        return 63
    elif (row['neighbourhood']=='Dorchester'): 
        return 87
    elif (row['neighbourhood']=='Jamaica Plain'): 
        return 72
    elif (row['neighbourhood']=='Mission Hill'): 
        return 88
    elif (row['neighbourhood']=='Hyde Park'): 
        return 86
    elif (row['neighbourhood']=='South End'): 
        return 96
    elif (row['neighbourhood']=='Leather District'): 
        return 94
    elif (row['neighbourhood']=='Financial District'): 
        return 100
    elif (row['neighbourhood']=='Theater District'): 
        return 99
    elif (row['neighbourhood']=='Mattapan'): 
        return 82
    elif (row['neighbourhood']=='Government Center'): 
        return 97
    elif (row['neighbourhood']=='Downtown'): 
        return 97
    elif (row['neighbourhood']=='Downtown Crossing'): 
        return 99
    elif (row['neighbourhood']=='West Roxbury'): 
        return 78
    elif (row['neighbourhood']=='Somerville'): 
        return 86
    elif (row['neighbourhood']=='Brookline'): 
        return 78
    elif (row['neighbourhood']=='Chelsea'): 
        return 99
    elif (row['neighbourhood']=='Chestnut Hill'): 
        return 59
    elif (row['neighbourhood']=='Cambridge'): 
        return 87
    elif (row['neighbourhood']=='Pilsen'): 
        return 88
    elif (row['neighbourhood']=='North Center'): 
        return 88
    elif (row['neighbourhood']=='Ukrainian Village'): 
        return 94
    elif (row['neighbourhood']=='Wicker Park'): 
        return 94
    elif (row['neighbourhood']=='Irving Park'): 
        return 85
    elif (row['neighbourhood']=='Bronzeville'): 
        return 74
    elif (row['neighbourhood']=='River North'): 
        return 97
    elif (row['neighbourhood']=='Edgewater'): 
        return 89
    elif (row['neighbourhood']=='Lakeview'): 
        return 91
    elif (row['neighbourhood']=='Norwood Park'): 
        return 62
    elif (row['neighbourhood']=='Back of the Yards'): 
        return 76
    elif (row['neighbourhood']=='Uptown'): 
        return 91
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Roscoe Village'):
        return 88
    elif (row['neighbourhood']=='Avondale'): 
        return 84
    elif (row['neighbourhood']=='Portage Park'): 
        return 74
    elif (row['neighbourhood']=='Lincoln Square'): 
        return 82
    elif (row['neighbourhood']=='Humboldt Park'): 
        return 84
    elif (row['neighbourhood']=='Logan Square'): 
        return 88
    elif (row['neighbourhood']=='Hyde Park'): 
        return 86
    elif (row['neighbourhood']=='West Town/Noble Square'): 
        return 96
    elif (row['neighbourhood']=='Bridgeport'): 
        return 81
    elif (row['neighbourhood']=='West Ridge'): 
        return 81
    elif (row['neighbourhood']=='West Town'): 
        return 91
    elif (row['neighbourhood']=='Woodlawn'): 
        return 69
    elif (row['neighbourhood']=='Little Italy/UIC'):
        return 93
    elif (row['neighbourhood']=='Lincoln Park'):
        return 94
    elif (row['neighbourhood']=='Loop'): 
        return 98
    elif (row['neighbourhood']=='Andersonville'): 
        return 95
    elif (row['neighbourhood']=='South Loop/Printers Row'): 
        return 97
    elif (row['neighbourhood']=='Near North Side'): 
        return 96
    elif (row['neighbourhood']=='Old Town'): 
        return 93
    elif (row['neighbourhood']=='Wrigleyville'): 
        return 93
    elif (row['neighbourhood']=='River West'): 
        return 84
    elif (row['neighbourhood']=='Rogers Park'): 
        return 86
    elif (row['neighbourhood']=='Kenwood'): 
        return 79
    elif (row['neighbourhood']=='Little Village'): 
        return 82
    elif (row['neighbourhood']=='Gold Coast'): 
        return 99
    elif (row['neighbourhood']=='West Loop/Greektown'): 
        return 94
    elif (row['neighbourhood']=='Streeterville'): 
        return 97
    elif (row['neighbourhood']=='Archer Heights'): 
        return 74
    elif (row['neighbourhood']=='Near West Side'): 
        return 86
    elif (row['neighbourhood']=='Oakland'): 
        return 55
    elif (row['neighbourhood']=='North Park'): 
        return 73
    elif (row['neighbourhood']=='Boystown'): 
        return 95
    elif (row['neighbourhood']=='Albany Park'): 
        return 87
    elif (row['neighbourhood']=='Garfield Park'): 
        return 84
    elif (row['neighbourhood']=='Grand Crossing'): 
        return 76
    elif (row['neighbourhood']=='Bucktown'): 
        return 91
    elif (row['neighbourhood']=='Pullman'): 
        return 49
    elif (row['neighbourhood']=='Belmont Cragin'): 
        return 82
    elif (row['neighbourhood']=='Jefferson Park'): 
        return 72
    elif (row['neighbourhood']=='South Chicago'): 
        return 70
    elif (row['neighbourhood']=='Armour Square'): 
        return 88
    elif (row['neighbourhood']=='Calumet Heights'): 
        return 63
    elif (row['neighbourhood']=='West Elsdon'): 
        return 70
    elif (row['neighbourhood']=='Dunning'): 
        return 68
    elif (row['neighbourhood']=='West Lawn'): 
        return 70
    elif (row['neighbourhood']=='Beverly'): 
        return 64
    elif (row['neighbourhood']=='Washington Park'): 
        return 79
    elif (row['neighbourhood']=='South Deering'): 
        return 49
    elif (row['neighbourhood']=='South Shore'): 
        return 75
    elif (row['neighbourhood']=='Chatham'): 
        return 82
    elif (row['neighbourhood']=='North Lawndale'): 
        return 72
    elif (row['neighbourhood']=='Englewood'): 
        return 70
    elif (row['neighbourhood']=='Morgan Park'): 
        return 66
    elif (row['neighbourhood']=='Sauganash'): 
        return 55
    elif (row['neighbourhood']=='Magnificent Mile'): 
        return 99
    elif (row['neighbourhood']=='McKinley Park'): 
        return 75
    elif (row['neighbourhood']=='Friendship Heights'): 
        return 92
    elif (row['neighbourhood']=='Kingman Park'): 
        return 82
    elif (row['neighbourhood']=='Southwest Waterfront'): 
        return 81
    elif (row['neighbourhood']=='Capitol Hill'): 
        return 86
    elif (row['neighbourhood']=='Columbia Heights'): 
        return 94
    elif (row['neighbourhood']=='Burleith'): 
        return 85
    elif (row['neighbourhood']=='Adams Morgan'): 
        return 95
    elif (row['neighbourhood']=='Mount Pleasant'): 
        return 90
    elif (row['neighbourhood']=='16th Street Heights'): 
        return 84
    elif (row['neighbourhood']=='Dupont Circle'): 
        return 98
    elif (row['neighbourhood']=='Georgetown'): 
        return 39
    elif (row['neighbourhood']=='Near Northeast/H Street Corridor'): 
        return 95
    elif (row['neighbourhood']=='Carver Langston'): 
        return 82
    elif (row['neighbourhood']=='Downtown/Penn Quarter'): 
        return 98
    elif (row['neighbourhood']=='U Street Corridor'): 
        return 99
    elif (row['neighbourhood']=='Petworth'): 
        return 85
    elif (row['neighbourhood']=='Bloomingdale'): 
        return 92
    elif (row['neighbourhood']=='Takoma Park, MD'): 
        return 81
    elif (row['neighbourhood']=='LeDroit Park'): 
        return 93
    elif (row['neighbourhood']=='Pleasant Hill'): 
        return 67
    elif (row['neighbourhood']=='Logan Circle'): 
        return 98
    elif (row['neighbourhood']=='Eastland Gardens'): 
        return 49
    elif (row['neighbourhood']=='Benning Ridge'): 
        return 58
    elif (row['neighbourhood']=='Mount Vernon Square'): 
        return 97
    elif (row['neighbourhood']=='Bellevue'): 
        return 69
    elif (row['neighbourhood']=='Kalorama'): 
        return 88
    elif (row['neighbourhood']=='Edgewood'): 
        return 79
    elif (row['neighbourhood']=='Barney Circle'): 
        return 69
    elif (row['neighbourhood']=='Eckington'): 
        return 84
    elif (row['neighbourhood']=='Glover Park'): 
        return 78
    elif (row['neighbourhood']=='Brookland'): 
        return 75
    elif (row['neighbourhood']=='Park View'): 
        return 92
    elif (row['neighbourhood']=='Michigan Park'): 
        return 58
    elif (row['neighbourhood']=='Cathedral Heights'): 
        return 78
    elif (row['neighbourhood']=='Shaw'): 
        return 98
    elif (row['neighbourhood']=='Fairlawn'): 
        return 73
    elif (row['neighbourhood']=='Foggy Bottom'): 
        return 91
    elif (row['neighbourhood']=='Washington Highlands'): 
        return 63
    elif (row['neighbourhood']=='Deanwood'): 
        return 51
    elif (row['neighbourhood']=='Cleveland Park'): 
        return 73
    elif (row['neighbourhood']=='Shipley Terrace'): 
        return 53
    elif (row['neighbourhood']=='West End'): 
        return 95
    elif (row['neighbourhood']=='Brentwood'): 
        return 54
    elif (row['neighbourhood']=='Judiciary Square'): 
        return 97
    elif (row['neighbourhood']=='Randle Highlands'): 
        return 69
    elif (row['neighbourhood']=='Chevy Chase'): 
        return 64
    elif (row['neighbourhood']=='Pleasant Plains'): 
        return 58
    elif (row['neighbourhood']=='Trinidad'): 
        return 80
    elif (row['neighbourhood']=='Woodridge'): 
        return 82
    elif (row['neighbourhood']=='Anacostia'): 
        return 64
    elif (row['neighbourhood']=='Palisades'): 
        return 56
    elif (row['neighbourhood']=='Garfield Heights'): 
        return 41
    elif (row['neighbourhood']=='Massachusetts Heights'): 
        return 60
    elif (row['neighbourhood']=='Truxton Circle'): 
        return 92
    elif (row['neighbourhood']=='Navy Yard'): 
        return 88
    elif (row['neighbourhood']=='Brightwood'): 
        return 90
    elif (row['neighbourhood']=='Shepherd Park'): 
        return 85
    elif (row['neighbourhood']=='Buena Vista'): 
        return 92
    elif (row['neighbourhood']=='Manor Park'): 
        return 77
    elif (row['neighbourhood']=='Stronghold'): 
        return 73
    elif (row['neighbourhood']=='American University Park'): 
        return 82
    elif (row['neighbourhood']=='North Cleveland Park'): 
        return 83
    elif (row['neighbourhood']=='Lamond Riggs'): 
        return 75
    elif (row['neighbourhood']=='Fort Lincoln'): 
        return 53
    elif (row['neighbourhood']=='Forest Hills'): 
        return 91
    elif (row['neighbourhood']=='Central Northeast/Mahaning Heights'): 
        return 99
    elif (row['neighbourhood']=='Langdon'): 
        return 74
    elif (row['neighbourhood']=='Good Hope'): 
        return 68
    elif (row['neighbourhood']=='Lincoln Heights'): 
        return 78
    elif (row['neighbourhood']=='Takoma'): 
        return 83
    elif (row['neighbourhood']=='Woodley Park'): 
        return 72
    elif (row['neighbourhood']=='Congress Heights'): 
        return 63
    elif (row['neighbourhood']=='Benning'): 
        return 71
    elif (row['neighbourhood']=='Marshall Heights'): 
        return 67
    elif (row['neighbourhood']=='Kent'): 
        return 47
    elif (row['neighbourhood']=='Colonial Village'): 
        return 44
    elif (row['neighbourhood']=='Fort Davis'): 
        return 53
    elif (row['neighbourhood']=='Ivy City'): 
        return 77
    elif (row['neighbourhood']=='River Terrace'): 
        return 57
    elif (row['neighbourhood']=='Crestwood'): 
        return 55
    elif (row['neighbourhood']=='Greenway'): 
        return 34
    elif (row['neighbourhood']=='Fort Dupont'): 
        return 58
    elif (row['neighbourhood']=='Knox Hill'): 
        return 36
    elif (row['neighbourhood']=='Douglass'): 
        return 67
    elif (row['neighbourhood']=='North Michigan Park'): 
        return 58
    elif (row['neighbourhood']=='Bethesda, MD'): 
        return 46
    elif (row['neighbourhood']=='Hillbrook'): 
        return 39
    elif (row['neighbourhood']=='Twining'): 
        return 71
    elif (row['neighbourhood']=='Gallaudet'): 
        return 83
    elif (row['neighbourhood']=='Foxhall'): 
        return 63
    elif (row['neighbourhood']=='Mt Rainier/Brentwood, MD'): 
        return 80
    elif (row['neighbourhood']=='Wesley Heights'): 
        return 50
    elif (row['neighbourhood']=='Santa Monica'): 
        return 83
    elif (row['neighbourhood']=='Marina Del Rey'): 
        return 64
    elif (row['neighbourhood']=='Palms'): 
        return 87
    elif (row['neighbourhood']=='Westlake'): 
        return 91
    elif (row['neighbourhood']=='Lawndale'): 
        return 72
    elif (row['neighbourhood']=='Mid-Wilshire'): 
        return 96
    elif (row['neighbourhood']=='San Pedro'): 
        return 86
    elif (row['neighbourhood']=='East Hollywood'): 
        return 89
    elif (row['neighbourhood']=='Los Feliz'): 
        return 80
    elif (row['neighbourhood']=='West Los Angeles'): 
        return 87
    elif (row['neighbourhood']=='Hollywood'): 
        return 90
    elif (row['neighbourhood']=='Long Beach'): 
        return 70
    elif (row['neighbourhood']=='Echo Park'): 
        return 85
    elif (row['neighbourhood']=='Venice'): 
        return 82
    elif (row['neighbourhood']=='Culver City'): 
        return 73
    elif (row['neighbourhood']=='Highland Park'): 
        return 75
    elif (row['neighbourhood']=='Woodland Hills/Warner Center'): 
        return 46
    elif (row['neighbourhood']=='El Segundo'): 
        return 69
    elif (row['neighbourhood']=='Tarzana'): 
        return 53
    elif (row['neighbourhood']=='Arcadia'): 
        return 36
    elif (row['neighbourhood']=='La Crescenta-Montrose'): 
        return 52
    elif (row['neighbourhood']=='Monrovia'): 
        return 60
    elif (row['neighbourhood']=='Encino'): 
        return 47
    elif (row['neighbourhood']=='Pacific Palisades'): 
        return 36
    elif (row['neighbourhood']=='Bell'): 
        return 88
    elif (row['neighbourhood']=='Hermosa Beach'): 
        return 84
    elif (row['neighbourhood']=='Valley Village'): 
        return 72
    elif (row['neighbourhood']=='Downtown'): 
        return 97
    elif (row['neighbourhood']=='North Hollywood'): 
        return 83
    elif (row['neighbourhood']=='Del Rey'): 
        return 73
    elif (row['neighbourhood']=='Eagle Rock'): 
        return 70
    elif (row['neighbourhood']=='Malibu'): 
        return 17
    elif (row['neighbourhood']=='Glendale'): 
        return 84
    elif (row['neighbourhood']=='West Adams'): 
        return 72
    elif (row['neighbourhood']=='West Hills'): 
        return 41
    elif (row['neighbourhood']=='South LA'): 
        return 84
    elif (row['neighbourhood']=='Bradbury'): 
        return 28
    elif (row['neighbourhood']=='San Marino'): 
        return 70
    elif (row['neighbourhood']=='Hollywood Hills'): 
        return 76
    elif (row['neighbourhood']=='Westwood'): 
        return 69
    elif (row['neighbourhood']=='West Hollywood'): 
        return 96
    elif (row['neighbourhood']=='Mar Vista'): 
        return 70
    elif (row['neighbourhood']=='Hawthorne'): 
        return 69
    elif (row['neighbourhood']=='Alhambra'): 
        return 81
    elif (row['neighbourhood']=='Redondo Beach'): 
        return 79
    elif (row['neighbourhood']=='Silver Lake'): 
        return 61
    elif (row['neighbourhood']=='Mid-City'): 
        return 72
    elif (row['neighbourhood']=='Brentwood'): 
        return 54
    elif (row['neighbourhood']=='Laurel Canyon'): 
        return 86
    elif (row['neighbourhood']=='Cahuenga Pass'): 
        return 39
    elif (row['neighbourhood']=='Sherman Oaks'): 
        return 62
    elif (row['neighbourhood']=='Lomita'): 
        return 69
    elif (row['neighbourhood']=='Boyle Heights'): 
        return 81
    elif (row['neighbourhood']=='Valley Glen'): 
        return 58
    elif (row['neighbourhood']=='South Pasadena'): 
        return 65
    elif (row['neighbourhood']=='Inglewood'): 
        return 69
    elif (row['neighbourhood']=='Beverly Hills'): 
        return 78
    elif (row['neighbourhood']=='Burbank'): 
        return 69
    elif (row['neighbourhood']=='Westchester/Playa Del Rey'): 
        return 59
    elif (row['neighbourhood']=='Toluca Lake'): 
        return 78
    elif (row['neighbourhood']=='Altadena'): 
        return 82
    elif (row['neighbourhood']=='Irwindale'): 
        return 52
    elif (row['neighbourhood']=='South Robertson'): 
        return 82
    elif (row['neighbourhood']=='Bel Air/Beverly Crest'): 
        return 24
    elif (row['neighbourhood']=='Westside'): 
        return 75
    elif (row['neighbourhood']=='Arts District'): 
        return 87
    elif (row['neighbourhood']=='Rosemead'): 
        return 61
    elif (row['neighbourhood']=='Pasadena'): 
        return 66
    elif (row['neighbourhood']=='Glassell Park'): 
        return 62
    elif (row['neighbourhood']=='Whittier'): 
        return 86
    elif (row['neighbourhood']=='Montebello'): 
        return 64
    elif (row['neighbourhood']=='Atwater Village'): 
        return 74
    elif (row['neighbourhood']=='Lynwood'): 
        return 65
    elif (row['neighbourhood']=='Mission Hills'): 
        return 46
    elif (row['neighbourhood']=='Lenox'): 
        return 78
    elif (row['neighbourhood']=='Hermon'): 
        return 62
    elif (row['neighbourhood']=='Monterey Park'): 
        return 61
    elif (row['neighbourhood']=='San Gabriel'): 
        return 69
    elif (row['neighbourhood']=='Montecito Heights'): 
        return 41
    elif (row['neighbourhood']=='Temple City'): 
        return 55
    elif (row['neighbourhood']=='Canoga Park'): 
        return 66
    elif (row['neighbourhood']=='Van Nuys'): 
        return 70
    elif (row['neighbourhood']=='Northridge'): 
        return 48
    elif (row['neighbourhood']=='Topanga'): 
        return 50
    elif (row['neighbourhood']=='West Covina'): 
        return 45
    elif (row['neighbourhood']=='Harbor City'): 
        return 68
    elif (row['neighbourhood']=='Studio City'): 
        return 63
    elif (row['neighbourhood']=='Manhattan Beach'): 
        return 78
    elif (row['neighbourhood']=='Reseda'): 
        return 60
    elif (row['neighbourhood']=='Mount Washington'): 
        return 51
    elif (row['neighbourhood']=='Lincoln Heights'): 
        return 78
    elif (row['neighbourhood']=='La Canada Flintridge'): 
        return 31
    elif (row['neighbourhood']=='Sunland/Tujunga'): 
        return 56
    elif (row['neighbourhood']=='Glendora'): 
        return 44
    elif (row['neighbourhood']=='Granada Hills North'): 
        return 48
    elif (row['neighbourhood']=='Norwalk'): 
        return 58
    elif (row['neighbourhood']=='Paramount'): 
        return 63
    elif (row['neighbourhood']=='Rancho Palos Verdes'): 
        return 21
    elif (row['neighbourhood']=='Gardena'): 
        return 74
    elif (row['neighbourhood']=='Signal Hill'): 
        return 77
    elif (row['neighbourhood']=='Carson'): 
        return 57
    elif (row['neighbourhood']=='Torrance'): 
        return 64
    elif (row['neighbourhood']=='Baldwin Hills'): 
        return 59
    elif (row['neighbourhood']=='Pico Rivera'): 
        return 56
    elif (row['neighbourhood']=='La Mirada'): 
        return 47
    elif (row['neighbourhood']=='Porter Ranch'): 
        return 22
    elif (row['neighbourhood']=='El Monte'): 
        return 60
    elif (row['neighbourhood']=='Chatsworth'): 
        return 48
    elif (row['neighbourhood']=='West Rancho Dominguez'): 
        return 36
    elif (row['neighbourhood']=='Elysian Valley'): 
        return 42
    elif (row['neighbourhood']=='Azusa'): 
        return 66
    elif (row['neighbourhood']=='El Sereno'): 
        return 39
    elif (row['neighbourhood']=='Skid Row'): 
        return 93
    elif (row['neighbourhood']=='Harbor Gateway'): 
        return 61
    elif (row['neighbourhood']=='Cerritos'): 
        return 54
    elif (row['neighbourhood']=='East Los Angeles'): 
        return 73
    elif (row['neighbourhood']=='South San Gabriel'): 
        return 44
    elif (row['neighbourhood']=='Compton'): 
        return 63
    elif (row['neighbourhood']=='East San Gabriel'): 
        return 79
    elif (row['neighbourhood']=='Sylmar'): 
        return 45
    elif (row['neighbourhood']=='Bellflower'): 
        return 62
    elif (row['neighbourhood']=='Winnetka'): 
        return 57
    elif (row['neighbourhood']=='Lakewood'): 
        return 53
    elif (row['neighbourhood']=='Watts'): 
        return 62
    elif (row['neighbourhood']=='Baldwin Park'): 
        return 86
    elif (row['neighbourhood']=='Panorama City'): 
        return 65
    elif (row['neighbourhood']=='Pacoima'): 
        return 58
    elif (row['neighbourhood']=='Huntington Park'): 
        return 80
    elif (row['neighbourhood']=='Monterey Hills'): 
        return 9
    elif (row['neighbourhood']=='Sierra Madre'): 
        return 81
    elif (row['neighbourhood']=='Lake Balboa'): 
        return 58
    elif (row['neighbourhood']=='Alondra Park'): 
        return 66
    elif (row['neighbourhood']=='South El Monte'): 
        return 64
    elif (row['neighbourhood']=='Cypress Park'): 
        return 73
    elif (row['neighbourhood']=='Westmont'): 
        return 61
    elif (row['neighbourhood']=='Duarte'): 
        return 47
    elif (row['neighbourhood']=='Palos Verdes'): 
        return 21
    elif (row['neighbourhood']=='Downey'): 
        return 59
    elif (row['neighbourhood']=='North Hills West'): 
        return 46
    elif (row['neighbourhood']=='South Whittier'): 
        return 47
    elif (row['neighbourhood']=='Sun Valley'): 
        return 54
    elif (row['neighbourhood']=='Rolling Hills Estates'): 
        return 18
    elif (row['neighbourhood']=='Florence-Graham'): 
        return 73
    elif (row['neighbourhood']=='Wilmington'): 
        return 68
    elif (row['neighbourhood']=='Williamsburg'): 
        return 96
    elif (row['neighbourhood']=='West Village'): 
        return 100
    elif (row['neighbourhood']=='Washington Heights'): 
        return 97
    elif (row['neighbourhood']=='Midtown East'): 
        return 99
    elif (row['neighbourhood']=="Hell's Kitchen"): 
        return 98
    elif (row['neighbourhood']=='Woodside'): 
        return 94
    elif (row['neighbourhood']=='Bushwick'): 
        return 95
    elif (row['neighbourhood']=='Meatpacking District'): 
        return 99
    elif (row['neighbourhood']=='Upper West Side'): 
        return 98
    elif (row['neighbourhood']=='East New York'): 
        return 86
    elif (row['neighbourhood']=='Ridgewood'): 
        return 95
    elif (row['neighbourhood']=='Graniteville'): 
        return 68
    elif (row['neighbourhood']=='Alphabet City'): 
        return 97
    elif (row['neighbourhood']=='Lower East Side'): 
        return 96
    elif (row['neighbourhood']=='Carroll Gardens'): 
        return 97
    elif (row['neighbourhood']=='Midtown'): 
        return 99
    elif (row['neighbourhood']=='Hamilton Heights'): 
        return 98
    elif (row['neighbourhood']=='Greenpoint'): 
        return 96
    elif (row['neighbourhood']=='Chelsea'): 
        return 99
    elif (row['neighbourhood']=='Upper East Side'): 
        return 99
    elif (row['neighbourhood']=='Kensington'): 
        return 95
    elif (row['neighbourhood']=='Crown Heights'): 
        return 95
    elif (row['neighbourhood']=='Bedford-Stuyvesant'): 
        return 94
    elif (row['neighbourhood']=='Coney Island'): 
        return 82
    elif (row['neighbourhood']=='Soho'): 
        return 100
    elif (row['neighbourhood']=='Rego Park'): 
        return 92
    elif (row['neighbourhood']=='Williamsbridge'): 
        return 87
    elif (row['neighbourhood']=='Sunnyside'): 
        return 69
    elif (row['neighbourhood']=='Harlem'): 
        return 98
    elif (row['neighbourhood']=='East Harlem'): 
        return 96
    elif (row['neighbourhood']=='Fort Greene'): 
        return 97
    elif (row['neighbourhood']=='Lefferts Garden'): 
        return 96
    elif (row['neighbourhood']=='Kew Garden Hills'): 
        return 90
    elif (row['neighbourhood']=='Long Island City'): 
        return 95
    elif (row['neighbourhood']=='Financial District'): 
        return 100
    elif (row['neighbourhood']=='Boerum Hill'): 
        return 98
    elif (row['neighbourhood']=='Astoria'): 
        return 92
    elif (row['neighbourhood']=='Flatbush'): 
        return 94
    elif (row['neighbourhood']=='The Rockaways'): 
        return 79
    elif (row['neighbourhood']=='East Village'): 
        return 98
    elif (row['neighbourhood']=='Battery Park City'): 
        return 97
    elif (row['neighbourhood']=='Flushing'): 
        return 89
    elif (row['neighbourhood']=='Greenwood Heights'): 
        return 91
    elif (row['neighbourhood']=='Gowanus'): 
        return 97
    elif (row['neighbourhood']=='Kips Bay'): 
        return 99
    elif (row['neighbourhood']=='Jackson Heights'): 
        return 93
    elif (row['neighbourhood']=='Times Square/Theatre District'): 
        return 99
    elif (row['neighbourhood']=='Roosevelt Island'): 
        return 77
    elif (row['neighbourhood']=='Wakefield'): 
        return 87
    elif (row['neighbourhood']=='Clinton Hill'): 
        return 96
    elif (row['neighbourhood']=='Brooklyn Navy Yard'): 
        return 65
    elif (row['neighbourhood']=='Jamaica'): 
        return 88
    elif (row['neighbourhood']=='Corona'): 
        return 93
    elif (row['neighbourhood']=='Morningside Heights'): 
        return 96
    elif (row['neighbourhood']=='Midwood'): 
        return 91
    elif (row['neighbourhood']=='Murray Hill'): 
        return 99
    elif (row['neighbourhood']=='Maspeth'): 
        return 84
    elif (row['neighbourhood']=='DUMBO'): 
        return 98
    elif (row['neighbourhood']=='Flatiron District'): 
        return 100
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Brooklyn Heights'): 
        return 98
    elif (row['neighbourhood']=='Windsor Terrace'): 
        return 90
    elif (row['neighbourhood']=='Union Square'): 
        return 100
    elif (row['neighbourhood']=='Tompkinsville'): 
        return 78
    elif (row['neighbourhood']=='Gramercy Park'): 
        return 100
    elif (row['neighbourhood']=='Howard Beach'): 
        return 69
    elif (row['neighbourhood']=='Fort Wadsworth'): 
        return 58
    elif (row['neighbourhood']=='Highbridge'): 
        return 93
    elif (row['neighbourhood']=='New Brighton'): 
        return 69
    elif (row['neighbourhood']=='Crotona'): 
        return 91
    elif (row['neighbourhood']=='Woodhaven'): 
        return 88
    elif (row['neighbourhood']=='Park Slope'): 
        return 97
    elif (row['neighbourhood']=='Sunset Park'): 
        return 95
    elif (row['neighbourhood']=='Ozone Park'): 
        return 88
    elif (row['neighbourhood']=='Greenwich Village'): 
        return 100
    elif (row['neighbourhood']=='East Flatbush'): 
        return 91
    elif (row['neighbourhood']=='Brighton Beach'): 
        return 96
    elif (row['neighbourhood']=='Stapleton'): 
        return 81
    elif (row['neighbourhood']=='Bay Ridge'): 
        return 91
    elif (row['neighbourhood']=='Sheepshead Bay'): 
        return 92
    elif (row['neighbourhood']=='Mott Haven'): 
        return 96
    elif (row['neighbourhood']=='Tremont'): 
        return 96
    elif (row['neighbourhood']=='Tribeca'): 
        return 99
    elif (row['neighbourhood']=='Nolita'): 
        return 100
    elif (row['neighbourhood']=='Downtown Brooklyn'): 
        return 97
    elif (row['neighbourhood']=='Pelham Bay'): 
        return 87
    elif (row['neighbourhood']=='Gravesend'): 
        return 91
    elif (row['neighbourhood']=='Prospect Heights'): 
        return 97
    elif (row['neighbourhood']=='Inwood'): 
        return 96
    elif (row['neighbourhood']=='Bensonhurst'): 
        return 93
    elif (row['neighbourhood']=='Elmhurst'): 
        return 95
    elif (row['neighbourhood']=='Columbia Street Waterfront'): 
        return 92
    elif (row['neighbourhood']=='Marble Hill'): 
        return 92
    elif (row['neighbourhood']=='Claremont'): 
        return 0
    elif (row['neighbourhood']=='Bath Beach'): 
        return 88
    elif (row['neighbourhood']=='Concourse Village'): 
        return 96
    elif (row['neighbourhood']=='Morrisania'): 
        return 94
    elif (row['neighbourhood']=='Flatlands'): 
        return 90
    elif (row['neighbourhood']=='Bronxdale'): 
        return 93
    elif (row['neighbourhood']=='Forest Hills'): 
        return 91
    elif (row['neighbourhood']=='Riverdale'): 
        return 80
    elif (row['neighbourhood']=='Red Hook'): 
        return 91
    elif (row['neighbourhood']=='Allerton'): 
        return 90
    elif (row['neighbourhood']=='Grymes Hill'): 
        return 57
    elif (row['neighbourhood']=='Eastchester'): 
        return 77
    elif (row['neighbourhood']=='Cobble Hill'): 
        return 97
    elif (row['neighbourhood']=='Hudson Square'): 
        return 100
    elif (row['neighbourhood']=='Mount Eden'): 
        return 96
    elif (row['neighbourhood']=='Canarsie'): 
        return 84
    elif (row['neighbourhood']=='Little Italy'): 
        return 100
    elif (row['neighbourhood']=='Civic Center'): 
        return 99
    elif (row['neighbourhood']=='Hunts Point'): 
        return 91
    elif (row['neighbourhood']=='University Heights'): 
        return 93
    elif (row['neighbourhood']=='Soundview'): 
        return 84
    elif (row['neighbourhood']=='Concourse'): 
        return 96
    elif (row['neighbourhood']=='East Elmhurst'): 
        return 87
    elif (row['neighbourhood']=='Bedford Park'): 
        return 93
    elif (row['neighbourhood']=='Parkchester'): 
        return 94
    elif (row['neighbourhood']=='Hillcrest'): 
        return 81
    elif (row['neighbourhood']=='Borough Park'): 
        return 95
    elif (row['neighbourhood']=='Mariners Harbor'): 
        return 62
    elif (row['neighbourhood']=='Richmond Hill'): 
        return 90
    elif (row['neighbourhood']=='Brownsville'): 
        return 92
    elif (row['neighbourhood']=='Clifton'): 
        return 71
    elif (row['neighbourhood']=='Randall Manor'): 
        return 77
    elif (row['neighbourhood']=='Spuyten Duyvil'): 
        return 75
    elif (row['neighbourhood']=='West Brighton'): 
        return 79
    elif (row['neighbourhood']=='Kingsbridge'): 
        return 92
    elif (row['neighbourhood']=='New Springville'): 
        return 55
    elif (row['neighbourhood']=='Glendale'): 
        return 84
    elif (row['neighbourhood']=='Midland Beach'): 
        return 73
    elif (row['neighbourhood']=='Port Morris'): 
        return 84
    elif (row['neighbourhood']=='Park Versailles'): 
        return 89
    elif (row['neighbourhood']=='St. George'): 
        return 84
    elif (row['neighbourhood']=='Ditmars / Steinway'): 
        return 96
    elif (row['neighbourhood']=='Baychester'): 
        return 84
    elif (row['neighbourhood']=='South Ozone Park'): 
        return 83
    elif (row['neighbourhood']=='Fordham'): 
        return 98
    elif (row['neighbourhood']=='Middle Village'): 
        return 80
    elif (row['neighbourhood']=='Bayside'): 
        return 73
    elif (row['neighbourhood']=='Kingsbridge Heights'): 
        return 93
    elif (row['neighbourhood']=='City Island'): 
        return 68
    elif (row['neighbourhood']=='Todt Hill'): 
        return 48
    elif (row['neighbourhood']=='Manhattan Beach'): 
        return 78
    elif (row['neighbourhood']=='Norwood'): 
        return 92
    elif (row['neighbourhood']=='Rosebank'): 
        return 76
    elif (row['neighbourhood']=='Whitestone'): 
        return 70
    elif (row['neighbourhood']=='Noho'): 
        return 100
    elif (row['neighbourhood']=='Morris Heights'): 
        return 91
    elif (row['neighbourhood']=='Throgs Neck'): 
        return 69
    elif (row['neighbourhood']=='Grasmere'): 
        return 67
    elif (row['neighbourhood']=='Woodlawn'): 
        return 69
    elif (row['neighbourhood']=='Eltingville'): 
        return 66
    elif (row['neighbourhood']=='Dongan Hills'): 
        return 75
    elif (row['neighbourhood']=='College Point'): 
        return 78
    elif (row['neighbourhood']=='Utopia'): 
        return 88
    elif (row['neighbourhood']=='Melrose'): 
        return 97
    elif (row['neighbourhood']=='Brooklyn'): 
        return 96
    elif (row['neighbourhood']=='South Street Seaport'): 
        return 97
    elif (row['neighbourhood']=='Fresh Meadows'): 
        return 76
    elif (row['neighbourhood']=='Van Nest'): 
        return 91
    elif (row['neighbourhood']=='Manhattan'): 
        return 97
    elif (row['neighbourhood']=='Longwood'): 
        return 96
    elif (row['neighbourhood']=='Dyker Heights'): 
        return 90
    elif (row['neighbourhood']=='Concord'): 
        return 63
    elif (row['neighbourhood']=='Great Kills'): 
        return 66
    elif (row['neighbourhood']=='Belmont'): 
        return 96
    elif (row['neighbourhood']=='New Dorp'): 
        return 77
    elif (row['neighbourhood']=='South Beach'): 
        return 72
    elif (row['neighbourhood']=='Port Richmond'): 
        return 83
    elif (row['neighbourhood']=='Vinegar Hill'): 
        return 94
    elif (row['neighbourhood']=='West Farms'): 
        return 93
    elif (row['neighbourhood']=='Lindenwood'): 
        return 72
    elif (row['neighbourhood']=='Meiers Corners'): 
        return 85
    elif (row['neighbourhood']=='Bergen Beach'): 
        return 75
    elif (row['neighbourhood']=='Queens'): 
        return 73
    elif (row['neighbourhood']=='Westchester Village'): 
        return 94
    elif (row['neighbourhood']=='Sea Gate'): 
        return 50
    elif (row['neighbourhood']=='Richmond District'): 
        return 97
    elif (row['neighbourhood']=='Glen Park'): 
        return 78
    elif (row['neighbourhood']=='Western Addition/NOPA'): 
        return 96
    elif (row['neighbourhood']=='Mission District'): 
        return 97
    elif (row['neighbourhood']=='Union Square'): 
        return 100
    elif (row['neighbourhood']=='Outer Sunset'): 
        return 78
    elif (row['neighbourhood']=='Nob Hill'): 
        return 98
    elif (row['neighbourhood']=='SoMa'): 
        return 95
    elif (row['neighbourhood']=='The Castro'): 
        return 95
    elif (row['neighbourhood']=='Haight-Ashbury'): 
        return 96
    elif (row['neighbourhood']=='Parkside'): 
        return 79
    elif (row['neighbourhood']=='Bernal Heights'): 
        return 88
    elif (row['neighbourhood']=='Presidio Heights'): 
        return 90
    elif (row['neighbourhood']=='Duboce Triangle'): 
        return 98
    elif (row['neighbourhood']=='Chinatown'): 
        return 100
    elif (row['neighbourhood']=='Cow Hollow'): 
        return 93
    elif (row['neighbourhood']=='Downtown'): 
        return 97
    elif (row['neighbourhood']=='Marina'): 
        return 94
    elif (row['neighbourhood']=='Cole Valley'): 
        return 96
    elif (row['neighbourhood']=='Twin Peaks'): 
        return 58
    elif (row['neighbourhood']=='Hayes Valley'): 
        return 97
    elif (row['neighbourhood']=='Pacific Heights'): 
        return 96
    elif (row['neighbourhood']=='Financial District'): 
        return 100
    elif (row['neighbourhood']=='Lower Haight'): 
        return 96
    elif (row['neighbourhood']=='Noe Valley'): 
        return 91
    elif (row['neighbourhood']=='North Beach'): 
        return 99
    elif (row['neighbourhood']=='Sunnyside'): 
        return 69
    elif (row['neighbourhood']=='Russian Hill'): 
        return 96
    elif (row['neighbourhood']=='Dogpatch'): 
        return 91
    elif (row['neighbourhood']=='Tenderloin'): 
        return 99
    elif (row['neighbourhood']=='Excelsior'): 
        return 79
    elif (row['neighbourhood']=='Potrero Hill'): 
        return 87
    elif (row['neighbourhood']=='Ingleside'): 
        return 78
    elif (row['neighbourhood']=='Balboa Terrace'): 
        return 76
    elif (row['neighbourhood']=='Oceanview'): 
        return 70
    elif (row['neighbourhood']=="Fisherman's Wharf"): 
        return 98
    elif (row['neighbourhood']=='Lakeshore'): 
        return 39
    elif (row['neighbourhood']=='Daly City'): 
        return 63
    elif (row['neighbourhood']=='Inner Sunset'): 
        return 94
    elif (row['neighbourhood']=='South Beach'): 
        return 72
    elif (row['neighbourhood']=='Forest Hill'): 
        return 60
    elif (row['neighbourhood']=='Bayview'): 
        return 82
    elif (row['neighbourhood']=='Alamo Square'): 
        return 96
    elif (row['neighbourhood']=='Portola'): 
        return 76
    elif (row['neighbourhood']=='Mission Terrace'): 
        return 80
    elif (row['neighbourhood']=='Telegraph Hill'): 
        return 88
    elif (row['neighbourhood']=='Visitacion Valley'): 
        return 68
    elif (row['neighbourhood']=='Civic Center'): 
        return 99
    elif (row['neighbourhood']=='Mission Bay'): 
        return 89
    elif (row['neighbourhood']=='West Portal'): 
        return 91
    elif (row['neighbourhood']=='Crocker Amazon'): 
        return 71
    elif (row['neighbourhood']=='Diamond Heights'): 
        return 69
    elif (row['neighbourhood']=='Japantown'): 
        return 98
    elif (row['neighbourhood']=='Sea Cliff'): 
        return 81

data_test['walkscore']=data_test.apply(lambda row: walkscoret(row), axis=1)

def transitscoret(row):
    if (row['neighbourhood']=='Back Bay'):
        return 97
    elif (row['neighbourhood']=='Beacon Hill'):
        return 100
    elif (row['neighbourhood']=='Roslindale'):
        return 65
    elif (row['neighbourhood']=='East Boston'):
        return 67
    elif (row['neighbourhood']=='West End'):
        return 100
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Allston-Brighton'):
        return 66
    elif (row['neighbourhood']=='South Boston'):
        return 72
    elif (row['neighbourhood']=='Roxbury'):
        return 73
    elif (row['neighbourhood']=='North End'):
        return 100
    elif (row['neighbourhood']=='Charlestown'):
        return 68
    elif (row['neighbourhood']=='Fenway/Kenmore'):
        return 95
    elif (row['neighbourhood']=='Revere'):
        return 54
    elif (row['neighbourhood']=='Dorchester'):
        return 72
    elif (row['neighbourhood']=='Jamaica Plain'):
        return 80
    elif (row['neighbourhood']=='Mission Hill'):
        return 91
    elif (row['neighbourhood']=='Hyde Park'):
        return 66
    elif (row['neighbourhood']=='South End'):
        return 94
    elif (row['neighbourhood']=='Leather District'):
        return 100
    elif (row['neighbourhood']=='Financial District'):
        return 100
    elif (row['neighbourhood']=='Theater District'):
        return 100
    elif (row['neighbourhood']=='Mattapan'):
        return 69
    elif (row['neighbourhood']=='Government Center'):
        return 100
    elif (row['neighbourhood']=='Downtown'):
        return 100
    elif (row['neighbourhood']=='Downtown Crossing'):
        return 100
    elif (row['neighbourhood']=='West Roxbury'):
        return 44
    elif (row['neighbourhood']=='Somerville'):
        return 63
    elif (row['neighbourhood']=='Brookline'):
        return 68
    elif (row['neighbourhood']=='Chelsea'):
        return 100
    elif (row['neighbourhood']=='Chestnut Hill'):
        return 60
    elif (row['neighbourhood']=='Cambridge'):
        return 72
    elif (row['neighbourhood']=='Pilsen'):
        return 66
    elif (row['neighbourhood']=='North Center'):
        return 66
    elif (row['neighbourhood']=='Ukrainian Village'):
        return 70
    elif (row['neighbourhood']=='Wicker Park'):
        return 75
    elif (row['neighbourhood']=='Irving Park'):
        return 66
    elif (row['neighbourhood']=='Bronzeville'):
        return 67
    elif (row['neighbourhood']=='River North'):
        return 100
    elif (row['neighbourhood']=='Edgewater'):
        return 72
    elif (row['neighbourhood']=='Lakeview'):
        return 79
    elif (row['neighbourhood']=='Norwood Park'):
        return 51
    elif (row['neighbourhood']=='Back of the Yards'):
        return 61
    elif (row['neighbourhood']=='Uptown'):
        return 79
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Roscoe Village'):
        return 66
    elif (row['neighbourhood']=='Avondale'):
        return 68
    elif (row['neighbourhood']=='Portage Park'):
        return 59
    elif (row['neighbourhood']=='Lincoln Square'):
        return 60
    elif (row['neighbourhood']=='Humboldt Park'):
        return 67
    elif (row['neighbourhood']=='Logan Square'):
        return 68
    elif (row['neighbourhood']=='Hyde Park'):
        return 66
    elif (row['neighbourhood']=='West Town/Noble Square'):
        return 75
    elif (row['neighbourhood']=='Bridgeport'):
        return 61
    elif (row['neighbourhood']=='West Ridge'):
        return 56
    elif (row['neighbourhood']=='West Town'):
        return 75
    elif (row['neighbourhood']=='Woodlawn'):
        return 73
    elif (row['neighbourhood']=='Little Italy/UIC'):
        return 75
    elif (row['neighbourhood']=='Lincoln Park'):
        return 79
    elif (row['neighbourhood']=='Loop'):
        return 100
    elif (row['neighbourhood']=='Andersonville'):
        return 68
    elif (row['neighbourhood']=='South Loop/Printers Row'):
        return 100
    elif (row['neighbourhood']=='Near North Side'):
        return 90
    elif (row['neighbourhood']=='Old Town'):
        return 84
    elif (row['neighbourhood']=='Wrigleyville'):
        return 80
    elif (row['neighbourhood']=='River West'):
        return 80
    elif (row['neighbourhood']=='Rogers Park'):
        return 74
    elif (row['neighbourhood']=='Kenwood'):
        return 66
    elif (row['neighbourhood']=='Little Village'):
        return 62
    elif (row['neighbourhood']=='Gold Coast'):
        return 91
    elif (row['neighbourhood']=='West Loop/Greektown'):
        return 95
    elif (row['neighbourhood']=='Streeterville'):
        return 100
    elif (row['neighbourhood']=='Archer Heights'):
        return 57
    elif (row['neighbourhood']=='Near West Side'):
        return 82
    elif (row['neighbourhood']=='Oakland'):
        return 49
    elif (row['neighbourhood']=='North Park'):
        return 54
    elif (row['neighbourhood']=='Boystown'):
        return 84
    elif (row['neighbourhood']=='Albany Park'):
        return 62
    elif (row['neighbourhood']=='Garfield Park'):
        return 74
    elif (row['neighbourhood']=='Grand Crossing'):
        return 66
    elif (row['neighbourhood']=='Bucktown'):
        return 73
    elif (row['neighbourhood']=='Pullman'):
        return 62
    elif (row['neighbourhood']=='Belmont Cragin'):
        return 61
    elif (row['neighbourhood']=='Jefferson Park'):
        return 64
    elif (row['neighbourhood']=='South Chicago'):
        return 63
    elif (row['neighbourhood']=='Armour Square'):
        return 75
    elif (row['neighbourhood']=='Calumet Heights'):
        return 58
    elif (row['neighbourhood']=='West Elsdon'):
        return 62
    elif (row['neighbourhood']=='Dunning'):
        return 53
    elif (row['neighbourhood']=='West Lawn'):
        return 57
    elif (row['neighbourhood']=='Beverly'):
        return 50
    elif (row['neighbourhood']=='Washington Park'):
        return 75
    elif (row['neighbourhood']=='South Deering'):
        return 51
    elif (row['neighbourhood']=='South Shore'):
        return 68
    elif (row['neighbourhood']=='Chatham'):
        return 67
    elif (row['neighbourhood']=='North Lawndale'):
        return 62
    elif (row['neighbourhood']=='Englewood'):
        return 65
    elif (row['neighbourhood']=='Morgan Park'):
        return 50
    elif (row['neighbourhood']=='Sauganash'):
        return 41
    elif (row['neighbourhood']=='Magnificent Mile'):
        return 100
    elif (row['neighbourhood']=='McKinley Park'):
        return 64
    elif (row['neighbourhood']=='Friendship Heights'):
        return 73
    elif (row['neighbourhood']=='Kingman Park'):
        return 65
    elif (row['neighbourhood']=='Southwest Waterfront'):
        return 82
    elif (row['neighbourhood']=='Capitol Hill'):
        return 76
    elif (row['neighbourhood']=='Columbia Heights'):
        return 79
    elif (row['neighbourhood']=='Burleith'):
        return 54
    elif (row['neighbourhood']=='Adams Morgan'):
        return 78
    elif (row['neighbourhood']=='Mount Pleasant'):
        return 76
    elif (row['neighbourhood']=='16th Street Heights'):
        return 64
    elif (row['neighbourhood']=='Dupont Circle'):
        return 87
    elif (row['neighbourhood']=='Georgetown'):
        return 47
    elif (row['neighbourhood']=='Near Northeast/H Street Corridor'):
        return 73
    elif (row['neighbourhood']=='Carver Langston'):
        return 64
    elif (row['neighbourhood']=='Downtown/Penn Quarter'):
        return 100
    elif (row['neighbourhood']=='U Street Corridor'):
        return 77
    elif (row['neighbourhood']=='Petworth'):
        return 69
    elif (row['neighbourhood']=='Bloomingdale'):
        return 76
    elif (row['neighbourhood']=='Takoma Park, MD'):
        return 68
    elif (row['neighbourhood']=='LeDroit Park'):
        return 77
    elif (row['neighbourhood']=='Pleasant Hill'):
        return 75
    elif (row['neighbourhood']=='Logan Circle'):
        return 87
    elif (row['neighbourhood']=='Eastland Gardens'):
        return 66
    elif (row['neighbourhood']=='Benning Ridge'):
        return 64
    elif (row['neighbourhood']=='Mount Vernon Square'):
        return 98
    elif (row['neighbourhood']=='Bellevue'):
        return 57
    elif (row['neighbourhood']=='Kalorama'):
        return 74
    elif (row['neighbourhood']=='Edgewood'):
        return 71
    elif (row['neighbourhood']=='Barney Circle'):
        return 78
    elif (row['neighbourhood']=='Eckington'):
        return 71
    elif (row['neighbourhood']=='Glover Park'):
        return 51
    elif (row['neighbourhood']=='Brookland'):
        return 68
    elif (row['neighbourhood']=='Park View'):
        return 75
    elif (row['neighbourhood']=='Michigan Park'):
        return 64
    elif (row['neighbourhood']=='Cathedral Heights'):
        return 66
    elif (row['neighbourhood']=='Shaw'):
        return 78
    elif (row['neighbourhood']=='Fairlawn'):
        return 63
    elif (row['neighbourhood']=='Foggy Bottom'):
        return 87
    elif (row['neighbourhood']=='Washington Highlands'):
        return 60
    elif (row['neighbourhood']=='Deanwood'):
        return 61
    elif (row['neighbourhood']=='Cleveland Park'):
        return 63
    elif (row['neighbourhood']=='Shipley Terrace'):
        return 69
    elif (row['neighbourhood']=='West End'):
        return 100
    elif (row['neighbourhood']=='Brentwood'):
        return 43
    elif (row['neighbourhood']=='Judiciary Square'):
        return 98
    elif (row['neighbourhood']=='Randle Highlands'):
        return 62
    elif (row['neighbourhood']=='Chevy Chase'):
        return 51
    elif (row['neighbourhood']=='Pleasant Plains'):
        return 51
    elif (row['neighbourhood']=='Trinidad'):
        return 59
    elif (row['neighbourhood']=='Woodridge'):
        return 56
    elif (row['neighbourhood']=='Anacostia'):
        return 67
    elif (row['neighbourhood']=='Palisades'):
        return 40
    elif (row['neighbourhood']=='Garfield Heights'):
        return 60
    elif (row['neighbourhood']=='Massachusetts Heights'):
        return 58
    elif (row['neighbourhood']=='Truxton Circle'):
        return 77
    elif (row['neighbourhood']=='Navy Yard'):
        return 69
    elif (row['neighbourhood']=='Brightwood'):
        return 67
    elif (row['neighbourhood']=='Shepherd Park'):
        return 68
    elif (row['neighbourhood']=='Buena Vista'):
        return 89
    elif (row['neighbourhood']=='Manor Park'):
        return 60
    elif (row['neighbourhood']=='Stronghold'):
        return 66
    elif (row['neighbourhood']=='American University Park'):
        return 65
    elif (row['neighbourhood']=='North Cleveland Park'):
        return 67
    elif (row['neighbourhood']=='Lamond Riggs'):
        return 55
    elif (row['neighbourhood']=='Fort Lincoln'):
        return 43
    elif (row['neighbourhood']=='Forest Hills'):
        return 91
    elif (row['neighbourhood']=='Central Northeast/Mahaning Heights'):
        return 100
    elif (row['neighbourhood']=='Langdon'):
        return 56
    elif (row['neighbourhood']=='Good Hope'):
        return 60
    elif (row['neighbourhood']=='Lincoln Heights'):
        return 59
    elif (row['neighbourhood']=='Takoma'):
        return 78
    elif (row['neighbourhood']=='Woodley Park'):
        return 65
    elif (row['neighbourhood']=='Congress Heights'):
        return 62
    elif (row['neighbourhood']=='Benning'):
        return 70
    elif (row['neighbourhood']=='Marshall Heights'):
        return 63
    elif (row['neighbourhood']=='Kent'):
        return 41
    elif (row['neighbourhood']=='Colonial Village'):
        return 68
    elif (row['neighbourhood']=='Fort Davis'):
        return 46
    elif (row['neighbourhood']=='Ivy City'):
        return 64
    elif (row['neighbourhood']=='River Terrace'):
        return 63
    elif (row['neighbourhood']=='Crestwood'):
        return 55
    elif (row['neighbourhood']=='Greenway'):
        return 54
    elif (row['neighbourhood']=='Fort Dupont'):
        return 56
    elif (row['neighbourhood']=='Knox Hill'):
        return 64
    elif (row['neighbourhood']=='Douglass'):
        return 65
    elif (row['neighbourhood']=='North Michigan Park'):
        return 51
    elif (row['neighbourhood']=='Bethesda, MD'):
        return 45
    elif (row['neighbourhood']=='Hillbrook'):
        return 43
    elif (row['neighbourhood']=='Twining'):
        return 60
    elif (row['neighbourhood']=='Gallaudet'):
        return 62
    elif (row['neighbourhood']=='Foxhall'):
        return 39
    elif (row['neighbourhood']=='Mt Rainier/Brentwood, MD'):
        return 72
    elif (row['neighbourhood']=='Wesley Heights'):
        return 44
    elif (row['neighbourhood']=='Santa Monica'):
        return 63
    elif (row['neighbourhood']=='Marina Del Rey'):
        return 48
    elif (row['neighbourhood']=='Palms'):
        return 61
    elif (row['neighbourhood']=='Westlake'):
        return 83
    elif (row['neighbourhood']=='Lawndale'):
        return 66
    elif (row['neighbourhood']=='Mid-Wilshire'):
        return 79
    elif (row['neighbourhood']=='San Pedro'):
        return 46
    elif (row['neighbourhood']=='East Hollywood'):
        return 66
    elif (row['neighbourhood']=='Los Feliz'):
        return 60
    elif (row['neighbourhood']=='West Los Angeles'):
        return 66
    elif (row['neighbourhood']=='Hollywood'):
        return 62
    elif (row['neighbourhood']=='Long Beach'):
        return 51
    elif (row['neighbourhood']=='Echo Park'):
        return 63
    elif (row['neighbourhood']=='Venice'):
        return 53
    elif (row['neighbourhood']=='Culver City'):
        return 51
    elif (row['neighbourhood']=='Highland Park'):
        return 87
    elif (row['neighbourhood']=='Woodland Hills/Warner Center'):
        return 40
    elif (row['neighbourhood']=='El Segundo'):
        return 0
    elif (row['neighbourhood']=='Tarzana'):
        return 40
    elif (row['neighbourhood']=='Arcadia'):
        return 0
    elif (row['neighbourhood']=='La Crescenta-Montrose'):
        return 0
    elif (row['neighbourhood']=='Monrovia'):
        return 32
    elif (row['neighbourhood']=='Encino'):
        return 35
    elif (row['neighbourhood']=='Pacific Palisades'):
        return 27
    elif (row['neighbourhood']=='Bell'):
        return 0
    elif (row['neighbourhood']=='Hermosa Beach'):
        return 0
    elif (row['neighbourhood']=='Valley Village'):
        return 45
    elif (row['neighbourhood']=='Downtown'):
        return 100
    elif (row['neighbourhood']=='North Hollywood'):
        return 51
    elif (row['neighbourhood']=='Del Rey'):
        return 45
    elif (row['neighbourhood']=='Eagle Rock'):
        return 45
    elif (row['neighbourhood']=='Malibu'):
        return 0
    elif (row['neighbourhood']=='Glendale'):
        return 70
    elif (row['neighbourhood']=='West Adams'):
        return 63
    elif (row['neighbourhood']=='West Hills'):
        return 34
    elif (row['neighbourhood']=='South LA'):
        return 67
    elif (row['neighbourhood']=='Bradbury'):
        return 0
    elif (row['neighbourhood']=='San Marino'):
        return 0
    elif (row['neighbourhood']=='Hollywood Hills'):
        return 53
    elif (row['neighbourhood']=='Westwood'):
        return 66
    elif (row['neighbourhood']=='West Hollywood'):
        return 59
    elif (row['neighbourhood']=='Mar Vista'):
        return 51
    elif (row['neighbourhood']=='Hawthorne'):
        return 44
    elif (row['neighbourhood']=='Alhambra'):
        return 81
    elif (row['neighbourhood']=='Redondo Beach'):
        return 0
    elif (row['neighbourhood']=='Silver Lake'):
        return 60
    elif (row['neighbourhood']=='Mid-City'):
        return 60
    elif (row['neighbourhood']=='Brentwood'):
        return 43
    elif (row['neighbourhood']=='Laurel Canyon'):
        return 45
    elif (row['neighbourhood']=='Cahuenga Pass'):
        return 30
    elif (row['neighbourhood']=='Sherman Oaks'):
        return 43
    elif (row['neighbourhood']=='Lomita'):
        return 34
    elif (row['neighbourhood']=='Boyle Heights'):
        return 65
    elif (row['neighbourhood']=='Valley Glen'):
        return 39
    elif (row['neighbourhood']=='South Pasadena'):
        return 0
    elif (row['neighbourhood']=='Inglewood'):
        return 50
    elif (row['neighbourhood']=='Beverly Hills'):
        return 0
    elif (row['neighbourhood']=='Burbank'):
        return 39
    elif (row['neighbourhood']=='Westchester/Playa Del Rey'):
        return 44
    elif (row['neighbourhood']=='Toluca Lake'):
        return 39
    elif (row['neighbourhood']=='Altadena'):
        return 0
    elif (row['neighbourhood']=='Irwindale'):
        return 0
    elif (row['neighbourhood']=='South Robertson'):
        return 60
    elif (row['neighbourhood']=='Bel Air/Beverly Crest'):
        return 49
    elif (row['neighbourhood']=='Westside'):
        return 63
    elif (row['neighbourhood']=='Arts District'):
        return 64
    elif (row['neighbourhood']=='Rosemead'):
        return 40
    elif (row['neighbourhood']=='Pasadena'):
        return 0
    elif (row['neighbourhood']=='Glassell Park'):
        return 44
    elif (row['neighbourhood']=='Whittier'):
        return 0
    elif (row['neighbourhood']=='Montebello'):
        return 0
    elif (row['neighbourhood']=='Atwater Village'):
        return 53
    elif (row['neighbourhood']=='Lynwood'):
        return 0
    elif (row['neighbourhood']=='Mission Hills'):
        return 39
    elif (row['neighbourhood']=='Lenox'):
        return 59
    elif (row['neighbourhood']=='Hermon'):
        return 49
    elif (row['neighbourhood']=='Monterey Park'):
        return 0
    elif (row['neighbourhood']=='San Gabriel'):
        return 37
    elif (row['neighbourhood']=='Montecito Heights'):
        return 42
    elif (row['neighbourhood']=='Temple City'):
        return 40
    elif (row['neighbourhood']=='Canoga Park'):
        return 49
    elif (row['neighbourhood']=='Van Nuys'):
        return 50
    elif (row['neighbourhood']=='Northridge'):
        return 37
    elif (row['neighbourhood']=='Topanga'):
        return 0
    elif (row['neighbourhood']=='West Covina'):
        return 0
    elif (row['neighbourhood']=='Harbor City'):
        return 35
    elif (row['neighbourhood']=='Studio City'):
        return 43
    elif (row['neighbourhood']=='Manhattan Beach'):
        return 70
    elif (row['neighbourhood']=='Reseda'):
        return 43
    elif (row['neighbourhood']=='Mount Washington'):
        return 52
    elif (row['neighbourhood']=='Lincoln Heights'):
        return 59
    elif (row['neighbourhood']=='La Canada Flintridge'):
        return 0
    elif (row['neighbourhood']=='Sunland/Tujunga'):
        return 28
    elif (row['neighbourhood']=='Glendora'):
        return 31
    elif (row['neighbourhood']=='Granada Hills North'):
        return 27
    elif (row['neighbourhood']=='Norwalk'):
        return 43
    elif (row['neighbourhood']=='Paramount'):
        return 0
    elif (row['neighbourhood']=='Rancho Palos Verdes'):
        return 0
    elif (row['neighbourhood']=='Gardena'):
        return 0
    elif (row['neighbourhood']=='Signal Hill'):
        return 0
    elif (row['neighbourhood']=='Carson'):
        return 0
    elif (row['neighbourhood']=='Torrance'):
        return 37
    elif (row['neighbourhood']=='Baldwin Hills'):
        return 58
    elif (row['neighbourhood']=='Pico Rivera'):
        return 0
    elif (row['neighbourhood']=='La Mirada'):
        return 0
    elif (row['neighbourhood']=='Porter Ranch'):
        return 8
    elif (row['neighbourhood']=='El Monte'):
        return 0
    elif (row['neighbourhood']=='Chatsworth'):
        return 36
    elif (row['neighbourhood']=='West Rancho Dominguez'):
        return 0
    elif (row['neighbourhood']=='Elysian Valley'):
        return 50
    elif (row['neighbourhood']=='Azusa'):
        return 54
    elif (row['neighbourhood']=='El Sereno'):
        return 36
    elif (row['neighbourhood']=='Skid Row'):
        return 77
    elif (row['neighbourhood']=='Harbor Gateway'):
        return 53
    elif (row['neighbourhood']=='Cerritos'):
        return 0
    elif (row['neighbourhood']=='East Los Angeles'):
        return 54
    elif (row['neighbourhood']=='South San Gabriel'):
        return 25
    elif (row['neighbourhood']=='Compton'):
        return 0
    elif (row['neighbourhood']=='East San Gabriel'):
        return 0
    elif (row['neighbourhood']=='Sylmar'):
        return 39
    elif (row['neighbourhood']=='Bellflower'):
        return 0
    elif (row['neighbourhood']=='Winnetka'):
        return 40
    elif (row['neighbourhood']=='Lakewood'):
        return 0
    elif (row['neighbourhood']=='Watts'):
        return 56
    elif (row['neighbourhood']=='Baldwin Park'):
        return 0
    elif (row['neighbourhood']=='Panorama City'):
        return 47
    elif (row['neighbourhood']=='Pacoima'):
        return 46
    elif (row['neighbourhood']=='Huntington Park'):
        return 53
    elif (row['neighbourhood']=='Monterey Hills'):
        return 23
    elif (row['neighbourhood']=='Sierra Madre'):
        return 0
    elif (row['neighbourhood']=='Lake Balboa'):
        return 42
    elif (row['neighbourhood']=='Alondra Park'):
        return 0
    elif (row['neighbourhood']=='South El Monte'):
        return 0
    elif (row['neighbourhood']=='Cypress Park'):
        return 60
    elif (row['neighbourhood']=='Westmont'):
        return 55
    elif (row['neighbourhood']=='Duarte'):
        return 0
    elif (row['neighbourhood']=='Palos Verdes'):
        return 0
    elif (row['neighbourhood']=='Downey'):
        return 0
    elif (row['neighbourhood']=='North Hills West'):
        return 36
    elif (row['neighbourhood']=='South Whittier'):
        return 0
    elif (row['neighbourhood']=='Sun Valley'):
        return 44
    elif (row['neighbourhood']=='Rolling Hills Estates'):
        return 0
    elif (row['neighbourhood']=='Florence-Graham'):
        return 60
    elif (row['neighbourhood']=='Wilmington'):
        return 39
    elif (row['neighbourhood']=='Williamsburg'):
        return 93
    elif (row['neighbourhood']=='West Village'):
        return 100
    elif (row['neighbourhood']=='Washington Heights'):
        return 97
    elif (row['neighbourhood']=='Midtown East'):
        return 100
    elif (row['neighbourhood']=="Hell's Kitchen"):
        return 100
    elif (row['neighbourhood']=='Woodside'):
        return 95
    elif (row['neighbourhood']=='Bushwick'):
        return 95
    elif (row['neighbourhood']=='Meatpacking District'):
        return 100
    elif (row['neighbourhood']=='Upper West Side'):
        return 99
    elif (row['neighbourhood']=='East New York'):
        return 90
    elif (row['neighbourhood']=='Ridgewood'):
        return 90
    elif (row['neighbourhood']=='Graniteville'):
        return 60
    elif (row['neighbourhood']=='Alphabet City'):
        return 100
    elif (row['neighbourhood']=='Lower East Side'):
        return 93
    elif (row['neighbourhood']=='Carroll Gardens'):
        return 89
    elif (row['neighbourhood']=='Midtown'):
        return 100
    elif (row['neighbourhood']=='Hamilton Heights'):
        return 100
    elif (row['neighbourhood']=='Greenpoint'):
        return 71
    elif (row['neighbourhood']=='Chelsea'):
        return 100
    elif (row['neighbourhood']=='Upper East Side'):
        return 99
    elif (row['neighbourhood']=='Kensington'):
        return 86
    elif (row['neighbourhood']=='Crown Heights'):
        return 100
    elif (row['neighbourhood']=='Bedford-Stuyvesant'):
        return 96
    elif (row['neighbourhood']=='Coney Island'):
        return 77
    elif (row['neighbourhood']=='Soho'):
        return 100
    elif (row['neighbourhood']=='Rego Park'):
        return 91
    elif (row['neighbourhood']=='Williamsbridge'):
        return 80
    elif (row['neighbourhood']=='Sunnyside'):
        return 61
    elif (row['neighbourhood']=='Harlem'):
        return 100
    elif (row['neighbourhood']=='East Harlem'):
        return 99
    elif (row['neighbourhood']=='Fort Greene'):
        return 100
    elif (row['neighbourhood']=='Lefferts Garden'):
        return 96
    elif (row['neighbourhood']=='Kew Garden Hills'):
        return 68
    elif (row['neighbourhood']=='Long Island City'):
        return 87
    elif (row['neighbourhood']=='Financial District'):
        return 100
    elif (row['neighbourhood']=='Boerum Hill'):
        return 100
    elif (row['neighbourhood']=='Astoria'):
        return 78
    elif (row['neighbourhood']=='Flatbush'):
        return 95
    elif (row['neighbourhood']=='The Rockaways'):
        return 63
    elif (row['neighbourhood']=='East Village'):
        return 96
    elif (row['neighbourhood']=='Battery Park City'):
        return 100
    elif (row['neighbourhood']=='Flushing'):
        return 83
    elif (row['neighbourhood']=='Greenwood Heights'):
        return 90
    elif (row['neighbourhood']=='Gowanus'):
        return 97
    elif (row['neighbourhood']=='Kips Bay'):
        return 100
    elif (row['neighbourhood']=='Jackson Heights'):
        return 86
    elif (row['neighbourhood']=='Times Square/Theatre District'):
        return 100
    elif (row['neighbourhood']=='Roosevelt Island'):
        return 90
    elif (row['neighbourhood']=='Wakefield'):
        return 77
    elif (row['neighbourhood']=='Clinton Hill'):
        return 99
    elif (row['neighbourhood']=='Brooklyn Navy Yard'):
        return 73
    elif (row['neighbourhood']=='Jamaica'):
        return 94
    elif (row['neighbourhood']=='Corona'):
        return 79
    elif (row['neighbourhood']=='Morningside Heights'):
        return 99
    elif (row['neighbourhood']=='Midwood'):
        return 87
    elif (row['neighbourhood']=='Murray Hill'):
        return 100
    elif (row['neighbourhood']=='Maspeth'):
        return 75
    elif (row['neighbourhood']=='DUMBO'):
        return 100
    elif (row['neighbourhood']=='Flatiron District'):
        return 100
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Brooklyn Heights'):
        return 100
    elif (row['neighbourhood']=='Windsor Terrace'):
        return 86
    elif (row['neighbourhood']=='Union Square'):
        return 100
    elif (row['neighbourhood']=='Tompkinsville'):
        return 70
    elif (row['neighbourhood']=='Gramercy Park'):
        return 100
    elif (row['neighbourhood']=='Howard Beach'):
        return 66
    elif (row['neighbourhood']=='Fort Wadsworth'):
        return 64
    elif (row['neighbourhood']=='Highbridge'):
        return 94
    elif (row['neighbourhood']=='New Brighton'):
        return 65
    elif (row['neighbourhood']=='Crotona'):
        return 91
    elif (row['neighbourhood']=='Woodhaven'):
        return 72
    elif (row['neighbourhood']=='Park Slope'):
        return 97
    elif (row['neighbourhood']=='Sunset Park'):
        return 85
    elif (row['neighbourhood']=='Ozone Park'):
        return 77
    elif (row['neighbourhood']=='Greenwich Village'):
        return 100
    elif (row['neighbourhood']=='East Flatbush'):
        return 90
    elif (row['neighbourhood']=='Brighton Beach'):
        return 78
    elif (row['neighbourhood']=='Stapleton'):
        return 66
    elif (row['neighbourhood']=='Bay Ridge'):
        return 84
    elif (row['neighbourhood']=='Sheepshead Bay'):
        return 79
    elif (row['neighbourhood']=='Mott Haven'):
        return 99
    elif (row['neighbourhood']=='Tremont'):
        return 93
    elif (row['neighbourhood']=='Tribeca'):
        return 100
    elif (row['neighbourhood']=='Nolita'):
        return 100
    elif (row['neighbourhood']=='Downtown Brooklyn'):
        return 100
    elif (row['neighbourhood']=='Pelham Bay'):
        return 80
    elif (row['neighbourhood']=='Gravesend'):
        return 83
    elif (row['neighbourhood']=='Prospect Heights'):
        return 100
    elif (row['neighbourhood']=='Inwood'):
        return 88
    elif (row['neighbourhood']=='Bensonhurst'):
        return 77
    elif (row['neighbourhood']=='Elmhurst'):
        return 97
    elif (row['neighbourhood']=='Columbia Street Waterfront'):
        return 63
    elif (row['neighbourhood']=='Marble Hill'):
        return 87
    elif (row['neighbourhood']=='Claremont'):
        return 0
    elif (row['neighbourhood']=='Bath Beach'):
        return 74
    elif (row['neighbourhood']=='Concourse Village'):
        return 98
    elif (row['neighbourhood']=='Morrisania'):
        return 92
    elif (row['neighbourhood']=='Flatlands'):
        return 88
    elif (row['neighbourhood']=='Bronxdale'):
        return 80
    elif (row['neighbourhood']=='Forest Hills'):
        return 91
    elif (row['neighbourhood']=='Riverdale'):
        return 72
    elif (row['neighbourhood']=='Red Hook'):
        return 68
    elif (row['neighbourhood']=='Allerton'):
        return 81
    elif (row['neighbourhood']=='Grymes Hill'):
        return 63
    elif (row['neighbourhood']=='Eastchester'):
        return 73
    elif (row['neighbourhood']=='Cobble Hill'):
        return 99
    elif (row['neighbourhood']=='Hudson Square'):
        return 100
    elif (row['neighbourhood']=='Mount Eden'):
        return 93
    elif (row['neighbourhood']=='Canarsie'):
        return 78
    elif (row['neighbourhood']=='Little Italy'):
        return 100
    elif (row['neighbourhood']=='Civic Center'):
        return 100
    elif (row['neighbourhood']=='Hunts Point'):
        return 82
    elif (row['neighbourhood']=='University Heights'):
        return 98
    elif (row['neighbourhood']=='Soundview'):
        return 77
    elif (row['neighbourhood']=='Concourse'):
        return 93
    elif (row['neighbourhood']=='East Elmhurst'):
        return 74
    elif (row['neighbourhood']=='Bedford Park'):
        return 98
    elif (row['neighbourhood']=='Parkchester'):
        return 86
    elif (row['neighbourhood']=='Hillcrest'):
        return 75
    elif (row['neighbourhood']=='Borough Park'):
        return 79
    elif (row['neighbourhood']=='Mariners Harbor'):
        return 55
    elif (row['neighbourhood']=='Richmond Hill'):
        return 81
    elif (row['neighbourhood']=='Brownsville'):
        return 99
    elif (row['neighbourhood']=='Clifton'):
        return 65
    elif (row['neighbourhood']=='Randall Manor'):
        return 58
    elif (row['neighbourhood']=='Spuyten Duyvil'):
        return 78
    elif (row['neighbourhood']=='West Brighton'):
        return 62
    elif (row['neighbourhood']=='Kingsbridge'):
        return 93
    elif (row['neighbourhood']=='New Springville'):
        return 57
    elif (row['neighbourhood']=='Glendale'):
        return 70
    elif (row['neighbourhood']=='Midland Beach'):
        return 61
    elif (row['neighbourhood']=='Port Morris'):
        return 72
    elif (row['neighbourhood']=='Park Versailles'):
        return 93
    elif (row['neighbourhood']=='St. George'):
        return 74
    elif (row['neighbourhood']=='Ditmars / Steinway'):
        return 72
    elif (row['neighbourhood']=='Baychester'):
        return 76
    elif (row['neighbourhood']=='South Ozone Park'):
        return 69
    elif (row['neighbourhood']=='Fordham'):
        return 100
    elif (row['neighbourhood']=='Middle Village'):
        return 70
    elif (row['neighbourhood']=='Bayside'):
        return 63
    elif (row['neighbourhood']=='Kingsbridge Heights'):
        return 100
    elif (row['neighbourhood']=='City Island'):
        return 37
    elif (row['neighbourhood']=='Todt Hill'):
        return 53
    elif (row['neighbourhood']=='Manhattan Beach'):
        return 70
    elif (row['neighbourhood']=='Norwood'):
        return 95
    elif (row['neighbourhood']=='Rosebank'):
        return 64
    elif (row['neighbourhood']=='Whitestone'):
        return 56
    elif (row['neighbourhood']=='Noho'):
        return 100
    elif (row['neighbourhood']=='Morris Heights'):
        return 94
    elif (row['neighbourhood']=='Throgs Neck'):
        return 53
    elif (row['neighbourhood']=='Grasmere'):
        return 70
    elif (row['neighbourhood']=='Woodlawn'):
        return 73
    elif (row['neighbourhood']=='Eltingville'):
        return 61
    elif (row['neighbourhood']=='Dongan Hills'):
        return 65
    elif (row['neighbourhood']=='College Point'):
        return 58
    elif (row['neighbourhood']=='Utopia'):
        return 70
    elif (row['neighbourhood']=='Melrose'):
        return 100
    elif (row['neighbourhood']=='Brooklyn'):
        return 100
    elif (row['neighbourhood']=='South Street Seaport'):
        return 100
    elif (row['neighbourhood']=='Fresh Meadows'):
        return 68
    elif (row['neighbourhood']=='Van Nest'):
        return 85
    elif (row['neighbourhood']=='Manhattan'):
        return 100
    elif (row['neighbourhood']=='Longwood'):
        return 93
    elif (row['neighbourhood']=='Dyker Heights'):
        return 77
    elif (row['neighbourhood']=='Concord'):
        return 67
    elif (row['neighbourhood']=='Great Kills'):
        return 56
    elif (row['neighbourhood']=='Belmont'):
        return 92
    elif (row['neighbourhood']=='New Dorp'):
        return 65
    elif (row['neighbourhood']=='South Beach'):
        return 64
    elif (row['neighbourhood']=='Port Richmond'):
        return 65
    elif (row['neighbourhood']=='Vinegar Hill'):
        return 99
    elif (row['neighbourhood']=='West Farms'):
        return 90
    elif (row['neighbourhood']=='Lindenwood'):
        return 65
    elif (row['neighbourhood']=='Meiers Corners'):
        return 60
    elif (row['neighbourhood']=='Bergen Beach'):
        return 78
    elif (row['neighbourhood']=='Queens'):
        return 75
    elif (row['neighbourhood']=='Westchester Village'):
        return 81
    elif (row['neighbourhood']=='Sea Gate'):
        return 50
    elif (row['neighbourhood']=='Richmond District'):
        return 79
    elif (row['neighbourhood']=='Glen Park'):
        return 81
    elif (row['neighbourhood']=='Western Addition/NOPA'):
        return 90
    elif (row['neighbourhood']=='Mission District'):
        return 87
    elif (row['neighbourhood']=='Union Square'):
        return 100
    elif (row['neighbourhood']=='Outer Sunset'):
        return 62
    elif (row['neighbourhood']=='Nob Hill'):
        return 100
    elif (row['neighbourhood']=='SoMa'):
        return 100
    elif (row['neighbourhood']=='The Castro'):
        return 95
    elif (row['neighbourhood']=='Haight-Ashbury'):
        return 81
    elif (row['neighbourhood']=='Parkside'):
        return 66
    elif (row['neighbourhood']=='Bernal Heights'):
        return 77
    elif (row['neighbourhood']=='Presidio Heights'):
        return 76
    elif (row['neighbourhood']=='Duboce Triangle'):
        return 99
    elif (row['neighbourhood']=='Chinatown'):
        return 100
    elif (row['neighbourhood']=='Cow Hollow'):
        return 77
    elif (row['neighbourhood']=='Downtown'):
        return 100
    elif (row['neighbourhood']=='Marina'):
        return 77
    elif (row['neighbourhood']=='Cole Valley'):
        return 77
    elif (row['neighbourhood']=='Twin Peaks'):
        return 65
    elif (row['neighbourhood']=='Hayes Valley'):
        return 99
    elif (row['neighbourhood']=='Pacific Heights'):
        return 89
    elif (row['neighbourhood']=='Financial District'):
        return 100
    elif (row['neighbourhood']=='Lower Haight'):
        return 98
    elif (row['neighbourhood']=='Noe Valley'):
        return 74
    elif (row['neighbourhood']=='North Beach'):
        return 96
    elif (row['neighbourhood']=='Sunnyside'):
        return 61
    elif (row['neighbourhood']=='Russian Hill'):
        return 94
    elif (row['neighbourhood']=='Dogpatch'):
        return 74
    elif (row['neighbourhood']=='Tenderloin'):
        return 100
    elif (row['neighbourhood']=='Excelsior'):
        return 78
    elif (row['neighbourhood']=='Potrero Hill'):
        return 74
    elif (row['neighbourhood']=='Ingleside'):
        return 80
    elif (row['neighbourhood']=='Balboa Terrace'):
        return 72
    elif (row['neighbourhood']=='Oceanview'):
        return 82
    elif (row['neighbourhood']=="Fisherman's Wharf"):
        return 89
    elif (row['neighbourhood']=='Lakeshore'):
        return 50
    elif (row['neighbourhood']=='Daly City'):
        return 0
    elif (row['neighbourhood']=='Inner Sunset'):
        return 72
    elif (row['neighbourhood']=='South Beach'):
        return 64
    elif (row['neighbourhood']=='Forest Hill'):
        return 80
    elif (row['neighbourhood']=='Bayview'):
        return 68
    elif (row['neighbourhood']=='Alamo Square'):
        return 90
    elif (row['neighbourhood']=='Portola'):
        return 69
    elif (row['neighbourhood']=='Mission Terrace'):
        return 85
    elif (row['neighbourhood']=='Telegraph Hill'):
        return 56
    elif (row['neighbourhood']=='Visitacion Valley'):
        return 67
    elif (row['neighbourhood']=='Civic Center'):
        return 100
    elif (row['neighbourhood']=='Mission Bay'):
        return 88
    elif (row['neighbourhood']=='West Portal'):
        return 80
    elif (row['neighbourhood']=='Crocker Amazon'):
        return 69
    elif (row['neighbourhood']=='Diamond Heights'):
        return 67
    elif (row['neighbourhood']=='Japantown'):
        return 90
    elif (row['neighbourhood']=='Sea Cliff'):
        return 68


data_test['transitscore']=data_test.apply(lambda row: transitscoret(row), axis=1)

sid = SentimentIntensityAnalyzer()
for sentence in data_test['name'].values[:2]:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
'''sid = SentimentIntensityAnalyzer()
for sentence in data_test['description'].values[:3]:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()

'''
from nltk.corpus import stopwords   # stopwords to detect language
from nltk import wordpunct_tokenize # function to split up our words

def get_language_likelihood(input_text):
    """Return a dictionary of languages and their likelihood of being the 
    natural language of the input text
    """
 
    input_text = input_text.lower()
    input_words = wordpunct_tokenize(input_text)
 
    language_likelihood = {}
    total_matches = 0
    for language in stopwords._fileids:
        language_likelihood[language] = len(set(input_words) &
                set(stopwords.words(language)))
 
    return language_likelihood
 
def get_language(input_text):
    """Return the most likely language of the given text
    """ 
    likelihoods = get_language_likelihood(input_text)
    return sorted(likelihoods, key=likelihoods.get, reverse=True)[0]


data_test['name'][1904]="room"
data_test['name'][13463]="room"
data_test['name'][21249]="room"
data_test['name'][21665]="room"
data_test['name'][21802]="room"

data_test['description'][9445]="room"
data_test['description'][9486]="room"
data_test['description'][22537]="room"
data_test['description'][5235]="room"
data_test['description'][17764]="room"
data_test['description'][19889]="room"
data_test['description'][3103]="room"
data_test['description'][16733]="room"
data_test['description'][2965]="room"
data_test['description'][7114]="room"
data_test['description'][10672]="room"
data_test['description'][11408]="room"
data_test['description'][13700]="room"
data_test['description'][24607]="room"
name_ft = [r for r in data_test['name']]

description_ft = [r for r in data_test['description']]

sid = SentimentIntensityAnalyzer()
pscores = [sid.polarity_scores(comment) for comment in name_ft]
pscoresd = [sid.polarity_scores(comment) for comment in description_ft]

data_test['name_compound'] = [score['compound'] for score in pscores]
data_test['name_negativity'] = [score['neg'] for score in pscores]
data_test['name_neutrality'] = [score['neu'] for score in pscores]
data_test['name_positivity'] = [score['pos'] for score in pscores]

data_test['desc_compound'] = [score['compound'] for score in pscoresd]
data_test['desc_negativity'] = [score['neg'] for score in pscoresd]
data_test['desc_neutrality'] = [score['neu'] for score in pscoresd]
data_test['desc_positivity'] = [score['pos'] for score in pscoresd]

def data_as_of_t(row):
    if (row['city']=='Boston'):
        return datetime.strptime('2017-10-06', '%Y-%m-%d').date()
    elif (row['city']=='NYC'):
        return datetime.strptime('2017-10-02', '%Y-%m-%d').date()
    elif (row['city']=='LA'):
        return datetime.strptime('2017-05-02', '%Y-%m-%d').date()   
    elif (row['city']=='SF'):
        return datetime.strptime('2017-10-02', '%Y-%m-%d').date()     
    elif (row['city']=='Chicago'):
        return datetime.strptime('2017-05-10', '%Y-%m-%d').date()    
    elif (row['city']=='DC'):
        return datetime.strptime('2017-05-10', '%Y-%m-%d').date() 

data_test['data_as_of']=data_test.apply(lambda row: data_as_of_t(row), axis=1)

data_test['first_review'] = pd.to_datetime(data_test['first_review'])
data_test['host_since'] = pd.to_datetime(data_test['host_since'])
data_test['last_review'] = pd.to_datetime(data_test['last_review'])


data_test['first_review'] = data_test['first_review'].apply(lambda x: x.date())
data_test['host_since'] = data_test['host_since'].apply(lambda x: x.date())
data_test['last_review'] = data_test['last_review'].apply(lambda x: x.date())

data_test['DateDiffFirstReview'] = (data_test.data_as_of - data_test.first_review)/ np.timedelta64(1, 'D')
data_test['DateDiffHostSince'] = (data_test.data_as_of - data_test.host_since)/ np.timedelta64(1, 'D')
data_test['DateDiffLastReview'] = (data_test.data_as_of - data_test.last_review)/ np.timedelta64(1, 'D')

data_test['DateDiffFirstReview'].fillna(0, inplace=True)
data_test['DateDiffHostSince'].fillna(0, inplace=True)
data_test['DateDiffLastReview'].fillna(0, inplace=True)

data_test['thumbnail_url'].fillna(0, inplace=True)

def picturet(row):
    if (row['thumbnail_url']==0):
        return 0
    else:
        return 1
    
data_test['picture']=data_test.apply(lambda row: picturet(row), axis=1)

data_test['amenities'] = data_test['amenities'].map(
    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")\
                           for amn in amns.split(",")]))

np.concatenate(data_test['amenities'].map(lambda amns: amns.split("|")).values)

amenities = np.unique(np.concatenate(data_test['amenities'].map(lambda amns: amns.split("|")).values))
amenities_matrix = np.array([data_test['amenities'].map(lambda amns: amn in amns).values for amn in amenities])

amenities_matrix
data_test['amenities'].map(lambda amns: amns.split("|")).head()

np.unique(np.concatenate(data_test['amenities'].map(lambda amns: amns.split("|"))))[1:]

amenities = np.unique(np.concatenate(data_test['amenities'].map(lambda amns: amns.split("|"))))[1:]
amenity_arr = np.array([data_test['amenities'].map(lambda amns: amn in amns) for amn in amenities])
amenity_arr

featurest = data_test[['accommodates', 'host_response_rate', 'city', 'bathrooms', 'cleaning_fee', 'bedrooms', 'beds', 'number_of_reviews',
                     'review_scores_rating','cancellation_policy', 'property_type', 'room_type', 'bed_type', 'host_identity_verified', 'host_has_profile_pic', 'instant_bookable', 'picture','jan_occupancy','feb_occupancy','mar_occupancy','apr_occupancy','may_occupancy','jun_occupancy','jul_occupancy','aug_occupancy','sep_occupancy','oct_occupancy','nov_occupancy','dec_occupancy', 'walkscore','transitscore','dist_to_citycenter','dist_to_attr1','dist_to_attr2','dist_to_attr3','dist_to_attr4','dist_to_attr5','dist_to_attr6', 'name_positivity','desc_positivity','DateDiffFirstReview','DateDiffHostSince','DateDiffLastReview']]


featurest = pd.concat([featurest, pd.DataFrame(data=amenity_arr.T, columns=amenities)], axis=1)

for tf_feature in ['host_identity_verified', 'host_has_profile_pic', 'instant_bookable']:
    featurest[tf_feature] = featurest[tf_feature].map(lambda s: False if s == "f" else True)

for categorical_feature in ['cancellation_policy', 'property_type', 'room_type', 'bed_type']:
    featurest = pd.concat([featurest, pd.get_dummies(featurest[categorical_feature])], axis=1)

featurest.head()
host_response_r = []
for i in featurest['host_response_rate']:
    i = str(i)
    i = i.strip('%')
    i = float(i)
    host_response_r.append(i)

featurest['host_response_rate'] = host_response_r
featurest.drop(['host_response_r'], axis = 1, inplace = True)
featurest.head()

featurest = pd.concat([featurest, pd.get_dummies(featurest['city'])], axis=1)
featurest.drop(['city','cancellation_policy', 'property_type', 'room_type','bed_type'], axis = 1, inplace = True)
featurest.head()

for col in featurest.columns[featurest.isnull().any()]:
    print(col)

list(featurest.columns.values)

for col in featurest.columns[featurest.isnull().any()]:
    featurest[col] = featurest[col].fillna(featurest[col].median())


featurest.sort_index(axis=1, inplace=True)
Xtest = featurest
Xtest.head()
Xtest.drop([ 'long_term', 'Calumet Heights', 'Knox Hill', 'Lenox', 'New Dorp', 'Revere', 'Sauganash', 'South Deering', 'West Rancho Dominguez'], axis = 1, inplace = True)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xtest2 = sc_X.fit_transform(Xtest)
Xtest2
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Target, test_size = 0.001, random_state = 891990)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import lightgbm as lgb

gbm = lgb.LGBMRegressor(num_leaves=200, learning_rate=0.009105, n_estimators=3000, verbose=1, max_depth=-1)
#gbm = lgb.LGBMRegressor(num_leaves=190, learning_rate=0.0109, n_estimators=4000, verbose=1, max_depth=-1)

gbm.fit(X_train, y_train)
y_pred = gbm.predict(Xtest2)
y_gbm = pd.DataFrame(y_pred)
submission = pd.concat([data_test['id'], y_gbm], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_knn_lgb18.csv', sep=',', index=False)

# Plot feature importance
top = 35
feature_importance = gbm.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize = (10,6))
plt.barh(pos[-top:], feature_importance[sorted_idx][-top:], align='center')
plt.yticks(pos[-top:], X.columns[sorted_idx][-top:])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
xgb = xgb.XGBRegressor(learning_rate=0.04, max_depth=10, min_child_weight=2, n_estimators=1500, seed=25)
xgb.fit(X_train, y_train)
y_pred2 = xgb.predict(Xtest2)
y_xgb = pd.DataFrame(y_pred2)
submission = pd.concat([data_test['id'], y_xgb], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_jnn_k18.csv', sep=',', index=False)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras import optimizers
keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
from keras.optimizers import SGD
sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
from keras import optimizers
keras.optimizers.RMSprop(lr=0.02, rho=0.9, epsilon=None, decay=0.0)
from keras.optimizers import RMSprop
opt = RMSprop(lr=0.0001, decay=1e-6)
from keras.callbacks import ModelCheckpoint

checkpointer=ModelCheckpoint(filepath='airbnb.model.best.hdf5',verbose=1,save_best_only=True)

#INITIATE
classifier=[]
classifier = Sequential()

#ADD INPUT AND HIDDEN
#classifier.add(Dense(output_dim=1077,activation='relu',init='glorot_uniform', input_dim=718))
classifier.add(Dense(output_dim=240,activation='relu', input_dim=211))
#classifier.add(Dense(output_dim=718,activation='relu'))
#classifier.add(Dense(output_dim=718,activation='relu'))
#classifier.add(Dense(output_dim=718,activation='tanh'))
#classifier.add(Dense(output_dim=718,activation='tanh'))
#classifier.add(Dense(output_dim=659,activation='relu'))
classifier.add(Dense(output_dim=240,activation='relu'))
classifier.add(Dense(output_dim=240,activation='relu'))
classifier.add(Dense(output_dim=220,activation='relu'))
classifier.add(Dense(output_dim=220,activation='relu'))


classifier.add(Dense(output_dim=220,activation='relu'))
classifier.add(Dense(output_dim=220,activation='relu'))
classifier.add(Dense(output_dim=220,activation='relu'))
#classifier.add(Dense(output_dim=733,activation='relu'))
#classifier.add(Dense(output_dim=5,activation='relu'))

#OUTPUT
classifier.add(Dense(output_dim=1,activation='relu'))

#COMPILE
#classifier.compile(optimizer='adam',loss='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine'])
#classifier.compile(optimizer="rmsprop",loss=root_mean_squared_error, metrics=['mse', 'mae', 'mape', 'cosine'])
#classifier.compile(optimizer=sgd,loss=root_mean_squared_error, metrics=['mse', 'mae', 'mape', 'cosine'])
classifier.compile(optimizer='adam',loss='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine'])


classifier.fit(X_train,y_train,batch_size=2200,epochs=30, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)

y_predANN = classifier.predict(Xtest2)
y_ann = pd.DataFrame(y_predANN)
submission = pd.concat([data_test['id'], y_ann], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_submission_ANNadam_kn23.csv', sep=',', index=False)

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

bag = ensemble.BaggingRegressor(n_estimators=100)
bag.fit(X_train, y_train)
y_pred4 = bag.predict(Xtest2)
y_bag = pd.DataFrame(y_pred4)
submission = pd.concat([data_test['id'], y_bag], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_submission_bag_kn018.csv', sep=',', index=False)

xtr = ensemble.ExtraTreesRegressor(n_estimators=100)
xtr.fit(X_train, y_train)
y_pred5 = bag.predict(Xtest2)
y_xtr = pd.DataFrame(y_pred5)
submission = pd.concat([data_test['id'], y_xtr], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_submission_xtr_kn018.csv', sep=',', index=False)

rf=ensemble.RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
y_pred6 = rf.predict(Xtest2)
y_rf = pd.DataFrame(y_pred6)
submission = pd.concat([data_test['id'], y_rf], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_submission_rf_kn02.csv', sep=',', index=False)

gb=ensemble.GradientBoostingRegressor(n_estimators=100)
gb.fit(X_train, y_train)
y_pred7 = rf.predict(Xtest2)
y_gb = pd.DataFrame(y_pred7)
submission = pd.concat([data_test['id'], y_gb], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_submission_gb_kn02.csv', sep=',', index=False)

#ensemble = y_xgb*0.05 + y_gbm*0.90 + y_ann*0.05
ensemble = y_xgb*0.01 + y_gbm*0.96 + y_ann*0 + y_bag*0.01 + y_xtr*0.01 + y_rf*0.01 + y_gb*0
y_ens = pd.DataFrame(ensemble)
submission = pd.concat([data_test['id'], y_ens], axis=1)
submission.columns = ['id', 'log_price']
submission.to_csv('RudyCelekli_k_ens15.csv', sep=',', index=False)
ensemble.head()

