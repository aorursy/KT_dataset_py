# modules we'll use
import pandas as pd
import numpy as np
import math
# read in all our data
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
usa_zipcodes= pd.read_csv("../input/usa-zip-codes-to-locations/US Zip Codes from 2013 Government Data.csv")
# set seed for reproducibility
np.random.seed(0) 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/"]).decode("utf8"))


#lat, long 
def haversine(lat1,long1, lat2,long2):
    radius = 6371 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(long2-long1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

def findZipcodeForLocation (lat1, long1, zipcodes):
    minDistance =10^400
    zipCode=0
    for i in range(len(zipcodes)):
        currDistance = haversine(lat1,long1, zipcodes.iloc[i,1], zipcodes.iloc[i,2])
        if currDistance < minDistance : 
            minDistance = currDistance
            zipCode = zipcodes.iloc[i,0]
    return int(round(zipCode))

sf_zipcode = pd.read_csv("../input/sf-zipcodes-limited/SFZ.csv")
sf_zipcode.sample(10)
print ("Number of rows in sf zipcodes: %d \n" % sf_zipcode.shape[0] )
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(5)

# your turn! Find out what percent of the sf_permits dataset is missing
sf_missing_values_count = sf_permits.isnull().sum()
# look at the # of missing points in the first ten columns
sf_missing_values_count[0:10]
sf_total_cells = np.product(sf_permits.shape)
sf_total_missing = sf_missing_values_count.sum()

# percent of data that is missing
(sf_total_missing/sf_total_cells) * 100
sf_permits.loc[:, 'Neighborhoods - Analysis Boundaries':'Zipcode'].sample(20)
sf_permits.rename(columns={'Neighborhoods - Analysis Boundaries': 'Neighborhood'}, inplace=True)
sf_nhoods = sf_permits['Neighborhood'].unique()
#sf_nhoods.sort()
sf_nhoods

sf_zipcode.drop(["Unnamed: 2","Unnamed: 3"], axis=1, inplace=True)
sf_zipcode.head()
sfz= sf_zipcode['Neighborhood'].unique()
sfz.sort()
sfz
sf_zipcode.Neighborhood = sf_zipcode.Neighborhood.replace("\xa0", "", regex=True)
sfzz = sf_zipcode.Neighborhood.unique()
sfzz.sort()
sfzz
sf_permits.Neighborhood.isna().sum()
sf_permits.Zipcode.isna().sum()
sf_permits_nn=sf_permits.query('Neighborhood.isnull() and Zipcode.isnull()', engine='python')
sf_permits_nn.shape[0]
sf_permits[['LAT','LNG']] = sf_permits.Location.str.split(',', expand = True)
sf_permits.LAT= sf_permits.LAT.str.replace('(','')
sf_permits.LNG= sf_permits.LNG.str.replace(')','')
sf_permits.LNG.sample(10)
sf_permits.LAT.sample(10)
sf_permits.Zipcode.unique()
sfz_unique= pd.DataFrame(sf_zipcode.Zipcode.unique())
sfz_unique.columns =['Zipcode']
sfz_unique.shape[0]
#sfz_unique.sample(10)
sfzz_unique =pd.DataFrame(sf_permits.Zipcode.unique())
sfzz_unique.columns =['Zipcode']
sfzz_unique.shape[0]
#ca_zipcodes=usa_zipcodes[(usa_zipcodes['ZIP']>=90001) & (usa_zipcodes['ZIP'] <=96162)]
caf_zipcodes = pd.merge(usa_zipcodes,sfz_unique, left_on =['ZIP'], right_on=['Zipcode'],how='inner')
caf_zipcodes.shape[0]
ca_zipcodes.head(5)
# Find the missing zip codes from location column (now split by LAT and LNG) using 
# USA Zip code dataset filtered by California zip codes
sf_permits.sample(5)
sf_permits['Zipcode'] = sf_permits.where(sf_permits['Zipcode'].isna()).apply(lambda row: findZipcodeForLocation(row['LAT'],row['LNG'],ca_zipcodes), axis=1)
sf_permits.Zipcode.isna().sum()

sff_permits = sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
n_m_values_count = sff_permits.isnull().sum()
n_cols = sff_permits.shape[1]
n_m_values_count[0:n_cols]
zsf_missing_values_count = sff_permits.isnull().sum()
# look at the # of missing points in the first ten columns
zsf_total_cells = np.product(sf_permits.shape)
zsf_total_missing = zsf_missing_values_count.sum()
# percent of data that is missing
(zsf_total_missing/zsf_total_cells) * 100