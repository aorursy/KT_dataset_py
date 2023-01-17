import numpy as np
import pandas as pd 
import os
import requests
import datetime as dt
import pickle
import gc
from math import sin, cos, sqrt, atan2, radians
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.offline.init_notebook_mode(connected=True)
%matplotlib inline
print(os.listdir("../input/io"))
print(os.listdir("../input/dc-geocode"))
print(os.listdir("../input/dc-zipcode-geocoding"))
projectCols = ['Project ID', 'School ID', 'Teacher ID',
               'Teacher Project Posted Sequence', 'Project Type',
               'Project Subject Category Tree', 'Project Subject Subcategory Tree',
               'Project Grade Level Category', 'Project Resource Category',
               'Project Cost', 'Project Posted Date', 'Project Expiration Date',
               'Project Current Status', 'Project Fully Funded Date']

resourcesCols = ['Project ID','Resource Quantity','Resource Unit Price', 'Resource Vendor Name']
donations = pd.read_csv('../input/io/Donations.csv', dtype = {'Donation Amount': np.float32, 'Donor Cart Sequence': np.int32})
donors = pd.read_csv('../input/io/Donors.csv', dtype = {'Donor Zip':'str'})
projects = pd.read_csv('../input/io/Projects.csv', usecols = projectCols, dtype = {'Teacher Project Posted Sequence': np.float32, 'Project Cost': np.float32})
resources = pd.read_csv('../input/io/Resources.csv', usecols = resourcesCols, dtype = {'Resource Quantity': np.float32,'Resource Unit Price': np.float32})
schools = pd.read_csv('../input/io/Schools.csv', dtype = {'School Zip': 'str'})
teachers = pd.read_csv('../input/io/Teachers.csv')

# These are the geocoding files
mapping = pd.read_csv('../input/dc-geocode/mappingFinal_all_modified.csv')
zipMapping = pd.read_csv('../input/dc-zipcode-geocoding/zipMapping.csv')
# donations
donations['Donation Received Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Included Optional Donation'].replace(('Yes', 'No'), (1, 0), inplace=True)
donations['Donation Included Optional Donation'] = donations['Donation Included Optional Donation'].astype('bool')
donations['Donation_Received_Year'] = donations['Donation Received Date'].dt.year
donations['Donation_Received_Month'] = donations['Donation Received Date'].dt.month
donations['Donation_Received_Day'] = donations['Donation Received Date'].dt.day

# donors
donors['Donor Is Teacher'].replace(('Yes', 'No'), (1, 0), inplace=True)
donors['Donor Is Teacher'] = donors['Donor Is Teacher'].astype('bool')

# projects
cols = ['Project Posted Date', 'Project Fully Funded Date']
projects.loc[:, cols] = projects.loc[:, cols].apply(pd.to_datetime)
projects['Days_to_Fullyfunded'] = projects['Project Fully Funded Date'] - projects['Project Posted Date']

# teachers
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
def name_dataframes(dfList, dfNames):
    '''
    give names to a list of dataframes. 
    Argument:
        dfList = list of dataframes,
        dfNames = list of names for the dataframes
    Return:
        None
    '''
    for df, name in zip(dfList, dfNames):
        df.name = name
    
    return
dfList = [donations, donors, projects, resources, schools, teachers]
dfNames = ['donations', 'donors', 'projects', 'resources', 'schools', 'teachers']
name_dataframes(dfList, dfNames)
# print shapes of dataframes
for df in dfList:
    print(df.name, df.shape)
# print column names of dataframes
for df in dfList:
    print(df.name + ":\n" , df.columns)
def data_summary(df):
    '''print out information about dataframe
    Argument: dataframe name
    Return: dataframe with summary figures
    '''
    temp = pd.DataFrame(data = df.isnull().sum(axis=0))
    temp.columns=['NA count']
    temp['NA %'] = temp['NA count']/(df.shape[0])*100
    temp['# unique vals'] = df.nunique(axis=0)
    temp['dtype'] =  df.dtypes
    temp['dataset'] = df.name
    
    return temp

donations.head(2)
data_summary(donations)
donors.head(2)
data_summary(donors)
projects.head(2)
data_summary(projects)
resources.head(2)
data_summary(resources)
data = resources.loc[resources['Resource Unit Price'] < resources['Resource Unit Price'].quantile(0.90)]['Resource Unit Price']
sns.distplot(data)
plt.title("Distribution of Resources Unit Price \n For prices under 90% percentile")
plt.plot()
schools.head(2)
data_summary(schools)
teachers.head(2)
data_summary(teachers)
projects = projects.loc[projects['School ID'].isin(schools['School ID'])]
projects = projects.loc[projects['Project ID'].isin(resources['Project ID'])]
donations = donations.loc[donations['Project ID'].isin(projects['Project ID'])]
donations = donations.loc[donations['Donor ID'].isin(donors['Donor ID'])]
donors = donors.loc[donors['Donor ID'].isin(donations['Donor ID'])]
def geocode_address(addressList, key):
    '''
    The function takes a list of address and return a dataframe with geocoding information. 
    The most imporant column is "fullgeocode" which contains the output from Google Map API.
    
    Argument: a list of address, API key.  
    Return: a dataframe and two files: 'mappingFinal_{}.csv, 'citiesMappingFinal_{}.pkl'
    
    The neighborhood, country, and state should be used with caution depending on the order of returned json results.
    '''
    
    # create dataframe to store geo-coding data
    mapping = pd.DataFrame(columns = ['address', 'latitude', 'longitude', 'neighborhood', 'county', 'state', 'fullgeocode'])
    counter = -1
    
    # submit API calls and save output to mapping
    for address in addressList:
        
        # save every 500 queries
        counter+=1
        if counter == 500:
            now = dt.datetime.now().microsecond
            pd.DataFrame(mapping).to_csv('mapping_{}.csv'.format(now, encoding='utf8'))
            mapping.to_pickle('citiesMapping_{}.pkl'.format(now, encoding='utf8'))
            counter = -1
            print(len(mapping))
   
        # format query url
        url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(address, key)
        response = requests.get(url)
        
        # quit loop if response code is not okay (200)
        if response.status_code != 200:
            # print error
            print('error:', response.json())
            
            # save a copy of most recent data before break
            now = dt.datetime.now().microsecond
            pd.DataFrame(mapping).to_csv('mapping_{}.csv'.format(now, encoding='utf8'))
            mapping.to_pickle('citiesMapping_{}'.format(now, encoding='utf8'))
            
            break
        
        # continue if result is okay
        else:
            try:
                # extract data from json
                address = address
                latitude = response.json()['results'][0].get('geometry').get('location').get('lat')
                longitude = response.json()['results'][0].get('geometry').get('location').get('lng')
                neighborhood = response.json()['results'][0].get('address_components')[1].get('long_name')
                county = response.json()['results'][0].get('address_components')[2].get('long_name')
                state = response.json()['results'][0].get('address_components')[3].get('long_name')
                fullgeocode = response.json()['results']
            
                # append data to dataframe
                mapping.loc[len(mapping)] = [address, latitude, longitude, neighborhood, county, state, fullgeocode]
        
            except:
                next
    
    # save final copy as csv file
    now = dt.datetime.now().microsecond
    pd.DataFrame(mapping).to_csv('mappingFinal_{}.csv'.format(now, encoding='utf8'))
    mapping.to_pickle('citiesMappingFinal_{}.pkl'.format(now, encoding='utf8'))

    return fullgeocode

def geocode_zip(zipCodeList, apiKey):
    '''
    Input a list of zipcode and API Key and return geocoding information using Google API
    
    Arguments: A list of zipcodes and API key
    Returns: a dataframe with geocoding information
    '''
    
    # create dataframe to store geo-coding data
    mapping = pd.DataFrame(columns = ['zipcode', 'latitude', 'longitude', 'neighborhood', 'county', 'state'])
    
    # submit API calls and save output to mapping
    for zip in zipCodeList:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(zip, apiKey)
        response = requests.get(url)
        zipcode = zip
        
        try:
            # extract data from json
            latitude = response.json()['results'][0].get('geometry').get('location').get('lat')
            longitude = response.json()['results'][0].get('geometry').get('location').get('lng')
            neighborhood = response.json()['results'][0].get('address_components')[1].get('long_name')
            county = response.json()['results'][0].get('address_components')[2].get('long_name')
            state = response.json()['results'][0].get('address_components')[3].get('long_name')
            # append data to dataframe
            mapping.loc[len(mapping)] = [zipcode, latitude, longitude, neighborhood, county, state]
        
        except:
            next
    
    # save a copy as csv file
    pd.DataFrame(mapping).to_csv('mapping_{}.csv'.format(dt.datetime.now().microsecond), encoding='utf8')
    
    return mapping

donorCities = donors['Donor City'].map(str)+', '+donors['Donor State']+', USA'
schoolCities = schools['School City'].map(str)+', '+schools['School State']+', USA'
cities = list(set(donorCities).union(set(schoolCities)))
cities = sorted(cities)
print('# of cities in donor:', len(set(donorCities)))
print('# of cities in schools:', len(set(schoolCities)))
print("# of cities for query:", len(cities))
#mapping = geocode_address(cities, 'enter API key here')
mapping.head()
def parse_geocode(mapping):
    mapping['add_1']= mapping['fullgeocode'].str.extract('formatted_address\'[^\"]+\"([^\"]+)', expand=True)
    mapping['add_2']= mapping['fullgeocode'].str.extract('formatted_address\'[^\']+\'([^\']+)', expand=True)
    mapping['add_zip']= mapping['fullgeocode'].str.extract('\w\w\s(\d{5})', expand=True)
    reg = "long_name': '[a-zA-Z ]+', 'short_name': '([a-zA-Z ]+)', 'types': \['administrative_area_level_1', 'political'\]"
    mapping['add_state_short']= mapping['fullgeocode'].str.extract(reg, expand = True)
    reg = "long_name': '([a-zA-Z ]+)', 'short_name': '[a-zA-Z ]+', 'types': \['administrative_area_level_1', 'political'\]"
    mapping['add_state']= mapping['fullgeocode'].str.extract(reg, expand = True)
    reg = "'long_name': '([a-zA-Z ]+County)'"
    mapping['add_county']= mapping['fullgeocode'].str.extract(reg, expand = True)
    reg = "long_name': '([a-zA-Z ]+)', 'short_name': '[a-zA-Z ]+', 'types': \['locality', 'political'\]"
    mapping['add_city']= mapping['fullgeocode'].str.extract(reg, expand = True)
    reg = "long_name': '([a-zA-Z. ]+)', 'short_name': '[a-zA-Z ]+', 'types': \['country', 'political'\]"
    mapping['add_ctry']= mapping['fullgeocode'].str.extract(reg, expand = True)
    
    #### Revise the mapping file's neighborhood, coutry, state columns 
    mapping['neighborhood'] = mapping['add_city']
    mapping['county'] = mapping['add_county']
    mapping['state'] = mapping['add_state']
    
    #### Keep copy of the search City and search State
    mapping['search_state'] = mapping['address'].str.split(', ', expand = True)[1]
    states = list(mapping['search_state'].unique())
    states.remove('other')
    mapping['search_city'] = mapping['address'].str.split(',', expand = True)[0]
    
    return mapping
mapping = parse_geocode(mapping)
mapping.head()
mapping.loc[mapping['state'] != mapping['search_state'], 'state_mismatch'] = 1
mapping.loc[mapping['neighborhood'] != mapping['search_city'], 'city_mismatch'] = 1
mapping.loc[(mapping['state_mismatch'].isnull()) & (mapping['city_mismatch'].isnull()), 'no_mismatch'] = 1
mapping.loc[(mapping['state_mismatch'].notnull()) & (mapping['city_mismatch'].notnull()), 'both_mismatch'] = 1
print('No mismatches in city and state names: {0:.2f}%'.format(mapping.no_mismatch.sum()/len(mapping)*100))
print('Mismatches in city names: {0:.2f}%'.format(mapping.city_mismatch.sum()/len(mapping)*100))
print('Mismatches in state names: {0:.2f}%'.format(mapping.state_mismatch.sum()/len(mapping)*100))
print('Both city and state are mismatched: {0:.2f}%'.format(mapping.both_mismatch.sum()/len(mapping)*100))
cols = ['address', 'latitude','longitude','neighborhood', 'county', 'state', 'state_mismatch', 'city_mismatch', 'no_mismatch', 'both_mismatch']
donors['mapIndex'] = donors['Donor City'].map(str)+', '+donors['Donor State']+', USA'
donors = donors.merge(mapping[cols], how = 'left', left_on= 'mapIndex', right_on = 'address')
donors = donors.drop(['address'], axis=1)
donors = donors.rename(columns = {'latitude':'Donor_Lat', 'longitude': 'Donor_Lon' })
schools['mapIndex'] = schools['School City'].map(str)+', '+schools['School State']+', USA'
schools = schools.merge(mapping[['address', 'longitude', 'latitude']], how = 'left', left_on= 'mapIndex', right_on = 'address')
schools = schools.drop(['address'], axis=1)
schools = schools.rename(columns = {'latitude':'School_Lat', 'longitude': 'School_Lon' })
zipCodes = schools[schools['School_Lat'].isnull()]['School Zip'].unique()
print("# of zipcodes to query:", len(zipCodes))
#zipMapping = geocode_zip(zipCodes,'Enter API Key Here')
zipMapping.head(2)
zipMapping['zipcode'] = zipMapping['zipcode'].astype('str')
schools['School Zip'] = schools['School Zip'].astype('str')
m = schools['School_Lat'].isnull() # index for missing rows
schools.loc[m,'School_Lat'] = schools.loc[m, 'School Zip'].map(zipMapping.set_index('zipcode').latitude)
schools.loc[m,'School_Lon'] = schools.loc[m, 'School Zip'].map(zipMapping.set_index('zipcode').longitude)
schools.head(2)
donors.head(2)
donorID_2018 = donations[donations['Donation_Received_Year'] == 2018]['Donor ID'].unique()
donorID_prior_2018 = donations[donations['Donation_Received_Year'] < 2018]['Donor ID'].unique()
test_ID = list(set(donorID_2018).intersection(donorID_prior_2018))
donations_prior = donations[(donations['Donor ID'].isin(test_ID)) & (donations['Donation_Received_Year'] < 2018)]
donations_2018 = donations[(donations['Donor ID'].isin(test_ID)) & (donations['Donation_Received_Year'] == 2018)]
del donorID_2018, donorID_prior_2018
gc.collect()
# fill NA for schools
schools['School Percentage Free Lunch'] = schools['School Percentage Free Lunch'].fillna(schools['School Percentage Free Lunch'].median()) #61%
projects = projects.dropna(axis = 0, subset = ['Project Subject Category Tree','Project Subject Subcategory Tree', 'Project Resource Category'])
projects = projects.merge(schools, on = 'School ID', how = 'left')
cols = ['Project Type', 'School Metro Type', 'Project Grade Level Category', 'Project Resource Category']
projFeatures_a = pd.get_dummies(projects[cols], prefix = cols)
schoolStatus = pd.qcut(projects['School Percentage Free Lunch'], 5, labels=["rich", "upper-mid", "mid", "lower-mid", "poor"])
projFeatures_b = pd.get_dummies(schoolStatus, prefix = 'lunchAid')
categories = ['Applied Learning', 'Literacy & Language', 'Special Needs',
       'Math & Science', 'History & Civics', 'Health & Sports',
       'Music & The Arts','Warmth, Care & Hunger']
subCategories = ['Character Education', 'Early Development', 'ESL', 'Special Needs',
       'Literacy', 'College & Career Prep', 'Mathematics', 'Economics',
       'Health & Wellness', 'Environmental Science', 'Applied Sciences',
       'Literature & Writing', 'Health & Life Science', 'Music', 'Other',
       'Foreign Languages', 'Gym & Fitness', 'History & Geography',
       'Civics & Government', 'Visual Arts', 'Community Service',
       'Performing Arts', 'Social Sciences', 'Extracurricular',
       'Team Sports', 'Nutrition Education', 'Parent Involvement',
       'Financial Literacy', 'Warmth, Care & Hunger']
all_cats = list(set(categories).union(set(subCategories)))
all_cats.sort()

print('# of categories:', len(all_cats))
print("The newly defined categories:", all_cats)

# compress the two project category into a single one
projects['all_subjects'] = projects['Project Subject Category Tree']+', '+projects['Project Subject Subcategory Tree']

# Search stream-lined category in the compressed categories
projFeatures_c = pd.DataFrame()
for cat in all_cats:
    projFeatures_c['ProjCat_'+cat] = projects['all_subjects'].str.contains(cat)
# save
projFeatures_c = projFeatures_c.astype(np.float32)
projFeatures =  projFeatures_a.join(projFeatures_b).join(projFeatures_c)
projFeatures['Project ID'] = projects['Project ID']
print("Project Features Shape:", projFeatures.shape)
del projFeatures_a, projFeatures_b, projFeatures_c, schoolStatus
gc.collect()
projFeatures.head(2)
def distance(lat1, lat2, lon1, lon2):
    """
    Calculate the Haversine distance in miles.
    """
    radius = 6371  # km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) * sin(dlat / 2) +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(dlon / 2) * sin(dlon / 2))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = radius * c
    d = d/1.60934 # convert to miles
    
    return d
# get all the donor-project combinations
dist = donations[['Project ID', 'Donor ID']].merge(projects[['Project ID', 'School_Lon', 'School_Lat']], on = 'Project ID', how = 'left')
dist = dist.merge(donors[['Donor ID', 'Donor_Lat', 'Donor_Lon',  'state_mismatch', 'city_mismatch', 'no_mismatch', 'both_mismatch']], on = 'Donor ID', how = 'left')
dist = dist.drop_duplicates(subset=['Project ID', 'Donor ID'], keep='first')

# calculate distance only for pairs that have valid address
dist = dist.loc[dist['no_mismatch'] == 1]

# narrow-down calculation to distinct lon_lat pairs
toCalc = dist.drop_duplicates(subset=['School_Lon', 'School_Lat', 'Donor_Lon', 'Donor_Lat'], keep='first')

# calculate distance
toCalc['dist'] = toCalc.apply(lambda x: distance(x['Donor_Lat'], x['School_Lat'], x['Donor_Lon'], x['School_Lon']), axis = 1)

# merge back to dist
cols = ['School_Lon', 'School_Lat', 'Donor_Lat', 'Donor_Lon', 'dist']
cols_b = ['School_Lon', 'School_Lat', 'Donor_Lat', 'Donor_Lon']
dist = dist.merge(toCalc.filter(items = cols), left_on = cols_b, right_on = cols_b, how = 'left')

# remove rows with null distance calculation
dist = dist[dist['dist'].notnull()]

# Fill NA with 0 
cols = ['state_mismatch', 'city_mismatch', 'no_mismatch', 'both_mismatch']
dist.loc[:,cols] = dist.loc[:,cols].fillna(0)
dist['dist_cut'] = pd.cut(dist['dist'], bins = [-1, 0, 5, 10, 20, 50, 100, 15000], labels = ['0', '1-5', '6-10', '11-20', '21-50', '51-100', '>100'])
distFeatures = pd.get_dummies(dist['dist_cut'], prefix = 'Dist')
distFeatures['Project ID'] = dist['Project ID']
distFeatures['Donor ID'] = dist['Donor ID']
distFeatures['dist'] = dist['dist'] 
distFeatures = distFeatures.drop_duplicates(subset=['Project ID', 'Donor ID'], keep='first')
distFeatures.head()
def donor_summary(donations, projects, dist):
    ''' 
    Generate features to refelect donor's previous donation history:
    'num_proj', 'num_donation', 'num_cart', 'donation_median',
    'donation_mean', 'donation_sum', 'donation_std', 'School ID_count',
    'Teacher ID_count', 'schoolConcentration', 'TeacherConcentration',
    'lunchAid_Avg', 'Dist_median', 'Dist_mean', 'Dist_std', 'Dist_min',
    'Dist_max', 'Dist_Unknown'
    '''
    donations = donations.set_index('Donor ID', drop = False)
    donorSummary = pd.DataFrame(donations['Donor ID']).drop_duplicates(keep = 'first') 
    
    #### Obtain number of projects, # of donations, and the max cart number for each  donor
    countProj = donations.groupby(donations.index).agg({'Project ID':'nunique','Donor Cart Sequence':'max', 'Donation ID':'count'})
    countProj.columns = ['num_proj', 'num_donation','num_cart']
    donorSummary  = donorSummary.merge(countProj, left_index = True,  right_index=True, how = 'left')
    
    #### Summarize donation amount per donor (mean, median, etc)
    donAmt = donations.loc[:,'Donation Amount'].groupby([donations.index]).agg(['median','mean','sum', 'std'])
    donAmt.columns = 'donation_' + donAmt.columns
    donorSummary  = donorSummary.merge(donAmt, left_index = True,  right_index=True, how = 'left')
    
    #### Count # of schools and # of teachers that a donor donates to
    school_teacher = donations[['Project ID', 'Donation Amount', 'Donor ID']].merge(projects[['Project ID', 'School ID', 'Teacher ID']], left_on = 'Project ID', right_on = 'Project ID', how = 'left')
    concentration = school_teacher.groupby('Donor ID').agg({'School ID':'nunique', 'Teacher ID':'nunique'})
    concentration.columns = concentration.columns + '_count'
    donorSummary  = donorSummary.merge(concentration, left_index = True,  right_index=True, how = 'left')
    
    #### Design feature to capture the concentration of donation to schools.
    #### feature that captures doners that donates to multiple schools, and not just have one favorite school
    schoolSum = school_teacher.groupby(['Donor ID', 'School ID'])['Donation Amount'].sum().reset_index(drop = False)
    schoolSum = schoolSum.groupby(['Donor ID'])['Donation Amount'].agg(['sum', 'max'])
    schoolSum['SchoolConcentration'] = schoolSum['max']/schoolSum['sum']
    donorSummary['schoolConcentration'] = schoolSum['SchoolConcentration']
    
    #### Design feature to capture the concentration of donation to a teacher.  
    TeacherSum = school_teacher.groupby(['Donor ID', 'Teacher ID'])['Donation Amount'].sum().reset_index(drop = False)
    TeacherSum = TeacherSum.groupby(['Donor ID'])['Donation Amount'].agg(['sum', 'max'])
    TeacherSum['TeacherConcentration'] = TeacherSum['max']/TeacherSum['sum']
    donorSummary['TeacherConcentration'] = TeacherSum['TeacherConcentration']
    
    #### Calculate the Average School % Free Lunch of the schools that a donor donates to
    lunchAid_Avg = donations[['Project ID', 'Donor ID']].merge(projects[['Project ID', 'School Percentage Free Lunch']], on = 'Project ID', how = 'left')
    lunchAid_Avg['School Percentage Free Lunch'] = lunchAid_Avg ['School Percentage Free Lunch'].fillna(lunchAid_Avg['School Percentage Free Lunch'].median()) 
    lunchAid_Avg = lunchAid_Avg.drop_duplicates(subset=['Project ID', 'Donor ID'], keep='first')
    lunchAid_Avg = lunchAid_Avg.groupby('Donor ID').mean()
    donorSummary = donorSummary.merge(lunchAid_Avg, left_index = True,  right_index=True, how = 'left')
    donorSummary = donorSummary.rename(columns = {'School Percentage Free Lunch':'lunchAid_Avg'})
    
    #### Merge with Average Distance that a Donor Donates to
    dist = dist.set_index(['Donor ID', 'Project ID'])
    donations = donations.set_index(['Donor ID', 'Project ID'])
    donations = donations.merge(dist.filter(items = ['dist']), left_index = True, right_index = True, how = 'left')
    donations = donations.reset_index()
    
    distSummary = donations.groupby('Donor ID')['dist'].agg(['median', 'mean', 'std', 'min','max'])
    distSummary.columns = ['Dist_median', 'Dist_mean','Dist_std','Dist_min', 'Dist_max']
    donorSummary  = donorSummary.merge(distSummary, left_index = True,  right_index = True , how = 'left')
    
    donorSummary['Dist_Unknown'] = donorSummary['Dist_median'].isnull() # Flag rows with missing average distance
    donorSummary['Dist_median']  = donorSummary['Dist_median'].fillna(donorSummary['Dist_median'].median()) 
    donorSummary['Dist_mean']  = donorSummary['Dist_mean'].fillna(donorSummary['Dist_mean'].median()) 
    donorSummary['Dist_std']  = donorSummary['Dist_std'].fillna(donorSummary['Dist_std'].median()) 
    donorSummary['Dist_min']  = donorSummary['Dist_min'].fillna(donorSummary['Dist_min'].median()) 
    donorSummary['Dist_max']  = donorSummary['Dist_max'].fillna(donorSummary['Dist_max'].median()) 
    
    return donorSummary
donorSummary = donor_summary(donations_prior, projects, dist)
donorSummary.head()
donorSummary.shape
def donor_interest(donations, projFeatures, distFeatures):
    ####  Obtain the dollar amount that each donor donates to each project features
    
    #### Sum $ amount of donor's donation to each project
    donor_sum = donations[['Donor ID','Project ID','Donation Amount']].groupby(['Donor ID','Project ID'])['Donation Amount'].agg('sum')
    donor_sum = pd.DataFrame(data = donor_sum).reset_index()
    donor_sum = donor_sum.sort_values(by = 'Donor ID')
    
    ##### Merge donor_sum dataset with the projectFeatures dataset.
    allFeatures = donor_sum.merge(projFeatures, on = 'Project ID', how='left')
    
    #### Many donors donate to projects that are not in the projectFeatures.  Delete those rows from analysis.
    allFeatures = allFeatures.dropna(subset = ['Project Type_Professional Development'])
    
    ##### Add distFeatures to allFeatures.
    allFeatures = allFeatures.merge(distFeatures, on = ['Project ID', 'Donor ID'], how='left')
    
    # Fill missing distance data.
    # Spreadout between Dist 0 and 20 since about half of the donations are under 20 miles.
    allFeatures.loc[allFeatures['Dist_0'].isnull(),'Dist_0'] = 0.56
    allFeatures.loc[allFeatures['Dist_1-5'].isnull(),'Dist_1-5'] = 0.08
    allFeatures.loc[allFeatures['Dist_6-10'].isnull(),'Dist_6-10'] = 0.16
    allFeatures.loc[allFeatures['Dist_11-20'].isnull(),'Dist_11-20'] = 0.20
    allFeatures.loc[allFeatures['Dist_21-50'].isnull(),'Dist_21-50'] = 0
    allFeatures.loc[allFeatures['Dist_51-100'].isnull(),'Dist_51-100'] = 0
    allFeatures.loc[allFeatures['Dist_>100'].isnull(),'Dist_>100'] = 0
    
    ##### Multiply donation amount to each of the one-hot encoded categories for each donor.  
    x = allFeatures['Donation Amount']
    x = x.values.reshape(len(allFeatures), 1)  # reshape from (dim,) to (dim,1) for matrix multiplication (broadcasting)
    y = allFeatures.iloc[:,3:]
    donorInterest = np.multiply(x, y)
    donorInterest['Donor ID'] = allFeatures['Donor ID']
    
    ##### Get total amount of donation per donor
    #The ultimate goal is to get % of dollar breakdown invested in each broader catogry 
    #(Project Type, School Metro, Project Grade, etc).  Not all projects in the donations 
    #dataset are in the projects dataset, so some rows show all 0 (see index 5 below).  
    #Since there are no null values in Project Type, every known projects will have Project Type value, 
    #to get total donation per donor that is in the project list, we will sum the columns Project Type.  
    
    donorTotal = donorInterest['Project Type_Professional Development'] + donorInterest['Project Type_Student-Led'] + donorInterest['Project Type_Teacher-Led']
    donorTotal.index = donorInterest['Donor ID']
    donorTotal = donorTotal.groupby('Donor ID').sum()
    
    ##### Get total amount of donation per feature per donor
    donorInterest = donorInterest.groupby('Donor ID').sum()
    
    #### Step 3: Calculate the percentage of dollar amount that goes to each category for each donor
    donorInterest = donorInterest.div(donorTotal, axis=0)
    
    return donorInterest
donorInterest = donor_interest(donations_prior, projFeatures, distFeatures)
donorInterest.head()
def Get_donorFeatureMatrixNoAdj(donorInterest, donorSummary):
    '''
    Merge donorInterest with donorSummary
    '''
    donorFeatureMatrix = donorInterest.merge(donorSummary, left_index = True, right_index = True, how = 'left')
    donorFeatureMatrix.loc[donorFeatureMatrix['donation_std'].isnull(), 'donation_std'] = 0 # set standard deviation of the single donation donors to 0
    
    return donorFeatureMatrix
donorFeatureMatrixNoAdj = Get_donorFeatureMatrixNoAdj(donorInterest, donorSummary)
donorFeatureMatrixNoAdj.head()
def scaleFeatureMatrix(donorFeatureMatrix):
    '''
    Measure preference of each donor by subtracting the mean
    Arguments: donorFeatureMatrix
    '''
    donorFeatureMatrix = donorFeatureMatrix.drop(['Donor ID'], axis=1)
    
    ### Subtract each column by mean
    donorMeanMatrix = (donorFeatureMatrix.loc[:,'Project Type_Professional Development': 'Dist_>100']).mean(axis = 0)
    donorFeatureMatrix.loc[:,'Project Type_Professional Development': 'Dist_>100'] = donorFeatureMatrix.loc[:,'Project Type_Professional Development': 'Dist_>100'] - donorMeanMatrix # make sure matrix does not include Donor ID, otherwise pandas try to convert string to float, (set Donor ID as index) (set Donor ID as index)
    
    #### Add Percentile Calculation
    cols =  ['num_proj', 'num_donation', 'num_cart', 'donation_median',
       'donation_mean', 'donation_sum', 'donation_std', 'School ID_count',
       'Teacher ID_count', 'schoolConcentration', 'TeacherConcentration',
       'lunchAid_Avg', 'Dist_median', 'Dist_mean', 'Dist_std', 'Dist_min',
       'Dist_max']
    
    donorPercentRank = ((donorFeatureMatrix.loc[:,cols]).rank(ascending = True))/len(donorFeatureMatrix)
    donorPercentRank.columns = 'Percentile_' + donorPercentRank.columns
    donorFeatureMatrix = donorFeatureMatrix.merge(donorPercentRank, left_index = True, right_index = True, how = 'left')
    donorFeatureMatrix = donorFeatureMatrix.drop(cols, axis=1)
    
    return donorFeatureMatrix
donorFeatureMatrix = scaleFeatureMatrix(donorFeatureMatrixNoAdj)
donorFeatureMatrix.head()
donorFeatureMatrix.shape
donorFeatureMatrix.columns
projFeatures.shape
projFeatures.columns
distFeatures.shape
projFeatures.to_csv('projFeatures.csv') 
donorFeatureMatrixNoAdj.to_csv('donorFeatureMatrixNoAdj.csv') #prior to 2018
donorFeatureMatrix.to_csv('donorFeatureMatrix.csv') # prior to 2018
schoolsMapping = schools.filter(items = ['School ID', 'School_Lon', 'School_Lat'])
schoolsMapping.to_csv('schoolsMapping.csv')
donorsMapping = donors.filter(items = ['Donor ID', 'Donor_Lat', 'Donor_Lon', 'state_mismatch', 'city_mismatch', 'no_mismatch', 'both_mismatch'])
donorsMapping.to_csv('donorsMapping.csv')
distFeatures = distFeatures.filter(items = ['Donor ID', 'Project ID', 'dist'])
distFeatures.to_csv('distFeatures.csv')