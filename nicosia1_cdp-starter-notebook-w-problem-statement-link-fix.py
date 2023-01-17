# standard libs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json

# plotting libs
import seaborn as sns

# geospatial libs
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon
import geopandas as gpd
import folium
import plotly.graph_objects as go
import plotly_express as px

# set in line plotly 
from plotly.offline import init_notebook_mode;
init_notebook_mode(connected=True)

print(os.getcwd())
class cdp_kpi:
    """
    import corporate climate change response data
    """
    cc_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')
    """
    import corporate water security response data
    """
    ws_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2019_Full_Water_Security_Dataset.csv')
    # import cities response df
    cities_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")
    # external data - import CDC social vulnerability index data - census tract level
    svi_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/CDC Social Vulnerability Index 2018/SVI2018_US.csv")
    """
    cities metadata - lat,lon locations for US cities
    """
    cities_meta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Simple Maps US Cities Data/uscities.csv")
    """
    cities metadata - CDP metadata on organisation HQ cities
    """
    cities_cdpmeta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Locations of Corporations/NA_HQ_public_data.csv")
    
    def list_dedupe(self, x):
        """
        Convert list to dict and back to list to dedupe

        Parameters
        ----------
        x: list
            Python list object

        Returns
        -------
        dictionary:
            dictionary object with duplicates removed

        """
        return list(dict.fromkeys(x))
    
    def __init__(self):
        """
        Extract city response data for question 6.2 Does your city collaborate in partnership with businesses in your city on sustainability projects?
        Map cities to organisations who are headquartered within that city, using the NA_HQ_public_data.csv meta data file
        """
        self.cities_6_2 = self.cities_df[self.cities_df['Question Number'] == '6.2']\
            .rename(columns={'Organization': 'City'})
        self.cities_6_2['Response Answer'] = self.cities_6_2['Response Answer'].fillna('No Response')
        # map dict to clean full state names to abbreviations
        self.cities_cdpmeta_df['state'] = self.cities_cdpmeta_df['address_state'].map(self.us_state_abbrev)

        # infill non-matched from dict
        self.cities_cdpmeta_df['state'] = self.cities_cdpmeta_df['state'].fillna(self.cities_cdpmeta_df['address_state'])
        self.cities_cdpmeta_df['state'] = self.cities_cdpmeta_df['state'].replace({'ALBERTA':'AB'})
        self.cities_cdpmeta_df['address_city'] = self.cities_cdpmeta_df['address_city'].replace({'CALGARY':'Calgary'})
        self.cities_cdpmeta_df= self.cities_cdpmeta_df.drop(columns=['address_state'])

        # create joint city state variable
        self.cities_cdpmeta_df['city_state'] = self.cities_cdpmeta_df['address_city'].str.cat(self.cities_cdpmeta_df['state'],sep=", ")
        #Summarise the cities metadata to count the number organisations (HQ) per city
        self.cities_count = self.cities_cdpmeta_df[['organization', 'address_city', 'state', 'city_state']]\
        .groupby(['address_city', 'state', 'city_state']).count().\
            sort_values(by = ['organization'],ascending = False)\
                .reset_index()\
                    .rename(columns={'organization' : 'num_orgs'})
        # convert indexes to columns'
        self.cities_count.reset_index(inplace=True)
        self.cities_count = self.cities_count.rename(columns = {'index':'city_id'})
        self.cities_df.reset_index(inplace=True)
        self.cities_df = self.cities_df.rename(columns = {'index':'city_org_id'})

        # convert id and city label columns into lists
        self.city_id_no = self.list_dedupe(self.cities_count['city_id'].tolist())
        self.city_name = self.list_dedupe(self.cities_count['address_city'].tolist())

        self.city_org_id_no = self.list_dedupe(self.cities_df['city_org_id'].tolist())
        self.city_org_name = self.list_dedupe(self.cities_df['Organization'].tolist())

        # remove added index column in cities df
        self.cities_df.drop('city_org_id', inplace=True, axis=1)
        self.cities_count.drop('city_id', inplace=True, axis=1)

        # zip to join the lists and dict function to convert into dicts
        self.city_dict = dict(zip(self.city_id_no, self.city_name))
        self.city_org_dict = dict(zip(self.city_org_id_no, self.city_org_name))
        
        # compare dicts - matching when city name appears as a substring in the full city org name
        self.city_names_df = pd.DataFrame(columns=['City ID No.','address_city', 'City Org ID No.','City Org', 'Match']) # initiate empty df

        for ID, seq1 in self.city_dict.items():
            for ID2, seq2 in self.city_org_dict.items():
                m = re.search(seq1, seq2) # match string with regex search 
                if m:
                    match = m.group()
                    # Append rows in Empty Dataframe by adding dictionaries 
                    self.city_names_df = self.city_names_df.append({'City ID No.': ID, 'address_city': seq1, 'City Org ID No.': ID2, 'City Org': seq2, 'Match' : match}, ignore_index=True)

        # subset for city to city org name matches
        self.city_names_df = self.city_names_df.loc[:,['address_city','City Org']]
        self.cities_count  = pd.merge(self.cities_count, self.city_names_df, on='address_city', how='left')
        self.cities_6_2 = self.cities_6_2[['City', 'Response Answer']].rename(columns={'City' : 'City Org'})
        self.cities_count = pd.merge(left=self.cities_count, right=self.cities_6_2, how='left', 
                                on ='City Org').rename(columns={'Response Answer' : 'Sustainability Project Collab.'})

        self.cities_count['Sustainability Project Collab.'] = self.cities_count['Sustainability Project Collab.'].fillna('No Response')
        self.cities_meta_df = self.cities_meta_df[['city', 'state_id', 'lat','lng']].rename(columns={'city' : 'address_city', 'state_id' : 'state'})
        
        # join coordinates to cities count
        self.cities_count = pd.merge(left=self.cities_count, right=self.cities_meta_df, how='left', on=['address_city', 'state'])

        # convert text response to question 6.2 to an integar encoding 
        resp_int_df = self.cities_count[["Sustainability Project Collab."]]
        resp_int_df= resp_int_df.rename(columns={'Sustainability Project Collab.' : 'resp_int'})

        labels = resp_int_df['resp_int'].unique().tolist()
        mapping = dict( zip(labels,range(len(labels))) )
        resp_int_df.replace({'resp_int': mapping},inplace=True)

        resp_list = resp_int_df['resp_int'].tolist()
        self.cities_count['resp_int'] = resp_list 
        
        self.cc_2_4a = self.cc_df[self.cc_df['question_number'] == 'C2.4a']
        cities_cdpmeta_join = self.cities_cdpmeta_df[["account_number", 'survey_year', 'address_city']]
        self.cc_2_4a = pd.merge(left=self.cc_2_4a, right=cities_cdpmeta_join,  left_on=['account_number','survey_year'], right_on = ['account_number','survey_year'])
        
    def City_SVI_Geo(self, city_svi_df, city, shapefile):
        cc_nyc = self.cc_2_4a[(self.cc_2_4a['address_city'] == city)]
        self.cities_6_2['City Org'] = self.cities_6_2['City Org'].replace({city +' City':city})
        cc_nyc = pd.merge(left=cc_nyc, right= self.cities_6_2,  left_on=['address_city'], right_on = ['City Org']).rename(columns={'Response Answer' : 'sustain_collab'})

        #e.g.'../input/cdp-unlocking-climate-solutions/Supplementary Data/NYC CDP Census Tract Shapefiles/nyu_2451_34505.shp'
        # import shapefile of NYC census tracts
        self.geodf = gpd.read_file(shapefile)

        # join geospatial data to SVI unemployment rates ('E_UNEMP')
        gdf_join = self.geodf[['tractid', 'geometry']].to_crs('+proj=robin')
        nyc_join =  nyc_svi_df[['E_UNEMP', 'FIPS']]
        gdf_join["tractid"] = pd.to_numeric(self.geodf["tractid"])
        gdf_nyc = pd.merge(left=gdf_join, right=nyc_join, how='left', left_on='tractid', right_on = 'FIPS')
        return gdf_nyc
        
    def City_CC_Resp(self, cc_city, city_svi_df, county, bb_df):
        # subset for Bronx
        #bb_df = city_svi_df[(city_svi_df.COUNTY == county)]

        # join to city and climate change response data
        print(cc_city.shape)
        cc_city_temp = cc_city.rename(columns={'City Org' : 'City'})
        city_df = pd.merge(cc_city_temp,bb_df,on='City',how='outer')
        return city_df
        
    
    # state abbreviation dictionary
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'American Samoa': 'AS',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Guam': 'GU',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands':'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'
    }

c = cdp_kpi()
#verify data captured
print("c.cc_df.head()")
print(c.cc_df.head())
print("c.cities_df.head()")
print(c.cities_df.head())
print("c.svi_df.head()")
print(c.svi_df.head())
print("c.cities_meta_df.head()") 
print(c.cities_meta_df.head()) 
print("c.cities_cdpmeta_df.head()")
print(c.cities_cdpmeta_df.head())
print("c.cities_6_2.head()")
print(c.cities_6_2.head())
print("c.cities_count.head()")
print(c.cities_count.head())
print("c.cc_2_4a.head()")
print(c.cc_2_4a.head())
nyc_svi_df = c.svi_df[c.svi_df['STCNTY'].isin([36005, 36047, 36061, 36081, 36085])]
nyc_svi_df['City'] = 'New York'
cc_city = c.City_SVI_Geo(nyc_svi_df, "Bronx", '../input/cdp-unlocking-climate-solutions/Supplementary Data/NYC CDP Census Tract Shapefiles/nyu_2451_34505.shp')
bb_df = nyc_svi_df[(nyc_svi_df.COUNTY == "Bronx")]
cc_city = cc_city.rename(columns={'City Org' : 'City'})
#print(cc_city)
nyc_df = c.City_CC_Resp(cc_city,nyc_svi_df, "Bronx", bb_df)

print(nyc_df.shape)
nyc_df.head()
ws_df_4_1c = ws_df[ws_df['question_number'] == 'W4.1c']
ws_df_4_1c = ws_df_4_1c[ws_df_4_1c['response_value'].notnull()]
ws_df_4_1c.head()         
# pivot data
ws_df_4_1c_wide = ws_df_4_1c.pivot_table(index=['account_number', 'organization', 'row_number'],
                                     columns='column_name', 
                                     values='response_value',
                                     aggfunc=lambda x: ' '.join(x)).reset_index()
# identify orgs with facilities within the Hudson river basin
ws_df_4_1c_wide = ws_df_4_1c_wide[ws_df_4_1c_wide['W4.1c_C2River basin'].str.contains('Hudson', na=False)]
ws_df_4_1c_wide.head()
ws_df.head()

sub.to_csv('submission.csv')