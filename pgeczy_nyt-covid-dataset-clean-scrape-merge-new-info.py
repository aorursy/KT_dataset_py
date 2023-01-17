import pandas as pd # dataframe analysis and manipulation
import numpy as np # mostly for np.nan

from bs4 import BeautifulSoup # for scraping
import requests # for downloading html files
def github_link_formatter(url):
    '''
    Formats a given direct github-file url so it can be used with pd.read_excel()
    '''
    url = url.replace('github.com','raw.githubusercontent.com')
    url = url.replace('/blob','')
    
    return url
def list_diff(list1,list2):
    '''
    Finds what is missing from, or what is different between, the two lists.
    
    return
    ------
    list_difference: list
    '''
    list_difference = {}
    
    if len(list1) > len(list2):
        bigger = list1
        smaller = list2
        small_list = 'list2'
    else:
        bigger = list2
        smaller = list1
        small_list = 'list1'
        
    for item in bigger:
        if item not in smaller:
            list_difference[item] = f'missing from {small_list}'

    return list_difference
df_mask = pd.read_csv("../input/nytimes-covid19-data/mask-use/mask-use-by-county.csv")
df_mask.columns = df_mask.columns.str.lower()
df_mask.head()
url = github_link_formatter('https://github.com/pomkos/nyt-covid-data/blob/master/data/added_data/mask_mandates.xlsx')

df_mand = pd.read_excel(url, skiprows=2)
df_mand.columns = df_mand.columns.str.lower()
df_mand['type'] = df_mand['type'].str.lower()
df_mand['type_split'] = df_mand['type'].str.split(',')
del df_mand['type_split']
def who_exempt(cell):
    if pd.isna(cell):
        return 'no mandate'
    elif 'children' in cell:
        return 'child exempt'
    elif 'toddler' in cell:
        return 'toddler exempt'
    else:
        return 'no exemptions'
df_mand['children_toddlers_none'] = df_mand['type'].apply(who_exempt)
df_mand['month_mandate'] = df_mand['date'].dt.month
df_mand['month_mandate'] = df_mand['month_mandate'].fillna('no mandate')
df_mand.head()
import datetime as dt
def mandate_when(x):
    if pd.isna(x):
        return 'No Mandate'
    elif x<dt.datetime.strptime('20200515','%Y%m%d'):
        return 'Before May 15'
    elif x>dt.datetime.strptime('20200715','%Y%m%d'):
        return 'After Jul 15'
    else:
        return 'In Between'
df_mand['mandate_when'] = df_mand['date'].apply(mandate_when)
df_mand.head()
county_time = pd.read_csv("../input/nytimes-covid19-data/us-counties.csv",parse_dates=['date'])
county_time = county_time.astype({
    'county':str,
    'fips':float,
    'cases':int,
    'deaths':int
})
# Rename columns
county_time.columns = ['date','county','state','fips','covid_cases','covid_deaths']
# Rearrange columns
county_time = county_time[['date','state','county','fips','covid_cases','covid_deaths']]
county_time.head()
response = requests.get("https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697")
soup = BeautifulSoup(response.content)
table_tag = soup.find(class_='data')

county_scrape = pd.DataFrame(columns=['fips','county','state'])

for tr in table_tag.find_all('tr')[1:]:
    tds = tr.find_all('td')
    d = pd.DataFrame(data = {'fips':[tds[0].text], 'county':[tds[1].text], 'state':[tds[2].text]})
    county_scrape = county_scrape.append(d)
county_scrape = county_scrape.reset_index(drop=True)
county_scrape = county_scrape.astype({
    'fips':int,
    'county':str,
    'state':'category'
})
county_scrape.head()
missing = list_diff(county_scrape['county'].unique(),county_time['county'].unique())
missing_df = pd.DataFrame.from_dict(missing,orient='index')
scraped = county_scrape.set_index('county')
missing_df = missing_df.sort_index().reset_index()
missing_from_scrape = scraped.merge(missing_df, left_on='county',right_on='index')
missing_from_scrape.groupby('state').count().sort_values('fips',ascending=False)
missing_from_scrape[(missing_from_scrape['state']=='AS') ] # repeated for other states
df_mask.head()
county_time.sort_values('date', ascending=False).head(2)
county_cases_aug = county_time[county_time['date'] == '20200815'].groupby('fips').mean().reset_index()
county_cases_aug
county_mask = county_cases_aug.merge(df_mask,left_on='fips',right_on='countyfp')
county_mask = county_mask[['fips', 'covid_cases', 'covid_deaths', 'never', 'rarely', 'sometimes','frequently', 'always']]
county_mask.columns = ['fips', 'covid_cases', 'covid_deaths', 'mask_never', 'mask_rarely', 'mask_sometimes','mask_frequently', 'mask_always']
county_mask.head()
url2 = github_link_formatter('https://github.com/pomkos/nyt-covid-data/blob/master/data/added_data/countypop.xlsx')
countypop = pd.read_excel(url2)
countypop.columns = countypop.columns.str.lower()
region_key = {
    1:'northeast',
    2:'midwest',
    3:'south',
    4:'west'
}
division_key = {
    1:'new_england',
    2:'middle_atlantic',
    3:'east_north_central',
    4:'west_north_central',
    5:'south_atlantic',
    6:'east_south_central',
    7:'west_south_central',
    8:'mountain',
    9:'pacific'
}
sumlev_key = {
    40:'state_or_equiv',
    50:'county_or_equiv'
}
# select only the columns that are relevant, which is the latest (2019) estimates
countypop = countypop[['sumlev','region','division','state','county','stname','ctyname','popestimate2019',
                       'births2019','internationalmig2019','domesticmig2019','rbirth2019','rdeath2019']]
countypop.columns = ['sumlev','region_fips','division_fips','state_fips','county_fips','state','county',
                     'population','births','intnl_migration','domestic_migration','birth_rate','death_rate']
countypop.head()
# find the counties that are present in our dataset
cty_fip = county_time[['state','county','fips']].groupby(['state','county']).mean().reset_index()
cty_fip.head()
# format for merging
countypop['county'] = countypop['county'].str.replace('County','')
countypop['county'] = countypop['county'].str.replace('Parish','')
countypop['county'] = countypop['county'].str.replace(' ','')
# merge fips from covid dataset with 2019pop
cty_pop = cty_fip.merge(countypop,left_on=['state','county'],right_on=['state','county'])
cty_pop = cty_pop[['state', 'county', 'fips', 'sumlev', 'region_fips', 'division_fips',
       'population','births', 'intnl_migration', 'domestic_migration', 'birth_rate',
       'death_rate']]
cty_pop.head()
list_diff(county_time['county'].unique(),cty_pop['county'].unique())
df_county = county_mask.merge(cty_pop, on=['fips'], how='left')
100 * (sum(df_county['population'].isna()) / df_county.shape[0])
df_county.head()
df_county['region'] = df_county['region_fips'].map(region_key)
df_county['division'] = df_county['division_fips'].map(division_key)
df_county['area_type'] = df_county['sumlev'].map(sumlev_key)
df_county['cases_per_million'] = (df_county['covid_cases']/df_county['population']) * 1000000
df_county['cases_per_hthousand'] = (df_county['covid_cases']/df_county['population']) * 100000
df_county['cases_per_thousand'] = (df_county['covid_cases']/df_county['population']) * 1000
df_county['cases_per_hundred'] = (df_county['covid_cases']/df_county['population']) * 100
# rearrange the columns
df_county = df_county[['state','region', 'county', 'division', 'area_type',
                       'population', 'covid_cases', 'covid_deaths', 'cases_per_million', 'cases_per_hthousand', 
                       'cases_per_thousand', 'cases_per_hundred',
                       'mask_never', 'mask_rarely','mask_sometimes', 'mask_frequently', 'mask_always', 
                       'births','intnl_migration', 'domestic_migration', 
                       'birth_rate', 'death_rate',
                       'fips', 'sumlev', 'region_fips', 'division_fips'
                    ]]
df_county.head()
state_covid = pd.read_csv("../input/nytimes-covid19-data/us-states.csv",parse_dates=['date'])
state_covid = state_covid.astype({
    'state':str,
    'fips':float,
    'cases':int,
    'deaths':int
})
# Rename columns
state_covid.columns = ['date','state','state_fips','covid_cases','covid_deaths']
state_covid.head()
state_time = state_covid.copy()
state_covid = state_covid[state_covid['date'] >= '20200815'].reset_index(drop=True)
state_covid = state_covid.groupby('state_fips').mean().reset_index()
response = requests.get("https://www.census.gov/geographies/reference-files/2010/geo/state-area.html")
soup = BeautifulSoup(response.content)
table_tag = soup.find('tbody')

state_land_scrape = pd.DataFrame(columns=range(1,17))

for tr in table_tag.find_all('tr')[3:]:
    tds = tr.find_all('td')
    d = {}
    for i in range(0,17):
        d[i] = [tds[i].text]
    data = pd.DataFrame.from_dict(data=d)
    state_land_scrape = state_land_scrape.append(data)
cols = ['state']
areas = ['total_area_','land_area_','total_water_area_','inland_water_area_','coastal_water_area_',
         'great_lakes_water_area_','territorial_water_area_','latitude','longitude']
for i in range(1,17):
    if (i in range(1,16)) & (i % 2 == 0):
        unit = 'sqkm'
    elif (i in range(1,16)) & (i % 2 != 0):
        unit = 'sqmi'
    if (i == 1) | (i == 2):
        cols.append(f'total_area_{unit}')
    elif (i == 3) | (i == 4):
        cols.append(f'land_area_{unit}')
    elif (i == 5) | (i == 6):
        cols.append(f'total_water_area_{unit}')
    elif (i == 7) | (i == 8):
        cols.append(f'inland_water_area_{unit}')
    elif (i == 9) | (i == 10):
        cols.append(f'coastal_water_area_{unit}')
    elif (i == 11) | (i == 12):
        cols.append(f'great_lakes_water_area_{unit}')
    elif (i == 13) | (i == 14):
        cols.append(f'territorial_water_area_{unit}')
    elif (i == 15):
        cols.append('latitude')
    elif (i == 16):
        cols.append('longitude')
state_land_scrape.columns=cols
state_land_scrape = state_land_scrape.reset_index(drop=True)
state_land_scrape = state_land_scrape.iloc[3:,:].reset_index(drop=True)
state_land_scrape.head()
url3 = github_link_formatter("https://github.com/pomkos/nyt-covid-data/blob/master/data/added_data/statepop.csv")
statepop_raw = pd.read_csv(url3)
statepop_raw.columns = statepop_raw.columns.str.lower()
statepop_raw.head()
statepop = statepop_raw[['name','popestimate2019','sumlev','region','division','state']].reset_index(drop=True)
statepop.columns = ['state', 'population','sumlev','region_fips','division_fips','state_fips']
statepop.head()
# 0 represents territories and regions, which we are not interested in. 
# This data is also included in our county datasets.
statepop = statepop[statepop['state_fips']!=0].reset_index(drop=True)
statepop.head()
# Merge population with land area
statepop = statepop.merge(state_land_scrape, on='state')
statepop = statepop.replace(to_replace = '—', value = np.nan)
statepop = statepop.replace(to_replace = ',', value = '')
statepop.head()
df_state = state_covid.merge(statepop, on='state_fips')
cols = ['region_fips', 'division_fips',
       'total_area_sqmi', 'total_area_sqkm', 'land_area_sqmi',
       'land_area_sqkm', 'total_water_area_sqmi', 'total_water_area_sqkm',
       'inland_water_area_sqmi', 'inland_water_area_sqkm',
       'coastal_water_area_sqmi', 'coastal_water_area_sqkm',
       'great_lakes_water_area_sqmi', 'great_lakes_water_area_sqkm',
       'territorial_water_area_sqmi', 'territorial_water_area_sqkm',
       'latitude', 'longitude']
for col in cols:
    df_state[col] = df_state[col].str.replace('X','NaN')
    df_state[col] = df_state[col].str.replace(',','')
    df_state[col] = df_state[col].astype(float)
df_state['region'] = df_state['region_fips'].map(region_key)
df_state['division'] = df_state['division_fips'].map(division_key)
df_state['area_type'] = df_state['sumlev'].map(sumlev_key)
df_state['cases_per_million'] = (df_state['covid_cases']/df_state['population']) * 1000000
df_state['cases_per_hthousand'] = (df_state['covid_cases']/df_state['population']) * 100000
df_state['cases_per_thousand'] = (df_state['covid_cases']/df_state['population']) * 1000
df_state['cases_per_hundred'] = (df_state['covid_cases']/df_state['population']) * 100
df_state = df_state[['state', 'region', 'division', 'area_type','covid_cases', 'covid_deaths', 'population',
                    'cases_per_million', 'cases_per_hthousand', 'cases_per_thousand',
                    'cases_per_hundred', 'state_fips','sumlev', 'region_fips', 'division_fips', 
                    'total_area_sqmi','total_area_sqkm', 'land_area_sqmi', 'land_area_sqkm',
                    'total_water_area_sqmi', 'total_water_area_sqkm',
                    'inland_water_area_sqmi', 'inland_water_area_sqkm',
                    'coastal_water_area_sqmi', 'coastal_water_area_sqkm',
                    'great_lakes_water_area_sqmi', 'great_lakes_water_area_sqkm',
                    'territorial_water_area_sqmi', 'territorial_water_area_sqkm',
                    'latitude', 'longitude']]
df_state.head()
