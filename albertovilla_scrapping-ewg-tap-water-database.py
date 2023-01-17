import pandas as pd
import progressbar
import pprint
import requests
from bs4 import BeautifulSoup
# when reading the dataframe I specify that zip code is a string so the Os at the beginning of the zip code are not lost
POSTAL_CODES_PATH = '../input/simplemaps-zip/postalcodes.csv'
zip_df = pd.read_csv(POSTAL_CODES_PATH, dtype={'zip': str})
pp = pprint.PrettyPrinter(indent=4)
print('Postal codes dataframe shape: ', zip_df.shape)
print('Missing information in % rounded to 2 decimals')
pp.pprint((zip_df.isna().sum() / zip_df.shape[0]).round(2))
unused_fields = ['zcta', 'parent_zcta', 'county_fips', 'county_name', 'all_county_weights', 'imprecise', 'military']
zip_df.drop(unused_fields, axis=1, inplace=True)
zip_df.head()
pp.pprint(zip_df.state_id.unique())
states_dictionary = dict(zip(zip_df.state_id,zip_df.state_name))
pp.pprint(states_dictionary)
missing_state = {
    'AE': 'Armed Forces Africa, Canada, Europe, Middle East', 
    'AA': 'Armed Forces Americas (except Canada)',
    'AP': 'Armed Forces Pacific'
}

zip_df['state_name'] = zip_df['state_name'].fillna(zip_df['state_id'].map(missing_state))
zip_df.isna().sum()
zip_df[zip_df['state_id'] == 'PR'].head()
valid_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 
    'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 
    'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 
    'West Virginia', 'Wisconsin', 'Wyoming']
for state in valid_states:
    print(state, str((zip_df['state_name'] == state).sum()))
print('Initial number of zip codes: ', zip_df.shape[0])
zip_df = zip_df[zip_df['state_name'].isin(valid_states)]
print('Final number of zip codes: ', zip_df.shape[0])
BASE_URL = 'https://www.ewg.org/tapwater/'
SEARCH_URL_START = 'search-results.php?zip5='
SEARCH_URL_END = '&searchtype=zip'

url = 'https://www.ewg.org/tapwater/search-results.php?zip5=96799&searchtype=zip'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
def got_results_from_url(soup, url):
    error = soup.find('h2', text = 'No systems found that match your search')
    if (error):
        return False
    else:
        return True
got_results_from_url(soup, url)
zip_df.head()
def generate_url_from_zip(zip_value):
    return BASE_URL + SEARCH_URL_START + zip_value + SEARCH_URL_END

def get_population(people_served_tag):
    return int(people_served_tag.replace('Population served:', '').replace(',',''))

def get_city(element):
    return element.text.split(',')[0].strip()

def get_state(element):
    print(element.text)
    return element.text.split(',')[1].strip()

def get_city_and_state(element):
    split_element = element.text.split(',')
    if len(split_element) == 2:
        return split_element[0].strip(), split_element[1].strip()
    else:
        return split_element[0].strip(), '-'

def extract_info_from_row(elements):
    row_info = {}
    row_info['url'] = BASE_URL + elements[0].find('a')['href']
    row_info['utility_name'] = elements[0].text
    row_info['city'], row_info['state'] = get_city_and_state(elements[1])
    row_info['people_served'] = get_population(elements[2].text)
    return row_info

def process_results(results, zip_value):
    zip_results = []
    result_rows = results.find_all('tr')
    for row in result_rows:
        elements = row.find_all('td')
        if elements:
            element = extract_info_from_row(elements)
            element['zip'] = zip_value
            zip_results.append(element)
    return zip_results

def process_zip(zip_value):
    url = generate_url_from_zip(zip_value)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    if got_results_from_url(soup, url):
        results = soup.find_all('table', {'class': 'search-results-table'})
        # NOTE: there are two search-results-table, first one shows the results for the 
        # largest utilities serving County, the second one is more complete and includes
        # utilities serving the searched zip and the surrounding county
        # The process will be applied only to the LARGEST UTILITIES which is the first 
        # result
        return process_results(results[0], zip_value)
    else:
        return []
zip_results = process_zip('00501')
zip_results
def get_contaminants(soup, contaminant_type):
    section = soup.find('ul', {'class': 'contaminants-list', 'id': contaminant_type})
    contaminants_type = section.find_all('div', {'class': 'contaminant-name'})
    contaminants = []
    for contaminant in contaminants_type:
        contaminants.append(contaminant.find('h3').text)
    return contaminants

def get_contaminants_above_hbl(soup):
    return get_contaminants(soup, 'contams_above_hbl')

def get_contaminants_other(soup):
    return get_contaminants(soup, 'contams_other')    
url = 'https://www.ewg.org/tapwater/system.php?pws=NY5110526'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')

get_contaminants_above_hbl(soup)
get_contaminants_other(soup)
zip_df_small = zip_df.sample(3, random_state=8)
def scrap_ewg_tap_water_database(df):
    data = []
    
    # Step 1: get information about the utilities in each zip code    
    for zip_code in df['zip']:
        utilities = process_zip(zip_code)
        data = data + utilities
        
    # Step 2: for each utility obtain the contaminants
    for utility in data:
        r = requests.get(utility['url'])
        soup = BeautifulSoup(r.content, 'html.parser')
        print('Getting contaminants from: ', utility['url'])
        utility['contaminants_above_hbl'] = get_contaminants_above_hbl(soup)
        utility['contaminants_other'] = get_contaminants_other(soup)
    return data
ewg_tap_water = scrap_ewg_tap_water_database(zip_df_small)
ewg_tap_water_df = pd.DataFrame(ewg_tap_water)
ewg_tap_water_df.head()
for contaminant in ewg_tap_water_df['contaminants_other']:
    print(contaminant)
def generate_url_from_zip(zip_value):
    return BASE_URL + SEARCH_URL_START + zip_value + SEARCH_URL_END

def get_population(people_served_tag):
    return int(people_served_tag.replace('Population served:', '').replace(',',''))

def get_city(element):
    return element.text.split(',')[0].strip()

def extract_info_from_row(elements):
    row_info = {}
    row_info['url'] = BASE_URL + elements[0].find('a')['href']
    row_info['utility_name'] = elements[0].text
    row_info['city'] = get_city(elements[1])
    row_info['people_served'] = get_population(elements[2].text)
    return row_info

def process_results(results, zip_value, state_id):
    zip_results = []
    result_rows = results.find_all('tr')
    for row in result_rows:
        elements = row.find_all('td')
        if elements:
            element = extract_info_from_row(elements)
            element['zip'] = zip_value
            element['state'] = state_id
            zip_results.append(element)
    return zip_results

def process_zip(zip_value, state_id):
    url = generate_url_from_zip(zip_value)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    if got_results_from_url(soup, url):
        results = soup.find_all('table', {'class': 'search-results-table'})
        # NOTE: there are two search-results-table, first one shows the results for the 
        # largest utilities serving County, the second one is more complete and includes
        # utilities serving the searched zip and the surrounding county
        # The process will be applied only to the LARGEST UTILITIES which is the first 
        # result
        return process_results(results[0], zip_value, state_id)
    else:
        return []
    
def get_contaminants(soup, contaminant_type):
    section = soup.find('ul', {'class': 'contaminants-list', 'id': contaminant_type})
    if section:
        contaminants_type = section.find_all('div', {'class': 'contaminant-name'})
        contaminants = []
        for contaminant in contaminants_type:
            contaminant_name = contaminant.find('h3').text
            if len(contaminant_name) < 80:
                contaminants.append(contaminant.find('h3').text)
        return contaminants
    else:
        return []
    
def get_contaminants_above_hbl(soup):
    return get_contaminants(soup, 'contams_above_hbl')

def get_contaminants_other(soup):
    return get_contaminants(soup, 'contams_other')  

def get_all_contaminants(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    contaminants_above_hbl = get_contaminants_above_hbl(soup)
    contaminants_other = get_contaminants_other(soup)
    
    return (contaminants_above_hbl, contaminants_other)
    
def scrap_contaminants_from_df(df):
    contaminants_rows = []
   
    status = 0
    bar = progressbar.ProgressBar(max_value=df.shape[0])
    
    for index, utility in df.iterrows():
        # percentage of completion
        bar.update(status)        
        status = status + 1
        
        r = requests.get(utility['url'])
        soup = BeautifulSoup(r.content, 'html.parser')
        
        row = {}
        row['zip'] = utility['zip']
        row['city'] = utility['city']        
        row['contaminants_above_hbl'] = get_contaminants_above_hbl(soup)
        row['contaminants_other'] = get_contaminants_other(soup)
        contaminants_rows.append(row)
    bar.finish()
    
    return contaminants_rows
    
def scrap_ewg_tap_water_database(df):
    data = []
       
    status = 0
    bar = progressbar.ProgressBar(max_value=df.shape[0])
    
    # Step 1: get information about the utilities in each zip code    
    for index, row in df.iterrows():
        # percentage of completion
        bar.update(status)        
        status = status + 1
        
        utilities = process_zip(row['zip'], row['state_id'])
        data = data + utilities
    bar.finish()
    
    # Let's save this to a CSV just in case the second process does not work
    utilities_df = pd.DataFrame(data)
    utilities_df.to_csv('utilities.csv', index=False)
        
    # Step 2: for each utility obtain the contaminants
    status = 0
    bar = progressbar.ProgressBar(max_value=len(data))
    for utility in data:
        # percentage of completion
        bar.update(status)        
        status = status + 1
        
        r = requests.get(utility['url'])
        soup = BeautifulSoup(r.content, 'html.parser')
        utility['contaminants_above_hbl'] = get_contaminants_above_hbl(soup)
        utility['contaminants_other'] = get_contaminants_other(soup)
    bar.finish()
    
    return data
# IMPORTANT NOTE: THIS PROCESS TAKES A LONG TIME - UNCOMMENT IF YOU WANT TO PROCEED
# ewg_tap_water = scrap_ewg_tap_water_database(zip_df)
# ewg_tap_water_df = pd.DataFrame(ewg_tap_water)
# ewg_tap_water_df.to_csv('ewg.csv')