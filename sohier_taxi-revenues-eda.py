import json

import pandas as pd

import re



from Levenshtein import distance as edit_distance

from string import punctuation
MONTHS_PER_YEAR = 12

DIGITS_PER_MONTH_CODE = 2

BASE_PATH = "../input/chicago_taxi_trips_2016_{0}.csv"
column_datatypes = {

     'company': object,

     'dropoff_census_tract': float,

     'dropoff_community_area': float,

     'dropoff_latitude': float,

     'dropoff_longitude': float,

     'extras': float,

     'fare': float,

     'payment_type': object,

     'pickup_census_tract': float,

     'pickup_community_area': float,

     'pickup_latitude': float,

     'pickup_longitude': float,

     'taxi_id': object,

     'tips': float,

     'tolls': float,

     'trip_end_timestamp': object,

     'trip_miles': float,

     'trip_seconds': float,

     'trip_start_timestamp': object,

     'trip_total': float

         }



df = pd.read_csv(BASE_PATH.format('01'), dtype=column_datatypes)
df.head()
with open("../input/column_remapping.json") as f_open:

    column_id_map = json.load(f_open)

id_to_company = column_id_map['company']
original_company_names = [(x, id_to_company[x]) for x in df.company.unique()

                          if not pd.isnull(x)]

original_company_names.sort(key=lambda x: x[1])
for row in original_company_names[7:14]:

    print(row)
def re_search_with_nulls(pattern, string):

    # version of re.search made safe in case of no matches 

    match = re.search(pattern, string)

    if match:

        return match.group(0)



    

def condense_name(company_name):

    # strip whitespace and punctuation to handle name differences like 

    # "3591 - 63480 Chuks Cab" vs "3591- 63480 Chuk's Cab"

    condensed_name = ''.join([char for char in company_name.lower() if char not in punctuation])

    # drop stopwords to handle differences like 

    #

    stopwords = {'cab', 'co', 'corp', 'inc'}

    return ''.join([x for x in condensed_name.split() if x not in stopwords])





def get_match_id(base_raw_name, base_details, partner_raw_name, partner_details):

    # matches companies sharing IDs or condensed names that differ by at most one letter

    if (base_details['city_id'] == partner_details['city_id']) and base_details['city_id'] is not None:

        return partner_details['id']

    if not all(['condensed_name' in base_details, 'condensed_name' in partner_details]):

        return None

    if edit_distance(base_details['condensed_name'], partner_details['condensed_name']) <= 1:

        return partner_details['id']

company_info = {raw_name: {'id': id} for id, raw_name in id_to_company.items()}

for raw_name in company_info:

    # regex pattern matches exactly four digits at the beginning of the string

    company_info[raw_name]['city_id'] = re_search_with_nulls('^\d{4}', raw_name)

    # regex pattern matches any letter plus the rest of the string after that letter

    company_info[raw_name]['name'] = re_search_with_nulls('[a-zA-Z]+.*', raw_name)

    if company_info[raw_name]['name']:

        company_info[raw_name]['condensed_name'] = condense_name(company_info[raw_name]['name'])



all_matches = set()

for raw_name, details in company_info.items():

    current_matches = {get_match_id(raw_name, details, partner_raw_name, partner_details)

                       for partner_raw_name, partner_details in company_info.items()}

    current_matches.discard(None)

    if len(current_matches) > 1:

        all_matches.add(tuple(sorted(current_matches)))



company_dedupe_map = {x: x for x in column_id_map['company']}

for company_aliases in all_matches:

    for alias in company_aliases:

        company_dedupe_map[alias] = company_aliases[0]
consolidated_ids = [x for x in df.company.unique() 

                    if not pd.isnull(x) and company_dedupe_map[x] != x]



print('\n'.join(sorted([f'"{id_to_company[x]}" -> "{id_to_company[company_dedupe_map[x]]}"'

       for x in consolidated_ids])))
df.company = df.company.map(company_dedupe_map, na_action='ignore')
annual_revenues = df[['trip_total', 'company']].groupby('company').sum()



for month in range(2, MONTHS_PER_YEAR + 1):

    df = pd.read_csv(BASE_PATH.format(str(month).zfill(DIGITS_PER_MONTH_CODE)),

                                dtype=column_datatypes)

    df.company = df.company.map(company_dedupe_map, na_action='ignore')

    annual_revenues = annual_revenues.add(df[['trip_total', 'company']].groupby('company').sum(),

                                          fill_value=0)
def format_to_dollars(number):

    return '${:,.0f}'.format(number)
annual_revenues.sort_values(by='trip_total', inplace=True, ascending=False)

annual_revenues.index = annual_revenues.index.map(lambda x: id_to_company[x])

print(annual_revenues.trip_total.apply(format_to_dollars))