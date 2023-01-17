# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import requests

import json

print(os.listdir("../input"))

from datetime import datetime as dt



# Any results you write to the current directory are saved as output.
# Make a get request to get the latest position of the international space station from the opennotify api.

api_key = {'APIKey': '65794a30655841694f694a4b563151694c434a68624763694f694a49557a49314e694a392e65794a7063334d694f694a7a644849756333526c6247786863694973496e4e3159694936496d5268626d35355a6a4532514768766447316861577775593239744969776961574630496a6f784e54517a4f5445354f4455304c434a68645751694f694a7a6448497562325268496e302e70496474346b5763677546564d42486b6773484f306a6d5f536d556b6a33586e574946527041516f794f6f'}



response = requests.get("https://stat-xplore.dwp.gov.uk/webapi/rest/v1/schema", headers = api_key)



# Print the status code of the response.

print(response.status_code)
api_key = '65794a30655841694f694a4b563151694c434a68624763694f694a49557a49314e694a392e65794a7063334d694f694a7a644849756333526c6247786863694973496e4e3159694936496d5268626d35355a6a4532514768766447316861577775593239744969776961574630496a6f784e54517a4f5445354f4455304c434a68645751694f694a7a6448497562325268496e302e70496474346b5763677546564d42486b6773484f306a6d5f536d556b6a33586e574946527041516f794f6f'
# api = {'APIKey': '65794a30655841694f694a4b563151694c434a68624763694f694a49557a49314e694a392e65794a7063334d694f694a7a644849756333526c6247786863694973496e4e3159694936496d5268626d35355a6a4532514768766447316861577775593239744969776961574630496a6f784e54517a4f5445354f4455304c434a68645751694f694a7a6448497562325268496e302e70496474346b5763677546564d42486b6773484f306a6d5f536d556b6a33586e574946527041516f794f6f'}

# response = requests.get("http://stat-xplore.dwp.gov.uk/webapi/rest/v1/schema/str:valueset:AA_In_Payment:F_AA_QTR:DATE_NAME:C_AA_QTR", headers = api)



# data = response.json()
def query(url, payload, api_key):

    """

    Method for calling the Stat-Xplore API.



    Args:

        url (str): The statxplore table endpoint

        payload (str): The query string

        api_key (str): API key for authentication



    Returns:

        response (requests.models.Response): The http response object

    """

    # This can take some time depending on the data size

    # NB: The Stat-Xplore API  caches responses, so repeat responses are rapid

    print('[{0}] Making Stat-Xplore API query...'.format(dt.now()))

    start_time = dt.now()



    response = requests.post(url, data=payload, headers={'APIKey': api_key})



    time_taken = dt.now() - start_time

    print('[{0}] Response received after {1}'.format(dt.now(), time_taken))

    return response
def fetch_record_uris(items):

    """

    Simple function for extracting universal resource identifiers (URIs) from

    an API response field.



    This function is required (as opposed to the in-line approach used for

    labels) because 'totals' don't have a URI, and so need to be handled.



    Args:

        items (list of dict): dict of metadata for a record



    Returns:

        uri_list (list of str): list of URI strings

    """

    uri_list = []

    for i in items:

        try:

            uri = i['uris'][0]

            uri = ':'.join(uri.split(':')[-2:])

            uri_list += [uri]

        except KeyError:

            uri_list += [i['labels'][0]]

    return uri_list



def fetch_dimension_uri(item):

    """

    Simple function for extracting universal resource identifier (URI) headers

    for columns.



    Args:

        item (dict): dict of field metadata



    Returns:

        uri (str): column header URI name

    """

    try:

        uri = [item['uri']][0]

    except KeyError:

        uri = [item['label']][0]

    return uri



def extract_data(raw_response, index_mode='both'):

    """

    Function for extracting data from Stat-Xplore API JSON response.



    Args:

        raw_response (str): Json-formatted text containing API response

        index_mode (str): The variable naming scheme, human redable or codified

            Takes values ['label'|'code'|'both']



    Returns:

        data_frame (pandas.DataFrame): Narrow table of data

    """



    # Parse the string of JSON data into a dictionary, so it can be indexed

    print('[{0}] Parsing json from response...'.format(dt.now()))

    parsed_response = json.loads(raw_response)



    # Flatten n-dimensional data array from the API into a narrow dataframe

    print('[{0}] Extracting data from parsed response...'.format(dt.now()))

    values = (parsed_response

              ['cubes']

              [list(parsed_response['cubes'].keys())[0]]

              ['values']

              )

    values = np.array(values).flatten()

    data_frame = pd.DataFrame(values, columns=['measure'])



    # Iterate through data fields to build an index for the narrow data

    # Column index in human readable form (labels) and code form (uris)

    dimension_labels = []

    dimension_uris = []



    # Record index in human readable (label) and code (uri) forms

    label_indices = []

    uri_indices = []

    

    # Populate the indices

    for i in range(len(parsed_response['query']['dimensions'])):

        dimension_labels += [parsed_response['fields'][i]['label']]

        label_indices += [[record['labels'][0]

                           for record

                           in parsed_response['fields'][i]['items']

                           ]]

        dimension_uris += [fetch_dimension_uri(parsed_response['fields'][i])]

        uri_indices += [fetch_record_uris(parsed_response['fields']

                                          [i]

                                          ['items']

                                          )

                        ]



    # Build n-layered multi-indices (as the data has been flattened)

    label_index = pd.MultiIndex.from_product(label_indices,

                                             names=dimension_labels

                                             )

    code_index = pd.MultiIndex.from_product(uri_indices,

                                            names=dimension_uris

                                            )



    # Extract the measure names (percentage, amounts, counts, etc.)

    measure_label = parsed_response['measures'][0]['label']

    measure_uri = parsed_response['measures'][0]['uri']



    # Attach the indices to the data

    if (index_mode == 'label') | (index_mode == 'both'):

        data_frame[measure_label] = data_frame['measure']

        data_frame.index = label_index

        data_frame = data_frame.reset_index()



    if (index_mode == 'code') | (index_mode == 'both'):

        data_frame[measure_uri] = data_frame['measure']

        data_frame.index = code_index

        data_frame = data_frame.reset_index()



    # Drop the now-renamed measure field

    data_frame = data_frame.drop('measure', axis=1)



    # Sort the column names in 'both' mode so codes and labels are next

    # to one another

    if (index_mode == 'both'):

        data_frame = data_frame[np.array(list(zip(dimension_labels,

                                                  dimension_uris

                                                  )

                                              )

                                         )

                                .flatten()

                                .tolist()

                                + [measure_label, measure_uri]

                               ]

        

    # Finally, return the result!

    return data_frame
endpoint_url = 'https://stat-xplore.dwp.gov.uk/webapi/rest/v1/table'
payload = {

  "database" : "str:database:NINO",

  "measures" : [ "str:count:NINO:f_NINO" ],

  "recodes" : {

    "str:field:NINO:f_NINO:NEWNAT" : {

      "map" : [ [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:1" ], [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:2" ], [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:3" ], [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:4" ] ],

      "total" : 'true'

    },

    "str:field:NINO:f_NINO:QTR" : {

      "map" : [ [ "str:value:NINO:f_NINO:QTR:c_CALYR:16" ], [ "str:value:NINO:f_NINO:QTR:c_CALYR:17" ] ],

      "total" : 'true'

    }

  },

  "dimensions" : [ [ "str:field:NINO:f_NINO:NEWNAT" ], [ "str:field:NINO:f_NINO:QTR" ] ]

}
# raw_response = query(endpoint_url, payload, api_key)



# def check_response(raw_response):

#     while raw_response.ok == False:

#         print('[{0}] Response ERROR, code {1}.'

#               .format(dt.now(), raw_response.status_code)

#               )

#         print('[{0}] Preparing re-run...'.format(dt.now()))

#         raw_response = query(endpoint_url, payload, api_key)

#         #raw_response = raw_response.text

    

#     print('[{0}] Response OK.'.format(dt.now()))

#     raw_response = raw_response.text

#     return raw_response

# # raw_response = raw_response.text

# # df = extract_data(raw_response,'both')

# # print('[{0}] Done.'.format(dt.now()))

# raw_response = query(endpoint_url, payload, api_key)

# raw_response = check_response(raw_response)
# payload = {

#   "database" : "str:database:NINO",

#   "measures" : [ "str:count:NINO:f_NINO" ],

#   "recodes" : {

#     "str:field:NINO:f_NINO:NEWNAT" : {

#       "map" : [ [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:1" ], [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:2" ], [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:3" ], [ "str:value:NINO:f_NINO:NEWNAT:C_NINO_WORLDAREA:4" ] ],

#       "total" : 'true'

#     },

#     "str:field:NINO:f_NINO:QTR" : {

#       "map" : [ [ "str:value:NINO:f_NINO:QTR:c_CALYR:16" ], [ "str:value:NINO:f_NINO:QTR:c_CALYR:17" ] ],

#       "total" : 'true'

#     }

#   },

#   "dimensions" : [ [ "str:field:NINO:f_NINO:NEWNAT" ], [ "str:field:NINO:f_NINO:QTR" ] ]

# }

# payload.strip()
# url = 'https://stat-xplore.dwp.gov.uk/webapi/rest/v1/table'

# requests.post(url, data=payload, headers={'APIKey': api_key})

# labels = []

# for i in range(len(data['children'])):

#     if 'QTR' in data['children'][i]['location']:

#         labels.append(data['children'][i]['location'][-6:])

#         labels = labels[-8:]

        

#     else:

#         labels.append(data['children'][i]['location'][-6:])

#         labels = labels[-12:]

# labels
# if 'QTR' in data['children'][0]['location']:

#     labels = [data['children'][i]['location'][-6:] for i in range(len(data['children']))][-8:]

# else:

#     labels = [data['children'][i]['location'][-6:] for i in range(len(data['children']))][-12:]

# labels
# def custom_request(database, measures, recodes):

#     json_request = {'database:str:database:'+ database,

#                     'measures:' + ['str:count:'+ database + measures],

#                     'recodes:' + {'str:field:'+ database + measures + ':' + recodes},

#                     'dimensions:' + [['str:field:'+ database + ':' + measures + ':' + recodes]]}

#     return json_request

# custom_request('PIP_Monthly', 'V_F_PIP_MONTHLY', 'SEX')
# body_req = {"database":"str:database:PIP_Monthly",

#             "measures":["str:count:PIP_Monthly:V_F_PIP_MONTHLY"],

#             "recodes":{"str:field:PIP_Monthly:V_F_PIP_MONTHLY:SEX":{"total":'true'}},

#             "dimensions":[["str:field:PIP_Monthly:V_F_PIP_MONTHLY:SEX"]]}

# req = requests.post('https://stat-xplore.dwp.gov.uk/webapi/rest/v1/table', 

#                     data = json.dumps(body_req) ,headers = api)

# df = req.json()

# df
# df.keys()

# vals = df['cubes']['str:count:PIP_Monthly:V_F_PIP_MONTHLY']['values']

# vals1 = pd.DataFrame([i for i in vals[0]])

# vals1.plot(kind = 'bar')
json_df = pd.read_json('../input/GBmap5.json')

regions = ['North East', 'North West', 'Yorkshire', 'East Midlands', 'West Midlands',

          'East of England', 'London', 'South East','South West', 'Wales',

          'Scotland', 'All']
# def get_uk_data(json_df, years):

#     """

#         Function to get correct UK data from geojson

#         that we have formatted in DWP

        

#         Input: json file, region labels

        

#         Returns: dictionary with region labels and values

#     """

#     regions = ['North East', 'North West', 'Yorkshire', 'East Midlands', 'West Midlands',

#           'East of England', 'London', 'South East','South West', 'Wales',

#           'Scotland', 'All']

#     region_dict = dict((el,0) for el in regions)

#     for region in range(len(regions)):            

#         array = []

#         for i in range(len(json_df.buckets[0]['size']) - 2):

#             #print(json_df.buckets[0]['size'][i][region])

#             array.append(json_df.buckets[0]['size'][i][region])

#             #region_dict[regions[region]].append(json_df.buckets[0]['size'][i][region])

#             #counter += json_df.buckets[0]['size'][i][j]

#         region_dict[regions[region]] = array

#     # deal with total and sum of last 5 years

#         region_dict[regions[region]].append({'last ' + str(years)+ ' years': sum(region_dict[regions[region]][-years:]), 'Total': sum(region_dict[regions[region]])})

    

#     return region_dict

# regions_json = get_uk_data(json_df, regions, 5)

# regions_json
def get_data(bucket_number, json_df, years):

    """

        Function to get correct data from geojson

        that we have formatted in DWP: extract data 

        for all regions in all years for specific country

        

        Input:  bucket_number: which country we choose

                json file, region labels

                

        

        Returns: dictionary with region labels and values

                count of Nino registrations for each region

                of the UK. 

    """

    regions = ['North East', 'North West', 'Yorkshire', 'East Midlands', 'West Midlands',

          'East of England', 'London', 'South East','South West', 'Wales',

          'Scotland', 'All']

    region_dict = dict((el,0) for el in regions)

    for region in range(len(regions)):            

        region_dict[regions[region]] = [json_df.buckets[bucket_number]['size'][i][region] for i in range(len(json_df.buckets[bucket_number]['size']) - 2)]

    

        region_dict[regions[region]].append({'last ' + str(years)+ ' years': sum(region_dict[regions[region]][-years:]), 

                                         'Total': sum(region_dict[regions[region]])})

    return region_dict



# loop through all countries and call the function

#data_list = [ {json_df.buckets[i]['name'] : get_data(i, json_df, regions, 5)} for i in range(len(json_df.buckets))]
def get_all_data(json_df):

    data_list = [ {json_df.buckets[i]['name'] : get_data(i, json_df, 5)} for i in range(len(json_df.buckets))]

    return data_list

data_list = get_all_data(json_df)
# json_df.to_json('GBmap5.json')

import json

with open('world_data.json', 'w') as f:

    json.dump(data_list, f)

#data_list.to_json("world_data.csv")
# json_df.buckets[0]['size']

# json_df.buckets[0]['size'][0]
# # loop through outer array

# master_array = []

# other_array = []

# for j in range(len(json_df.buckets[0]['size'][0])):

#     # loops through inner arrays

#     for i in range(len(json_df.buckets[0]['size'])):

#         #print(json_df.buckets[0]['size'][i][j])

#         other_array.append(json_df.buckets[0]['size'][i][j])





# new_array_set = np.array(other_array).reshape(12,18)
# #save to csv files

# for arr in range(len(json_df.buckets[0]['size'][0])):

#     array_save = pd.DataFrame(new_array_set[arr].reshape(1,18))

#     array_save.to_csv('arr_'+ str(arr) +'.csv')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import requests

import json

print(os.listdir("../input"))

from datetime import datetime as dt





# Make a get request to get the latest position of the international space station from the opennotify api.

apiKey = {'APIKey': '65794a30655841694f694a4b563151694c434a68624763694f694a49557a49314e694a392e65794a7063334d694f694a7a644849756333526c6247786863694973496e4e3159694936496d5268626d35355a6a4532514768766447316861577775593239744969776961574630496a6f784e54517a4f5445354f4455304c434a68645751694f694a7a6448497562325268496e302e70496474346b5763677546564d42486b6773484f306a6d5f536d556b6a33586e574946527041516f794f6f'}



response = requests.get("https://stat-xplore.dwp.gov.uk/webapi/rest/v1/schema", headers = apiKey)



# Print the status code of the response.

print(response.status_code)
df = pd.read_csv("../input/datagb/GBmap6.csv")

df.head()
#years = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

country_list = list(df.country.unique())

del country_list[1]

counter = 0



import itertools

n = 12

country_col = list(itertools.chain.from_iterable(itertools.repeat(x, n) for x in country_list))

del df["country"]

df["country"] = country_col
temp = df.to_dict(orient='records')

df.replace({'..': 0}, inplace = True)

df.head()
new_list = []

new_dict = {}

counter = 0

for i in range(len(temp)):

    country = temp[counter]["country"]

    temp[i]



temp2 = pd.melt(df, id_vars = ['region', "country"], var_name = "total")

temp2['value'] = temp2["value"].astype(int)

temp2.head()
df2 = pd.pivot_table(temp2, 'value', ['country', 'total'], 'region')

df2.reset_index(inplace = True)

df2.sort_values(by = "country")

df2[(df2.country == "Afghanistan") & (df2.total == "2018/19")]





columns = ["country", "total", "North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands", "East of England", "London",

          "South East", "South West", "Wales", "Scotland", "Total"]



df3 = df2[columns].copy()

df3.head()
json_dict = df3.to_dict(orient='records')

json_dict
df = pd.read_csv("../input/datagb/GBmap6.csv")





def transform_data(df):

    """

    

    """

    country_list = list(df.country.unique())

    del country_list[1]

    counter = 0



    import itertools

    n = 12

    country_col = list(itertools.chain.from_iterable(itertools.repeat(x, n) for x in country_list))

    del df["country"]

    df["country"] = country_col



    df.replace({'..': 0}, inplace = True)

    temp2 = pd.melt(df, id_vars = ['region', "country"], var_name = "total")

    temp2['value'] = temp2["value"].astype(int)



    df2 = pd.pivot_table(temp2, 'value', ['country', 'total'], 'region')

    df2.reset_index(inplace = True)

    df2.sort_values(by = "country")

    df2[(df2.country == "Afghanistan") & (df2.total == "2018/19")]





    columns = ["country", "total", "North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands", "East of England", "London",

              "South East", "South West", "Wales", "Scotland", "Total"]



    df3 = df2[columns].copy()

    json_dict = df3.to_dict(orient='records')

    return json_dict



json_dict = transform_data(df)
json_dict[i]
def create_dict(json_dict):

    """

    

    """

    parent_dict = {}

    parent_dict["buckets"] = []

    child_dict = {"name": "", "series": []}

    parent_list = []

    child_list = []

    N = 18 # years



    for i in range(len(json_dict)): # country

        child_dict["name"] = json_dict[i]["country"]

        for region in list(json_dict[i].keys())[2:]:

            child_list.append(json_dict[i][region])

        child_dict["series"].append(child_list)

        child_list = []



        if i % N == 0 and i != 0:

            parent_dict["buckets"].append(child_dict)

            parent_list = []

            child_dict = {"name": "", "series": []}

            child_dict["name"] = json_dict[i]["country"]



    return parent_dict



new_dict = create_dict(json_dict)
new_dict["buckets"][0]
import json

json_data = json.dumps(new_dict)



with open('GBmap.json', 'w') as outfile:

    json.dump(json_data, outfile)