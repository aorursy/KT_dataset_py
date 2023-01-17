# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Libraries to access the data.medicare.gov API

!pip install sodapy #package to access general API

from sodapy import Socrata #Socrata is the API



# Libraries for plotting graphs

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pyplot import figure
# this code comes from: https://dev.socrata.com/foundry/data.medicare.gov/4pq5-n9py





# Unauthenticated client only works with public data sets. Note 'None'

# in place of application token, and no username or password:

client = Socrata("data.medicare.gov", None)



# Example authenticated client (needed for non-public datasets):

# client = Socrata(data.medicare.gov,

#                  MyAppToken,

#                  userame="user@example.com",

#                  password="AFakePassword")



# First 20,000 results, returned as JSON from API / converted to Python list of (nb: This data set has ~15.5k records)

# dictionaries by sodapy.

results = client.get("4pq5-n9py",limit=20000) # 4pq5-n9py is the serial for the nursing home providers data set



# Convert to pandas DataFrame

NH_Providers = pd.DataFrame.from_records(results)
NH_Providers.head(n=5) # looking at the data

NH_Providers.info() # looking the columns (number, name, data type, etc)

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k') # make the graph bigger



NH_Providers.provider_state.value_counts().plot(kind='bar') 
NH_Providers['type'] = "nursing_home"

NH_Providers.info()
NH_Providers_contact = NH_Providers[['provider_name','provider_address','provider_city','provider_state','provider_zip_code','provider_county_name',

                                             'provider_phone_number','type']]





NH_Providers_contact['provider_county_name'] = NH_Providers_contact['provider_county_name'].str.upper() 
NH_Providers_contact['provider_phone_number'] = NH_Providers_contact['provider_phone_number'].str.replace("(","")

NH_Providers_contact['provider_phone_number'] = NH_Providers_contact['provider_phone_number'].str.replace(")","")

NH_Providers_contact['provider_phone_number'] = NH_Providers_contact['provider_phone_number'].str.replace("-","")

NH_Providers_contact['provider_phone_number'] = NH_Providers_contact['provider_phone_number'].str.replace(" ","")
NH_Providers_contact.head(n=5)
NH_Providers_contact.isna().sum()
del NH_Providers
# Unauthenticated client only works with public data sets. Note 'None'

# in place of application token, and no username or password:

client = Socrata("data.medicare.gov", None)



# Example authenticated client (needed for non-public datasets):

# client = Socrata(data.medicare.gov,

#                  MyAppToken,

#                  userame="user@example.com",

#                  password="AFakePassword")



# First 10,000 results, returned as JSON from API / converted to Python list of (nb: This data set has ~7.5k records)

# dictionaries by sodapy.

results = client.get("23ew-n7w9",limit=10000)



# Convert to pandas DataFrame

D_Facilities = pd.DataFrame.from_records(results)
D_Facilities.info()
list(D_Facilities.columns)
pd.set_option('display.max_rows', None) # There are 118 columns so this allows all of the to be displayed



D_Facilities.isna().sum() # Looking for NaN values
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k') #make the plot bigger



D_Facilities.state.value_counts().plot(kind='bar')
D_Facilities['type'] = "dialysis_center"

D_Facilities.info()
D_Facilities_contact = D_Facilities[['facility_name','address_line_1','city','state','zip','county','phone_number','type']]
D_Facilities_contact['county'] = D_Facilities_contact['county'].str.upper() 
D_Facilities_contact['phone_number'] = D_Facilities_contact['phone_number'].str.replace("(","")

D_Facilities_contact['phone_number'] = D_Facilities_contact['phone_number'].str.replace(")","")

D_Facilities_contact['phone_number'] = D_Facilities_contact['phone_number'].str.replace("-","")

D_Facilities_contact['phone_number'] = D_Facilities_contact['phone_number'].str.replace(" ","")
D_Facilities_contact.info()
D_Facilities_contact.head(n=20)
D_Facilities_contact.isna().sum()
pd.set_option('display.max_rows', 50)

D_Facilities_contact[D_Facilities_contact.isna().any(axis=1)]
del D_Facilities
# Unauthenticated client only works with public data sets. Note 'None'

# in place of application token, and no username or password:

client = Socrata("data.medicare.gov", None)



# Example authenticated client (needed for non-public datasets):

# client = Socrata(data.medicare.gov,

#                  MyAppToken,

#                  userame="user@example.com",

#                  password="AFakePassword")



# First 1000 results, returned as JSON from API / converted to Python list of (nb: This data set has ~.37k records)

# dictionaries by sodapy.

results = client.get("azum-44iv",limit=1000)



# Convert to pandas DataFrame

LTC_Facilities = pd.DataFrame.from_records(results)
LTC_Facilities.info()
LTC_Facilities.isna().sum()
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k') #make the plot bigger



LTC_Facilities.state.value_counts().plot(kind='bar')
LTC_Facilities['type'] = "ltc_facility"

LTC_Facilities.info()
LTC_Facilities_contact = LTC_Facilities[['facility_name','address_line_1','city','state','zip_code','county_name','phonenumber','type']]
LTC_Facilities_contact['county_name'] = LTC_Facilities_contact['county_name'].str.upper() 
LTC_Facilities_contact['phonenumber'] = LTC_Facilities_contact['phonenumber'].str.replace("(","")

LTC_Facilities_contact['phonenumber'] = LTC_Facilities_contact['phonenumber'].str.replace(")","")

LTC_Facilities_contact['phonenumber'] = LTC_Facilities_contact['phonenumber'].str.replace("-","")

LTC_Facilities_contact['phonenumber'] = LTC_Facilities_contact['phonenumber'].str.replace(" ","")
del LTC_Facilities
# Unauthenticated client only works with public data sets. Note 'None'

# in place of application token, and no username or password:

client = Socrata("data.medicare.gov", None)



# Example authenticated client (needed for non-public datasets):

# client = Socrata(data.medicare.gov,

#                  MyAppToken,

#                  userame="user@example.com",

#                  password="AFakePassword")



# First 1000 results, returned as JSON from API / converted to Python list of (nb: This data set has ~5.7k records)

# dictionaries by sodapy.

results = client.get("xubh-q36u",limit=10000)



# Convert to pandas DataFrame

Hospitals = pd.DataFrame.from_records(results)
Hospitals.info()
Hospitals['type'] = "hospital"
Hospitals_contact = Hospitals[['hospital_name','address','city','state','zip_code','county_name','phone_number','type']]
Hospitals_contact['phone_number'] = Hospitals_contact['phone_number'].str.replace("(","")

Hospitals_contact['phone_number'] = Hospitals_contact['phone_number'].str.replace(")","")

Hospitals_contact['phone_number'] = Hospitals_contact['phone_number'].str.replace("-","")

Hospitals_contact['phone_number'] = Hospitals_contact['phone_number'].str.replace(" ","")
Hospitals_contact.head(n=5)
NH_Providers_contact = NH_Providers_contact.rename(columns={'provider_address':'address','provider_city':'city','provider_state':'state','provider_zip_code':'zip','provider_county_name':'county',

                                             'provider_phone_number':'phone_number',})

NH_Providers_contact.info()
D_Facilities_contact = D_Facilities_contact.rename(columns={'facility_name':'provider_name','address_line_1':'address'})           



D_Facilities_contact.info()
LTC_Facilities_contact = LTC_Facilities_contact.rename(columns={'facility_name':'provider_name','address_line_1':'address','zip_code':'zip','county_name':'county',

                                                        'phonenumber':'phone_number'})



LTC_Facilities_contact.info()
Hospitals_contact.info()
Hospitals_contact = Hospitals_contact.rename(columns={'hospital_name':'provider_name','zip_code':'zip','county_name':'county',

                                                })



Hospitals_contact.info()
Output=pd.concat([NH_Providers_contact, D_Facilities_contact, LTC_Facilities_contact, Hospitals_contact], axis=0, join='inner', ignore_index=False, keys=None,

          levels=None, names=None, verify_integrity=False, copy=True, sort=False)



Output = Output.sort_values(by=['state','county', 'city'])
Output.head(n=10)
Output.info()
Output.to_csv('Medicare Consolidated Contact Info.csv', index = False, header=True)