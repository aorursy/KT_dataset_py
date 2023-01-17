import pandas as pd

from fuzzywuzzy import process

import time

df = pd.read_excel('/kaggle/input/brazilian_geolocation_dataset.xlsx')
df.info()
df['geolocation_city'].unique()
df['geolocation_state'].unique()
#get a list of cities which are not ASCII (or with the wrong font)

list = df[['geolocation_city','geolocation_state']][df['geolocation_city'].apply(lambda x: x.isascii())== False] 

list = list.drop_duplicates(subset='geolocation_city',keep = 'first')

#once we get an unique list for city which have a font problem, we get another unique list of cities which have correct names 

correct_list = df[['geolocation_city','geolocation_state']][df['geolocation_city'].apply(lambda x: x.isascii()) == True].drop_duplicates(subset = 'geolocation_city',keep='first')
#METHOD 1: Process a dataframe directly without dictionary list

start = time.time()

def correction1(element):

    state = element['geolocation_state']

    strOptions = correct_list['geolocation_city'][correct_list['geolocation_state'] == state].unique()

    closest_value = process.extractOne(element['geolocation_city'],strOptions) 

    element['geolocation_city'] = closest_value[0] #replace with the closest match from the correct list based on the same state

    return element['geolocation_city']

    

df['geolocation_city'] = df.apply(correction1,axis=1)

print("--- %s seconds ---" % (time.time() - start))
#METHOD 2: Extract dictionary list and replace value on dataframe

start = time.time()

dictonary_list = []

for index,element in list.iterrows():

    state = element[1]

    strOptions = correct_list['geolocation_city'][correct_list['geolocation_state'] == state].unique()

    closest_value = process.extractOne(element[0],strOptions)

    dictonary_list.append([element[0],closest_value[0]])

dictonary_list = pd.DataFrame(dictonary_list, columns= ['geolocation_city','correct_city_name']) #convert list into dataframe

#merge two dataframes based on column geolocation_city 

df = df.merge(dictonary_list, on=['geolocation_city'], how='left', suffixes=('_','')) 

#fill null value in a column correct_city_name with value from column geolocation_city

df['correct_city_name'] = df['correct_city_name'].fillna(df['geolocation_city']).astype(str)  

#drop the column geolocation_city with wrong city names and change the name for column correct_city_name

df = df.drop('geolocation_city', axis=1)

df = df.rename(columns={"correct_city_name":"geolocation_city"})

print("--- %s seconds ---" % (time.time() - start))
print(df['geolocation_city'][df['geolocation_city'].apply(lambda x: x.isascii())== False])