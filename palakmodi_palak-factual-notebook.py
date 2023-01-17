#Move the data file in input folder. List the files present in the input folder.
import os
print(os.listdir("../input"))
#Libraries
import json 
import pandas as pd 

#load input json file into an json list of json objects
json_array = []
for line in open('../input/data.json', 'r'):
    json_array.append(json.loads(line))
# iterate through json array and take the values of uuid and payload into 2 separate lists
uuid=[]
payload= []
for my_json_dict in json_array:
    uuid.append(my_json_dict['uuid'])
    payload.append(my_json_dict['payload'])
    
#merge payload and uuid into a single dataframe
location_df =pd.DataFrame(payload)
location_df['uuid']=uuid
#List the columns in the dataframe
location_df.columns
#Peek inside the data. location_df will be used as a base dataframe to solve all the questions
location_df.head()
#To find the rows and columns of the dataframe
location_df.shape
# 1. Return every unique locality (cf. city) along with how often it occurs

#Get the count for each unique locality using pandas function value_counts
location_df['locality'].value_counts()
#answer is the output of the above statement
# 2. Return all addresses that start with a number (return just the address)

#Initialize a list to hold numeric addresses
numeric_adresses=[]
#iterate through the list and drop null values
for temp in list(location_df['address'].dropna().values):
    #split the data to create a list of addresses and check if first letter of address is numeric
    if temp.split() and temp.split()[0][0].isnumeric():
        #add the unpadded address to final list of numeric adress
        numeric_adresses.append(temp.strip())
        
#Ans: numeric_address is the final list of all addresses that start with a number
numeric_adresses
# 3. Return all rows with addresses that don't contain a number (return the entire row)

#has_numbers check for each address, if it contains a number and returns a dictionary with string and boolean values
def hasNumbers(address_list):
    address_boolean_dict = {} 
    for inputString in address_list:
        inputString = str(inputString)
        if inputString!='':
            address_boolean_dict[inputString]=any(char.isdigit() for char in inputString)
    return address_boolean_dict
#create a list of addresses and drop null adresses
address_list=[]
for address in list(location_df['address'].dropna().values):
     address_list.append(address)
        
#call hasnumbers to find all the addresses which does not contain a number[Value is False]
address_boolean_dict = hasNumbers(address_list)

#return addresses which does not contain a number
alpha_address = []
for k in final_dict:
    if not final_dict[k]:
        alpha_address.append(k)
#check the addresses obtained
alpha_address
#copy the location_df to alpha_adresses_df. This will be used to fetch all rows with addresses that don't contain a number
alpha_adresses_df = location_df

#return all the rows with address value as addresses that does not contain a number
alpha_adresses_df = alpha_adresses_df.loc[alpha_adresses_df['address'].isin(alpha_address)]

#there are 379 addresses with adresses that does not contain a number
alpha_adresses_df.shape
# 4. Return the number of records that are museums

#Get the category label for each location record
s = location_df['category_labels']

#category label is a nested list. Get the category values out of the nested list using pd.series.stack().reset_index()
s = s.apply(pd.Series).stack().reset_index(drop=True)
s = s.apply(pd.Series).stack().reset_index(drop=True)
#get category count for each category_labels, there are 72 Museums in ccategory count
s.value_counts()
#Return the number of records that are museums
(s == 'Museums').sum()

#Ans is 72 location records are museums
# 5. Return a new object containing uuid, name, website, and email address for all rows that have values for all four of these attributes; exclude any rows that don’t have all four

#create a new object containing uuid, name, website, and email address and drop any rows which has NaN for any of the 4 values
sliced_df = location_df[['uuid','name','website','email']].dropna(axis='rows',how='any')
sliced_df.shape
# 6. Return all rows, but transform the names to all lower case while changing nothing else

#use str.lower() function on name column in location_df and replace it in the original df
location_df['name']=location_df['name'].str.lower()
#check the results
location_df.head()
#check if location names are all in lower case.
location_df['name']
# 7. Return all rows for businesses that open before 10:00 a.m. on Sundays

#Remove all records from location_df that have NaN hours and create a dataframe with uuid and hours
location_uuid_hours_df = location_df[['uuid','hours']].dropna()
location_uuid_hours_df.shape
#check the dataframe
location_uuid_hours_df
#code snippet to find uuids corresponding to businesses open before 10am on Sundays
business_sunday_before10_uuids = []
for values in location_uuid_hours_df.iterrows():
    # get the uuids and hours value in a list
    temp=list(values[1])
    # load the hour values in a dictionary
    hours_dict = json.loads(temp[1])
    #iterate over hours_dictionary
    for k in hours_dict:
        #for all keys = Sunday
        if k.lower() =="sunday":
            #print(obj[k][0][0])
            #fetch the opening hour(not the minutes, just the hour)
            opening_hour = int(hours_dict[k][0][0].split(':')[0])
            #fetch all uuids for which opening_hour<10 for Sundays
            if opening_hour<10:
                business_sunday_before10_uuids.append(temp[0])

#check the number of uuids for which opening_hour<10 for Sundays:there are 100 records
len(business_sunday_before10_uuids)
#using df.loc, filter the uuids from location_df matching to those in business_sunday_before10_uuids
business_sunday_before10_df = location_df.loc[location_df['uuid'].isin(business_sunday_before10_uuids)]
# all rows corresponding to matching uuids is in business_sunday_before10_df. For Sanity checking, check the shape of the returned df
#it should match the number of uuids returned by business_sunday_before10_uuids
business_sunday_before10_df.shape