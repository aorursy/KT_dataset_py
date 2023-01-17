citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
print(citytracker.get('Chicago'),str("live in Chicago"))
#Assigning the variable 's' to the output of adding s. This answer is incorrect.  

s = citytracker['Seattle'] + int(17500)

print (s, str('live in the Seattle area'))



# This assigns the dictionary key-pair the new value created by adding 17,500

#ONLY RUN ONCE OR IT WILL KEEP ADDING 17500 each time its executed 

citytracker['Seattle'] = citytracker['Seattle'] + int(17500)

if citytracker['Seattle'] > 742225:

    citytracker['Seattle'] - int(17500)





print (citytracker)

    
#Defines the Key LA and assigns its Pair 45000

citytracker['Los Angeles'] = "45000"

#Prints the Dictionary line by line

for x in citytracker:

    print (x, citytracker[x])
#Prints a string then the dictionary's Key pair 



print('Denver:', citytracker['Denver'])
#Prints the Dictionary line by line

for x in citytracker:

    print (x, citytracker[x])
#A for loop using the variable 'a' to a temporary variable 

#prints 'a' and the corresponding key pair to 'a' as it cycles eache line

for a in citytracker:

    print (a,':', citytracker[a])
# 'a' takes user input 

print ('User input: ')

a = input()

#for a in citytracker:

#print(a)



#Compares the 'a' value to the keys and pair associated 

if a == 'New York' in citytracker:

    print('New York: ', citytracker['New York'])      

elif a == 'Atlanta' in citytracker:

    print ('Atlanta: ', citytracker['Atlanta'])

# If they are neither NY or Atlanta the else statement is triggered 

else:

    print ('Sorry that location is not in the City Tracker')
#Incorrect DO NOT GRADE 

# Attempts to solve the problem in a FOr loop



# 'a' takes user input 

#a = input()

#for a in citytracker:

#    #print(a)

#    if 'New York' in citytracker:

#        print('New York: ', citytracker['New York'])

#        

    #elif 'Atlanta' in citytracker:

        #print ('Atlanta: ', citytracker['Atlanta'])

#    else:

#        print ('Sorry that location is not in the City Tracker')
#Dictionary 

citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}





#PSEUDOCODE 

# CHECK potentialcities against citytracker





#IF TRUE print City name and population number 



# If FALSE print city name and print Zero 

# Print Zero using the Default Values ()



# potentialcities.get(key,default_val)

# potentialcities.get(key,0)





#TRUE DATA 

potentialcities = ['Cleveland', 'Atlanta', 'Phoenix','Nashville','Philadelphia','Milwaukee', 'Seattle']



#FALSE DATA 

# potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee' ]



temp_city = potentialcities 





# Test prints

#print (d)

#print (temp_city)

#print (potentialcities)



for pc in potentialcities:

    

    if pc == citytracker.get(pc):

        print(pc, "test1")

    

    elif pc != citytracker.get(pc):

        citytracker.setdefault(pc,0)

        print (pc, citytracker[pc])

    

    else:

        

        print (pc, "test2")

        

        

        

#d ={}

#potentialcities["Cleveland"] = 0

#print ("Cleveland" in potentialcities)



# Correct Syntax of default dict values 

# dictionary.setdefault(keyname, value)

# potentialcities

# Correct Syntax of default dict values 

# dictionary.setdefault(keyname, value)



#d = {}



#TRUE DATA 

#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee', 'Seattle']



#FALSE DATA 

# potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee' ]



#d = potentialcities 



#print (d)

#print (potentialcities)



#for pc in potentialcities:

#    citytracker.get(pc, 0)

#    print(citytracker[pc])

#    if pc == citytracker.get(pc):

#        print(pc, "test1")

#    else:

#        citytracker.get(pc, 0)

#        print (pc, "test2")

        



        

#d ={}

#potentialcities["Cleveland"] = 0

#print ("Cleveland" in potentialcities)
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']

pc = potentialcities



#for pc in potentialcities:

# If True print city name and population

    

   # if pc == citytracker:

       # print (citytracker, citytracker.get(pc,0))



  #  elif pc != citytracker:

       # print (pc,potentialcities.get(pc,0))



#print (pc, potentialcities.get("pc",0))
#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



#Compare Potential Cities against City Tracker 



#pc = potentialcities



#for pc in citytracker:

# If True print city name and population

    

#    if pc == citytracker:

#        print (pc, citytracker.get(pc,0))



#    elif pc != citytracker:

#        print (pc, )

#        

#        # If False print city name and default population to zero 

#    else:

#        print (pc, potentialcities.get("pc",0))

        

        













#print()



#citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



# 'a' takes user input 



# 



#p = potentialcities

#for p in citytracker:

    #print(p)



#Compares the 'p' value to the keys and pair associated 

#    if p == 'New York' in citytracker:

#        print('New York: ', citytracker['New York'])      

#    elif p == 'Atlanta' in citytracker:

#        print ('Atlanta: ', citytracker['Atlanta'])

    # If they are neither NY or Atlanta the else statement is triggered 

#    else:

#        print ('Sorry that location is not in the City Tracker')



# def student(firstname, lastname ='Mark', standard ='Fifth'): 

# 	print(firstname, lastname, 'studies in', standard, 'Standard') 



# 1 positional argument 

#student('John') 



# 3 positional arguments						 

#student('John', 'Gates', 'Seventh')	 



# 2 positional arguments 

#student('John', 'Gates')				 

#student('John', 'Seventh') 



   

#A for loop using the variable 'a' to a temporary variable 

#prints 'a' and the corresponding key pair to 'a' as it cycles eache line

for a in citytracker:

    #Printing off the key and joining by the ',' followed by the corresponding population size 

    #print (','.join(citytracker.keys()), citytracker[a])

    print (a,':', citytracker[a])

    

import os



import csv 





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here





# Write key and value pairs from citytracker out to a file named 'popreport.csv'.

# Hint: the procedure is very close to that of Step 9. 



# citytracker is dictionary declared earlier 

# THe w indicates we are writing the csv file 



w = csv.writer(open("popreport.csv", 'w'))

w.writerow(['city', 'pop'])

for key,val in citytracker.items():

#    w.writeheader([city,pop])

    w.writerow([key,val])





# You should also include a header to row describe each column, labeling them as "city" and "pop", and subsequent lines should contain the data. 

    









import os



import csv 





### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

### Add your code here





# Write key and value pairs from citytracker out to a file named 'popreport.csv'.

# Hint: the procedure is very close to that of Step 9. 



# citytracker is dictionary declared earlier 

# THe w indicates we are writing the csv file 



w = csv.writer(open("popreport.csv", 'w'))

for key,val in citytracker.items():

    w.writerow([key,val])





# You should also include a header to row describe each column, labeling them as "city" and "pop", and subsequent lines should contain the data. 

    









# Documented Example to get the header in the file





import csv



myDic = { "a" : 1, "b" : 2, "c" : 15}

with open('myFile.csv', 'w', newline='') as csvfile:

    fieldnames = ['word', 'count']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)



    writer.writeheader()

    for key in myDic:

        writer.writerow({'word': key, 'count': myDic[key]})
import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_opencage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("api_openweather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



geocoder = OpenCageGeocode(secret_value_0)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print (f"{query} is located at:")

print (f"Lat: {lat}, Lon: {lng}")





def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib2.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



def getForecast(city="Seattle"):

    key = secret_value_1

    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

    print(url)

    return safeGet(url)



data = json.load(getForecast())

print(data)

current_time = datetime.datetime.now() 



print(f"The current weather in Seattle is: {data['weather'][0]['description']}")

print("Retrieved at: %s" %current_time)



### You can add your own code here for Steps 2 and 3
#Active TEST SNIPPET





# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_opencage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("api_openweather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']





#Declares a new copy of the citytracker dictionary 

#new_citytracker = citytracker.copy() 



new_citytracker = {} 





#NEED to add third key to new_citytracker such that the key-pair can be written by the API input





city_details = {}





#Creates a for Loop which prints of data from all



for temp_city in citytracker:

    geocoder = OpenCageGeocode(secret_value_0)

    query = 'temp_city'  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

       

 

    

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

  #  print (f"{temp_city} is located at:")

  #  print (f"Lat: {lat}, Lon: {lng}")

    #print (temp_city) Testing line 

    



    def safeGet(url):

        try:

            return urllib.request.urlopen(url)

        except urllib2.error.URLError as e:

            if hasattr(e,"code"):

                print("The server couldn't fulfill the request.")

                print("Error code: ", e.code)

            elif hasattr(e,'reason'):

                print("We failed to reach a server")

                print("Reason: ", e.reason)

            return None



    def getForecast(city="query"):

        key = secret_value_1

        url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

        print(url)

        return safeGet(url)



    #Data is the variable data 

    data = json.load(getForecast())

   



    for d in citytracker:

        #print (d)



        #Getting the original population from 

        pop = citytracker[d]

        temp_city = new_citytracker

        

        new_citytracker[d] = {'population':pop, 'forecast': new_citytracker}

        print(new_citytracker)



    for tempcity in citytracker:

        yourupdatednamefordata= json.load(getForecast(tempcity)) 

        citytracker[tempcity]=[citytracker[tempcity],yourupdatednamefordata] 

    

    

    

    #print(data)

    #for key in citytracker:

    #    new_citytracker[key] = {'city':citytracker[key], "API_Data": {city_details}}  #appends the dictionary 

    

    

    #for key in orig_dict: #got though they keys in the original dictionary

    #create a new entry in the new dictionary based on data from original dictionary and new data

    #new_dict[key] = {"number": orig_dict[key], "new_information": {"length": randint(0,10), "width":randint(0,10)}}

    

    

    

    #SAVE DATA INTO THE NEW DICTIONARY 

    #type(data)

    #city_details = data

    #new_citytracker['API_data'] = city_details  #appends the dictionary 







    

    

    



        

    current_time = datetime.datetime.now() 

    

    print(f"The current weather in %s is: {data['weather'][0]['description']}" %temp_city)

    print("Retrieved at: %s" %current_time)



    

    ### You can add your own code here for Steps 2 and 3

















# print ("This is a test line", city_details)

#print (new_citytracker)



    

#Exports Data into citydata.txt



    

with open('citydata.json', 'w') as outfile:

    json.dump(data,outfile)

    

for dirname, _, filenames in os.walk ('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname,filename))

#Almost Working TEST SNIPPET





# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_opencage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("api_openweather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']





#Declares a new copy of the citytracker dictionary 

#new_citytracker = citytracker.copy() 



new_citytracker = {} 

#NEED to add third key to new_citytracker such that the key-pair can be written by  





city_details = {}





#Creates a for Loop which prints of data from all



for temp_city in citytracker:

    geocoder = OpenCageGeocode(secret_value_0)

    query = 'temp_city'  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (f"{temp_city} is located at:")

    print (f"Lat: {lat}, Lon: {lng}")

    #print (temp_city) Testing line 

    



    def safeGet(url):

        try:

            return urllib.request.urlopen(url)

        except urllib2.error.URLError as e:

            if hasattr(e,"code"):

                print("The server couldn't fulfill the request.")

                print("Error code: ", e.code)

            elif hasattr(e,'reason'):

                print("We failed to reach a server")

                print("Reason: ", e.reason)

            return None



    def getForecast(city="query"):

        key = secret_value_1

        url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

        print(url)

        return safeGet(url)



    #Data is the variable for data 

    data = json.load(getForecast())

   



    for d in citytracker:

        #print (d)



        #Getting the original population from 

        pop = citytracker[d]

        

        new_citytracker[d] = {'population':pop, 'forecast': new_citytracker}

        print(new_citytracker)



    for temp_city in citytracker:

        yourupdatednamefordata= json.load(getForecast(tempcity)) 

        citytracker[temp_city]=[citytracker[temp_city],yourupdatednamefordata]

    

    

    

    #print(data)

    #for key in citytracker:

    #    new_citytracker[key] = {'city':citytracker[key], "API_Data": {city_details}}  #appends the dictionary 

    

    

    #for key in orig_dict: #got though they keys in the original dictionary

    #create a new entry in the new dictionary based on data from original dictionary and new data

    #new_dict[key] = {"number": orig_dict[key], "new_information": {"length": randint(0,10), "width":randint(0,10)}}

    

    

    

    #SAVE DATA INTO THE NEW DICTIONARY 

    #type(data)

    city_details = data

    new_citytracker['API_data'] = city_details  #appends the dictionary 

   

        

    current_time = datetime.datetime.now() 

    

    print(f"The current weather in %s is: {data['weather'][0]['description']}" %temp_city)

    print("Retrieved at: %s" %current_time)



    

    ### You can add your own code here for Steps 2 and 3

















# print ("This is a test line", city_details)

#print (new_citytracker)



    

#Exports Data into citydata.txt



    

with open('citydata.json', 'w') as outfile:

    json.dump(data,outfile)

    

for dirname, _, filenames in os.walk ('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname,filename))

from random import * #just to have access to random functions

orig_dict = {"key1":1, "key2": 2, "key3": 3} #original dictionary

new_dict = {} # new dictionary you want to create

for key in orig_dict: #got though they keys in the original dictionary

    #create a new entry in the new dictionary based on data from original dictionary and new data

    new_dict[key] = {"number": orig_dict[key], "new_information": {"length": randint(0,10), "width":randint(0,10)}}

#show the new dictionary

print(orig_dict)

print(new_dict)
#new_citytracker = {"city":[], "forecast":[]}



new_citytracker = {}

citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}



for d in citytracker:

    print (d)

    

    #Getting the original population from 

    pop = citytracker[d]

    

    

    new_citytracker[d] = {'population':pop, 'forecast': {}}

       

    #new_citytracker['city'].append(d)

    #new_citytracker['forecast'].append(forecast)

    

    

    #forecast = {''}

    #new_citytracker['city'].append(d)

    #new_citytracker['forecast'].append(forecast)

    

print(new_citytracker)
#Old working TEST SNIPPET





# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_opencage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("api_openweather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']





#Declares new dictionary 

diction = {}





#Creates a for Loop which prints of data from all



for temp_city in citytracker:



    geocoder = OpenCageGeocode(secret_value_0)

    query = 'temp_city'  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (f"{temp_city} is located at:")

    print (f"Lat: {lat}, Lon: {lng}")

    #print (temp_city) Testing line 

    



    def safeGet(url):

        try:

            return urllib.request.urlopen(url)

        except urllib2.error.URLError as e:

            if hasattr(e,"code"):

                print("The server couldn't fulfill the request.")

                print("Error code: ", e.code)

            elif hasattr(e,'reason'):

                print("We failed to reach a server")

                print("Reason: ", e.reason)

            return None



    def getForecast(city="query"):

        key = secret_value_1

        url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

        print(url)

        return safeGet(url)



    data = json.load(getForecast())

    print(data)

    

#SAVE DATA INTO THE DICTIONARY 

    #type(data)

    diction = data

    

    

        

    current_time = datetime.datetime.now() 



    print(f"The current weather in %s is: {data['weather'][0]['description']}" %temp_city)

    print("Retrieved at: %s" %current_time)



    

    ### You can add your own code here for Steps 2 and 3



    

print (diction)



    

#Exports Data into citydata.txt



    

with open('citydata.txt', 'w') as outfile:

    json.dump(data,outfile)

# Backup Snippet 

import urllib.error, urllib.parse, urllib.request, json, datetime



# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_opencage") #replace "openCageKey" with the key name you created!

secret_value_1 = user_secrets.get_secret("api_openweather") #replace "openweathermap" with the key name you created!



from opencage.geocoder import OpenCageGeocode



citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}

#potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']



for temp_city in citytracker:



    geocoder = OpenCageGeocode(secret_value_0)

    query = 'temp_city'  # replace this city with cities from the names in your citytracker dictionary

    results = geocoder.geocode(query)

    lat = str(results[0]['geometry']['lat'])

    lng = str(results[0]['geometry']['lng'])

    print (f"{temp_city} is located at:")

    print (f"Lat: {lat}, Lon: {lng}")

    print (temp_city)





def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib2.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



def getForecast(city="query"):

    key = secret_value_1

    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key

    print(url)

    return safeGet(url)



data = json.load(getForecast())

print(data)

current_time = datetime.datetime.now() 



print(f"The current weather in %s Seattle is: {data['weather'][0]['description']}" %temp_city)

print("Retrieved at: %s" %current_time)



### You can add your own code here for Steps 2 and 3


