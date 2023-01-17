## Step 1. Import Library Dependencies

# Dependencies

import requests as req

import json

import pandas as pd

##print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# A. Get our base URL for the Open Weather API

base_url = "http://api.openweathermap.org/data/2.5/weather"



# B. Get our API Key 

key = "c703c966f9be8a0c4869b86832a0898f"



# C. Get our query (search filter)

query_city = "Los Angeles"

query_units = "metric"



# D. Combine everything into our final Query URL

query_url = base_url + "?apikey=" + key + "&q=" + query_city + "&units=" + query_units



# Display our final query url

query_url
# Perform a Request Call on our search query

response = req.get(query_url)

response
response = response.json()

response
# Using json.dumps() allows you to easily read the response output

print(json.dumps(response, indent=4, sort_keys=True))
# Extract the temperature data from our JSON Response

temperature = response['main']['temp']

print ("The temperature is " + str(temperature))



# Extract the weather description from our JSON Response

weather_description = response['weather'][0]['description']

print ("The description for the weather is " + weather_description)
# A. Get our base URL for the Open Weather API

base_url = "http://api.openweathermap.org/data/2.5/weather"



# B. Get our API Key 

key = "c703c966f9be8a0c4869b86832a0898f"



# C. Create an empty list to store our JSON response objects

weather_data = []



# D. Define the multiple cities we would like to make a request for

cities = ["London", "Paris", "Las Vegas", "Stockholm", "Sydney", "Hong Kong"]



# E. Read through each city in our cities list and perform a request call to the API.

# Store each JSON response object into the list

for city in cities:

    query_url = base_url + "?apikey=" + key + "&q=" + city

    weather_data.append(req.get(query_url).json())
# Now our weather_data list contains 6 different JSON objects for each city

# Print the first city (London) 

weather_data
# Create an empty list for each variable

city_name = []

temperature_data = []

weather_description_data = []



# Extract the city name, temperature, and weather description of each City

for data in weather_data:

    city_name.append(data['name'])

    temperature_data.append(data['main']['temp'])

    weather_description_data.append(data['weather'][0]['description'])



# Print out the list to make sure the queries were extracted 

print ("The City Name List: " + str(city_name))

print ("The Temperature List: " + str(temperature_data))

print ("The Weather Description List: " + str(weather_description_data))
# Extract the city name, temperature, and weather description of each City

city_name = [data['name'] for data in weather_data]

temperature_data = [data['main']['temp'] for data in weather_data]

weather_description_data = [data['weather'][0]['description'] for data in weather_data]



# Print out the list to make sure the queries were extracted 

print ("The City Name List: " + str(city_name))

print ("The Temperature List: " + str(temperature_data))

print ("The Weather Description List: " + str(weather_description_data))
# Create a dictionary containing our newly extracted information

weather_data = {"City": city_name, 

                "Temperature": temperature_data,

                "Weather Description": weather_description_data}
# Convert our dictionary into a Pandas Data Frame

weather_data = pd.DataFrame(weather_data).reset_index()

weather_data