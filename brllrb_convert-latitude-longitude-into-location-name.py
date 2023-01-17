### Bikram Maharjan
### Github: https://github.com/brlrb
### Created for Data Visualization course project @ https://unomaha.edu 
### Please add reference to this notebook if you use it :)


# What is this Python Script?
# Convert Latitude/Longitude data into location name
# Use GeoNames.org API to fetch location data
# Register @ https://www.geonames.org/login to get a Free API Key


# Import required dependencies
import csv
import requests
import json
import pandas as pd


# Converts Latitude & Longitude data into meaningful location name, 
# such as country or ocean name. We will use geonames.org API to fetch 
# country name given Latitude & Longitude values
# Please register here: www.geonames.org
def coordinates_to_Country_Name():

    # Open the CSV file with Latitude and Longitude value
    with open("latitude_longitude.csv") as csv_file:

        # Read the imported csv file, seperated by comma
        read_csv = csv.reader(csv_file, delimiter = ',')

        # Skip header column names
        next(read_csv, None)

        # Create an empty list to append new data with country name
        list_country_name = []

        # Track the API request count
        # Initialize it to 0
        tracker = 0
 
        # For each row in CSV:
        # fetch the Latitude & Longitude values and pass it to the API
        for row in read_csv:

            # Seperation of concern variables
            # Latitude is on row 3 and Longitude is on row 4
            latitude = row[3]
            longitude = row[4]

            # Base API URL
            API_URL = "http://api.geonames.org/"

            # Your API Key
            # Register @ www.geonames.org/login to get a Free API Key
            API_USERNAME = "<API_KEY>"

            # Increment the API tracker by 1 each time it makes a request
            # Print the current API request number
            tracker = tracker + 1
            print("\nCurrent API request #", tracker)

            # Request the API by passing Latitude and Longitude
            # Print the API url requested
            api_call_url = API_URL + "countryCodeJSON?" + "lat=" + latitude + "&lng=" + longitude + "&username=" + API_USERNAME
            print("API URL: ", api_call_url)
           
            # Save the JSON dump into result
            API_JSON_DUMP = requests.get(api_call_url)

            # Get the JSON result into results
            # Load the JSON from the result
            API_JSON_DUMP_DATA = API_JSON_DUMP.text
            JSON_API = json.loads(API_JSON_DUMP_DATA)

            # If the API has "Country" name
            if('countryName' in JSON_API):

                print("***** COUNTRY EXISTS *****:", JSON_API["countryName"])
                
                # Append the data into a JSON structure
                list_country_name.append({
                    "latitude" : float(latitude),
                    "longitude" : float(longitude),
                    "location" : JSON_API["countryName"],
                    "location_code" : JSON_API["countryCode"],
                    "distance" : 0,
                    "location_type" : "land"
                })

            # If the API does not contain "Country" names
            # Check if it contains "Ocean" names
            else:
                
                print("***** NOT A COUNTRY ***** ")

                # Request the API by passing Latitude and Longitude
                # Print the API url requested
                api_call_url = API_URL + "oceanJSON?formatted=true&lat=" + latitude + "&lng=" + longitude + "&username=" + API_USERNAME + "&style=full"
                print("API URL: ", api_call_url)
            
                # Save the JSON dump into result
                API_JSON_DUMP = requests.get(api_call_url)

                # Get the JSON result into results
                # Load the JSON from the result
                API_JSON_DUMP_DATA = API_JSON_DUMP.text
                JSON_API = json.loads(API_JSON_DUMP_DATA)
                
                if('ocean' in JSON_API):
                    print("***** OCEAN EXISTS *****: ", JSON_API["ocean"]["name"])

                    # Append the data into a JSON structure
                    list_country_name.append({
                        "latitude" : float(latitude),
                        "longitude" : float(longitude),
                        "location" : JSON_API["ocean"]["name"],
                        "location_code" : "OCN",
                        "distance" : JSON_API["ocean"]["distance"],
                        "location_type" :JSON_API["ocean"]
                    })

                # If the API does not contain "Country" names or "Ocean" names
                # This means that Geonames.org API does not have the requested information
                # It could also mean that we have exceed the request limit(if you have free version API)
                else:

                    print("***** THERE IS NO COUNTRY OR OCEAN DATA ***** ")

                    # Put a note in the JSON that for the Latitude & Longitude there is no data
                    # Append the data into a JSON structure
                    list_country_name.append({
                        "latitude" : float(latitude),
                        "longitude" : float(longitude),
                        "location" : "Unknown",
                        "location_code" : "Unknown",
                        "distance" : 0,
                        "location_type" : "Unknown"
                    })


        # Print the final JSON list  
        print("\nFinal JSON: ", list_country_name , '\n')

        # Create a new file so that we can write the data into it
        create_json_file = open("latitude_longitude.json","w+") 

        # Save the file as JSON extention
        create_json_file.write(str(list_country_name))

        # Close the "create_json_file" file to end the processes
        create_json_file.close() 



# If you also need the converted data as a CSV
# Convert the JSON file into CSV that you created using Pandas
def csv_to_json():

    # Read the JSON file
    # Save it as a DataFrame
    df = pd.read_json (r'latitude_longitude.json')

    # Convert the Pandas DataFrame into CSV
    df.to_csv (r'latitude_longitude_country.csv', index = None)




##############################
###### CALL FUNCTION #########

### FUNCTION: Use geonames.org API to get meaningful location names
coordinates_to_Country_Name()

### FUNCTION: Convert the JSON into CSV
csv_to_json()




### Thats all.
### Hope it was useful :)