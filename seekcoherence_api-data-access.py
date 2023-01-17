

# Desc: This program uses an API from openWeatherMap to get current weather conditions for a users location entry.





import json

import requests



# This function passes the user entered zipcode into the OpenWeatherMap API and converts the response to a Python dict

def getapi(User_Location):



    response = requests.get("https://api.openweathermap.org/data/2.5/weather?q={}&units=imperial&APPID=7f9b08f4bdaea2a929e3d53c720e6829".format(User_Location))

    convertTo = json.loads(response.text)

    validateconnection(convertTo)



#This function checks if a valid 200 response is received from the OpenWeatherMap API and displays a message to user based on success response. Also continues with another function if successful

def validateconnection(convertTo):

    try:

        if convertTo['cod'] == 200:

            print("~~Weather Request Successful~~ \n")

            prettyPrint(convertTo)

        else:

            print("Request Failed \n")

            print("Please double check entry to ensure it's a valid USA zipcode")



    except:

        pass





# This function will format the API response into a readable format.

def prettyPrint(convertTo):

    print("The current weather in " + convertTo['name'] + " is " + convertTo['weather'][0]['main'])

    print("Current Temp: " +str(round(convertTo['main']['temp']))+" F`")

    print("Today's Low Temp: " +str(round(convertTo['main']['temp_min']))+" F`")

    print("Today's High Temp: " +str(round(convertTo['main']['temp_max']))+" F`")

    print("Current Humidity: " +str(round(convertTo['main']['humidity']))+"%")

    print("Current Wind Speed: " + str(round(convertTo['wind']['speed'])) + "/MPH")

    print("\n")







# Main function that starts sequence of passing user input to API.

def main(User_Location):

    getapi(User_Location)









#Greets same user only once during program run.

print("Hello welcome to Rain'N Shine's weather app \n")





# Asks for user input. If input is 5 characters in length main() is executed. Else user can quit the program or have error messages displayed if other entries are inputted.

goaround = 10

while goaround != 1 :



    User_Location = input("Please enter a zipcode to see current weather conditions or enter 'quit' to quit \n")



    if len(User_Location) == 5:

        print("~~User Entry Accepted: Forwarding request to weather webservice~~ \n")

        main(User_Location)



    elif User_Location == "quit":

        print("Thank you the program will now close. Please press enter")

        input()

        exit()



    else:

        print("You've entered an invalid entry please enter a valid zipcode and try again \n")

        continue
















