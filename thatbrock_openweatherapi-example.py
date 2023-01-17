from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("openweathermap")
import urllib.error, urllib.parse, urllib.request, json

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
    key = secret_value_0
    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
    print(url)
    return safeGet(url)


data = json.load(getForecast())
print(data)
print(f"The current weather in Seattle is: {data['weather'][0]['description']}")
#dataParis=json.load(getForecast('Paris'))
tempK = data['main']['temp']  # the temperature is provided in °Kelvin 
tempF = (tempK-273.15)*(9/5) + 32  # conversion to °F
description = data['weather'][0]['description']
print(f"The current temperature in Paris is: {tempF} °F" )
print(data['weather'][0]['main'])