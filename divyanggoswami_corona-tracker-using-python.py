#import request library
import requests

url = 'https://api.covid19api.com/summary' #api URL

r = requests.get(url)
statusCode = r.status_code

#API testing Working or not
if (statusCode == 200):
    print("API is working")
else:
    print("API URL is incorrect")
        


r.headers
res = r.text
type(res) # response comes in the form of strings

#convert str into Dict
# Here we will convert the Strings response into Dictionary

#Import JSON code
import json
res_dict = json.loads(res) #convert strings into JSON dictionary
type(res_dict)
    
fullData =  res_dict["Countries"]
print(fullData)

#In the response you can see all the data are coming from the Covid-19 API where we will fetch the details of the Country
# Field and then do the further calculation of Total Number of Covid Cases, Recovered Cases etc.
     
#Here we will get the country Input from the User

countryName = input("Insert Country Name: ").title()

# Now we will fetched all the country records from the API by using a For Loop
Count = 0
for i in fullData:
    dataCountry =  i['Country']
    if(dataCountry == countryName ): # Here we will matched the user inserted strings
        code = Count 
        break
    
    Count= Count +1 
    
# Fetched Country Name from User inserted Country 
country = res_dict["Countries"][code]['Country']

# Fetched Total Confirmed Cases from User inserted Country 
Confirmed = res_dict["Countries"][code]['TotalConfirmed']

# Fetched Total Recovered Cases from User inserted Country 
Recovered = res_dict["Countries"][code]['TotalRecovered']

# Fetched Total Fatal Cases from User inserted Country 
Fatal =  res_dict["Countries"][code]['TotalDeaths']

# Calculate Active Cases
ActiveCases = (Confirmed - Recovered)

#Calculate Recovery Rate
recoveryRate = (Recovered/Confirmed)*100;

if (Confirmed == 0):
    print ('Not Available')


#Print All Calculated Results
print('Country:',country )
print("Corona  Confirmed Cases: ", Confirmed)
print("Recovered Cases: ", Recovered)
print("Active Cases", ActiveCases)
print("Fatal Cases: ", Fatal)
print("Recovery Rate", recoveryRate, "%")
