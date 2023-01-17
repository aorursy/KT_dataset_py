#import Libraries

import numpy as np
#getting Data From File

fileCountryWithGDP = open("../input/Countries with GDP.txt","r")

listCountryWithGDP = fileCountryWithGDP.readlines()

print(listCountryWithGDP)
# there are four lines in files and second and fourth lines are of our concern

strCountries = listCountryWithGDP[1]

strCountriesGDP =listCountryWithGDP[3]
print(strCountries)
print(strCountriesGDP)
# Convert string into numpy array 

listCountries = strCountries.split(",") # Converting String to list

arrCountries = np.array(listCountries) # Converting List to numpy 

print(arrCountries)
#same process for country

listCountriesGDP = strCountriesGDP.split(",")

arrCountriesGDP = np.array(listCountriesGDP)

print(arrCountriesGDP)
# Now we have two numpy array one for countries and another for GDP

# lets check whether both have same elements or not to check gdp corresponding to the country

arrCountriesGDP.size == arrCountries.size
# Both datasets have equal number of elements now we have accurate data so we can proceed further 

# now lets clean our data values inside array lets filter countries array first

arrCountries
# Countries name have extra single quotes and a \n is also there with the last element

arrCountries = np.chararray.replace(arrCountries,"\n","")

arrCountries = np.chararray.replace(arrCountries,"'","")

print(arrCountries)
arrCountriesGDP
#now Comes the main part 



# find and print the name of country with highest GDP

#typecasting of GDP to float as we need to 

arrCountriesGDP = arrCountriesGDP.astype('float')

arrCountries[arrCountriesGDP.argmax()]

#find and print the name of country with lowest GDP

arrCountries[arrCountriesGDP.argmin()]
#print out text and input values iteratively

for country in arrCountries:

    print(country)
#print out the entire list of countries with their GDPs

for index,country in enumerate(arrCountries):

    print("Country : ",country.strip(),"| GDP: ",arrCountriesGDP[index])
# Print the highest GDP value, lowest GDP value, mean GDP value, standardized GDP value and sum of all GDP values

max(arrCountriesGDP)
min(arrCountriesGDP)
arrCountriesGDP.mean()
arrCountriesGDP.std()
sum(arrCountriesGDP)