#Import the necessary library

import numpy as np

import pandas as pd

import warnings



warnings.filterwarnings('ignore')
#Manually add the Summer Olympics, London 2012 dataset as arrays

df = pd.read_excel("../input/Olympic 2012 Medal Tally.xlsx",sheet_name='Sheet1',skiprows = [1,2])

arrCountries = np.array(df['Unnamed: 1'])

arrCountriesCode = np.array(df['Unnamed: 2'])

arrCountriesWonYear = np.array(df['Unnamed: 3'])

arrCountriesWonTotalGold = np.array(df['Unnamed: 4'])

arrCountriesWonTotalSilver = np.array(df['Unnamed: 5'])

arrCountriesWonTotalBronze = np.array(df['Unnamed: 6'])



#Use the argmax() method to find the highest number of gold medals

highestCountryGoldIndex = arrCountriesWonTotalGold.argmax()

arrCountries[highestCountryGoldIndex]
#Print the name of the country

for countryName in arrCountries:

    print(countryName)
#Use Boolean indexing technique to find the required output

arrCountriesMoreThan20Gold = arrCountries[arrCountriesWonTotalGold > 20]

for countryName in arrCountriesMoreThan20Gold:

    print(countryName)
#Use a for loop to create the required output

print ("{:<20} {:<10} {:<10}".format('Country','Golds','Total Medals')) # formatting tables

for index,country in enumerate(arrCountries):

    totalMedals = (arrCountriesWonTotalGold[index]+arrCountriesWonTotalSilver[index]+arrCountriesWonTotalBronze[index]);

    print ("{:<20} {:<10} {:<10}".format(country,arrCountriesWonTotalGold[index],totalMedals))