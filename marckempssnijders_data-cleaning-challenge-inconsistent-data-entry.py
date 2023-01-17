# modules we'll use
import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

# set seed for reproducibility
np.random.seed(0)
# look at the first ten thousand bytes to guess the character encoding
with open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)
# read in our dat
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
# Your turn! Take a look at all the unique values in the "Province" column. 
# Then convert the column to lowercase and remove any trailing white spaces
provinces = suicide_attacks['Province'].unique()

# sort them alphabetically and then take a closer look
provinces.sort()
provinces
# convert to lower case
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
# remove trailing white spaces
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()

provinces = suicide_attacks['Province'].unique()

# sort them alphabetically and then take a closer look
provinces.sort()
provinces
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# get the top 10 closest matches to "d.i khan"
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")
# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.
matches = fuzzywuzzy.process.extract("kuram agency", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
#We can see from the results above that'kuram agency' and 'kurram agency' score above the default ratio (90) 
#of the replace_matches_in_column method. We can therefore use this without any problem
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kuram agency")

# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
#From the excercise above: inspect Province
provinces = suicide_attacks['Province'].unique()

# sort them alphabetically and then take a closer look
provinces.sort()
provinces
# From https://en.wikipedia.org/wiki/Administrative_units_of_Pakistan

# There is a mix of abbreviations and full names.
# As a first choice, abbreviations may be used rather than full names
#Note: although province encoding to abbreviation will not make any difference for future learning models
#it is good practice to standardize data to a common format

#'balochistan' and 'baluchistan' are the same province with abbreviation 'BL'
#'capital' appears to refer to 'Islamabad Captial' with abbreviation 'ICT'

#'fata' appears to have been merged with KP province, but we'll keep it for now as it is more specific
#and thus informative
#      'On 31 May the final step in the merger of the Fata with Khyber Pakhtunkhwa (KP) was completed'
#'punjab' and 'sindh' have 'PJ' and 'SN' codes respectively
#'kpk' appears to indicate the 'Khyber Pakhtunkhwa' province with abbreviation 'KP'

#Note: 'Province' and 'City' are of course related. While chosing the abbreviation code it makes sense 
#to (cross) check provinces and cities


matches = fuzzywuzzy.process.extract("balochistan", provinces, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
replace_matches_in_column(df=suicide_attacks, column='Province', string_to_match="balochistan")
provinces = suicide_attacks['Province'].unique()

# sort them alphabetically and then take a closer look
provinces.sort()
provinces
from datetime import date
from convertdate import islamic

suicide_attacks.loc[(suicide_attacks['City']=='quetta') & (suicide_attacks['Latitude'].isnull()),'Latitude']=30.209500
suicide_attacks.loc[(suicide_attacks['City']=='quetta') & (suicide_attacks['Longitude'].isnull()),'Longitude']=67.0182

suicide_attacks.loc[(suicide_attacks['City']=='lahore') & (suicide_attacks['Latitude'].isnull()),'Latitude']=31.545100
suicide_attacks.loc[(suicide_attacks['City']=='lahore') & (suicide_attacks['Longitude'].isnull()),'Longitude']=74.435675

#Convert and parse dates
# convert to lower case
suicide_attacks['Date'] = suicide_attacks['Date'].str.lower()
# remove trailing white spaces
suicide_attacks['Date'] = suicide_attacks['Date'].str.strip()
suicide_attacks['Date_parsed'] = pd.to_datetime(suicide_attacks['Date'], infer_datetime_format=True, errors='coerce')
#We are left with one NaT entry in Date_parsed, correct this manually
#thursay-aug 27-2015
suicide_attacks.loc[(suicide_attacks['Date_parsed'].isnull()), 'Date_parsed']=date(2015, 8, 27)

#Now we can convert the dates to islamic dates
suicide_attacks['IslamicDate_parsed'] = suicide_attacks['Date_parsed'].apply(lambda x: islamic.from_gregorian(x.year, x.month, x.day))


suicide_attacks.loc[(suicide_attacks['Time'].isnull()), 'Time']='0:00 AM'

suicide_attacks.loc[(suicide_attacks['Influencing Event/Event'].isnull()), 'Influencing Event/Event']='Unknown'

#['0:00 AM', 'Unknown', 2.0, 1.0, 3.0, nan, 4.0]
suicide_attacks.loc[(suicide_attacks['No. of Suicide Blasts'].isnull()), 'No. of Suicide Blasts']='Unknown'
suicide_attacks.loc[(suicide_attacks['No. of Suicide Blasts']=='0:00 AM'), 'No. of Suicide Blasts']='Unknown'

#['Unknown', 16.0, 14.0, 2.0, nan, 25.0, 4.0, 20.0, 5.0, 37.0, 47.0,
#       6.0, 40.0, 1.0, 13.0, 8.0, 23.0, 17.0, 28.0, 3.0, 15.0, 125.0,
#       21.0, 24.0, 10.0, 30.0, 7.0, 22.0, 27.0, 11.0, 42.0, 82.0, 58.0,
#       45.0, 0.0, 70.0]
#We will want an integer to appear here, use -1 as an unspecified value
suicide_attacks.loc[(suicide_attacks['Killed Min'].isnull()), 'Killed Min']="Unknown" 
#We will want an integer to appear here, use -1 as an unspecified value
suicide_attacks.loc[(suicide_attacks['Killed Min']=='Unknown'), 'Killed Min']=-1 

#Killed Max
suicide_attacks.loc[(suicide_attacks['Killed Max'].isnull()), 'Killed Max']=-1

#'Injured Min' contains -1 as a value wich may be used for null values
suicide_attacks.loc[(suicide_attacks['Injured Min'].isnull()), 'Injured Min']=-1 

#'Injured Max' contains -1 which can this be used for null values
#a more problematic value are 40+ '100+', which should be converted an int or dropped
suicide_attacks.loc[(suicide_attacks['Injured Max'].isnull()), 'Injured Max']=-1 
suicide_attacks.loc[(suicide_attacks['Injured Max']=='40+'), 'Injured Max']=40
suicide_attacks.loc[(suicide_attacks['Injured Max']=='100+'), 'Injured Max']=100

#[-1, nan, 'Weekend', 'Iqbal Day', 'Ashura']
#-1 and nan are converted to 'Unknown'
suicide_attacks.loc[(suicide_attacks['Holiday Type'].isnull()), 'Holiday Type']='Unknown'
suicide_attacks.loc[(suicide_attacks['Holiday Type']==-1), 'Holiday Type']='Unknown'

#['Unknown', 'shiite', 'None', nan, 'Shiite', 'Sunni']
suicide_attacks.loc[(suicide_attacks['Targeted Sect if any'].isnull()), 'Targeted Sect if any']='Unknown'

#['Unknown', 'Mayo Hospital ', '1.District Headquarters \nHospital ',
#       nan, '1.Hayatabad Medical Complex',
#       '1.Lady Reading Hospital 2.Hayatabad medical Complex',
#       'sakhi sarwar civil hospital-dera district headquarters hospital']
suicide_attacks.loc[(suicide_attacks['Hospital Names'].isnull()), 'Hospital Names']='Unknown'

#['Unknown', nan, '5 to 6 Kg']
suicide_attacks.loc[(suicide_attacks['Explosive Weight (max)'].isnull()), 'Explosive Weight (max)']='Unknown'

#at this point we should no longer have null values appearing which we may validate below
# get the number of missing data points per column
missing_values_count = suicide_attacks.isnull().sum()

#[-1, 'Hotel', 'Religious', 'Park/Ground', 'Mobile', 'Military',
#       'Airport', 'Government', 'Police', 'Transport', 'Residence',
#       'Foreign', 'Market', 'Hospital', 'Educational', 'Bank',
#       'Foreigner', 'Office Building', 'foreign', nan,
#       'Residential Building', 'Commercial/residence', ' ', 'Highway'
suicide_attacks.loc[(suicide_attacks['Location Category'].isnull()), 'Location Category']='Unknown'
suicide_attacks.loc[(suicide_attacks['Location Category']==-1), 'Location Category']='Unknown'
suicide_attacks.loc[(suicide_attacks['Location Category']==' '), 'Location Category']='Unknown'

#[-1, 'Medium', 'Low', 'High', nan]
suicide_attacks.loc[(suicide_attacks['Location Sensitivity'].isnull()), 'Location Sensitivity']='Unknown'
suicide_attacks.loc[(suicide_attacks['Location Sensitivity']==-1), 'Location Sensitivity']='Unknown'

#[-1, 'Closed', 'Open', 'open', nan, 'closed', 'Open/Closed']
suicide_attacks.loc[(suicide_attacks['Open/Closed Space'].isnull()), 'Open/Closed Space']='Unknown'
suicide_attacks.loc[(suicide_attacks['Open/Closed Space']==-1), 'Open/Closed Space']='Unknown'

#'Location' contains free text, but also -1 and nan convert nan and -1 to 'Unknown'
suicide_attacks.loc[(suicide_attacks['Location'].isnull()), 'Location']='Unknown'
suicide_attacks.loc[(suicide_attacks['Location']==-1), 'Location']='Unknown'

#[-1, 'Foreigner', 'Religious', 'Government Official', 'Civilian',
#       'Military', 'Police', 'Government official', 'civilian', 'police',
#       'Children/Women', 'Anti-Militants', 'foreigner', 'Media',
#       'religious', nan, 'Police & Rangers', 'Civilian & Police', 'Army',
#       'Frontier Corps ', 'advocates (lawyers)', 'Civilian Judges',
#       'Shia sect', 'Judges & lawyers'
suicide_attacks.loc[(suicide_attacks['Target Type'].isnull()), 'Target Type']='Unknown'
suicide_attacks.loc[(suicide_attacks['Target Type']==-1), 'Target Type']='Unknown'

#[-1, 'Working Day', 'Holiday', nan, 'Weekend']
suicide_attacks.loc[(suicide_attacks['Blast Day Type'].isnull()), 'Blast Day Type']='Unknown'
suicide_attacks.loc[(suicide_attacks['Blast Day Type']==-1), 'Blast Day Type']='Unknown'

# 	Temperature(C) 	Temperature(F)
#165 	NaN 	NaN
#449 	NaN 	NaN
#450 	NaN 	NaN
#473 	NaN 	NaN
#479 	NaN 	NaN
#Celsius to Fahrenheit conversion(°C × 9/5) + 32 = °F 
#1. Convert all null temperature to a ridiculously low temperature (temperatures in data range from -2.3700e+00 upward)
#use –273,15C for this
#2.convert null Temperature(F) from Temperature(C)
suicide_attacks.loc[(suicide_attacks['Temperature(C)'].isnull()), 'Temperature(C)']= float(-275.15)
suicide_attacks.loc[(suicide_attacks['Temperature(F)'].isnull()), 'Temperature(F)']= (suicide_attacks['Temperature(C)'] *9/5)+32


#Finally we may drop selected columns that are of no interest to us(anymore)
#In this case I choose to drop 'Date' and 'Islamic Date' since these were converted above
suicide_attacks = suicide_attacks.drop(['Date', 'Islamic Date'], axis=1)


# get the number of missing data points per column
missing_values_count = suicide_attacks.isnull().sum()

# look at the # of missing points in the columns
missing_values_count[:]