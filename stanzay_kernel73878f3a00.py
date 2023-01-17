import csv
f = open("/kaggle/input/gun-deaths-in-the-us/guns.csv", "r")
data = list(csv.reader(f))
headers = data[:1]
print(headers)
data = data[1:]
print(data[:6])
# year column  -- the year in which the fatality occurred.
years = [item[1] for item in data]
set(years)
# month column -- the month in which the fatality occurred.
month = [item[2] for item in data]
set(month)
# intent column -- the intent of the perpetrator of the crime.
intents = [item[3] for item in data]
set(intents)
# police column -- whether a police officer was involved with the shooting.
police = [item[4] for item in data]
set(police)
# sex column -- the gender of the victim.
sex = [item[5] for item in data]
set(sex)
# age column -- the age of the victim.
age = [item[6] for item in data]
print(set(age))
# race column -- the race of the victim.
races = [item[7] for item in data]
set(races)
# hispanic column  -- a code indicating the Hispanic origin of the victim.
hispanic = [item[8] for item in data]
print(set(hispanic))
# place column -- where the shooting occurred.
place = [item[9] for item in data]
set(place)
# education column -- educational status of the victim. 
education = [item[10] for item in data]
set(education)
year_counts = {}
for item in years:
    if item in year_counts:
        year_counts[item] += 1
    else:
        year_counts[item] = 1
year_counts
import datetime
dates_objects = [datetime.datetime(year=int(item[1]), month=int(item[2]), day=1) for item in data]
dates_objects[:6]
date_counts = {}
for item in dates_objects:
    if item in date_counts:
        date_counts[item] += 1
    else:
        date_counts[item] = 1
date_counts        
sex_counts = {}
for item in data:
    if item[5] in sex_counts:
        sex_counts[item[5]] += 1
    else:
        sex_counts[item[5]] = 1
        
sex_counts
race_counts = {}
for item in data:
    if item[7] in race_counts:
        race_counts[item[7]] += 1
    else:
        race_counts[item[7]] = 1
        
race_counts
file = open("/kaggle/input/census/census.csv", "r")
census = list(csv.reader(file))
print(census)
census_header = census[0]
census_header
census_data = census[1:]
census_data
# Manual mapping of racial values between datasets
mapping = {
    "Asian/Pacific Islander": 15834141,
    "Black": 40250635,
    "Native American/Native Alaskan": 3739506,
    "Hispanic": 44618105,
    "White": 197318956
}
# Gun deaths per race by 100000 of people in a racial category
race_per_hundredk = {}
for item in race_counts:
    race_per_hundredk[item] = (race_counts[item] / mapping[item]) * 100000

race_per_hundredk
homicide_race_counts = {}
for key,value in enumerate(races):
    if value not in homicide_race_counts:
        homicide_race_counts[value] = 0
    if intents[key] == "Homicide":
        homicide_race_counts[value] += 1
print(homicide_race_counts)
homicide_race_per_hundredk = {}
for item,value in homicide_race_counts.items():
    homicide_race_per_hundredk[item] = (value / mapping[item])*100000
    
homicide_race_per_hundredk   
