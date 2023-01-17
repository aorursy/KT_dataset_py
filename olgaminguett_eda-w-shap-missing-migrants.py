import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")



%matplotlib inline
data = pd.read_csv('/kaggle/input/missingmigrants/MissingMigrantsProject.csv', encoding='latin-1') # load data from csv

data.head() #first 5 rows
data.sample(5)
data.columns
#Renaming data columns for better understanding of the fields

data.rename(columns={'id': 'ID',

                      'cause_of_death': 'CauseOfDeath',

                      'region_origin': 'RegionOfOrigin',

                      'affected_nationality': 'Nationality',

                      'missing': 'MissingNumberOfPeople',

                      'dead': 'DeadNumberOfPeople',

                      'incident_region': 'IncidentRegion',

                      'date': 'Date',

                      'source': 'Source',

                      'reliability': 'Reliability',

                      'lat': 'Latitud',

                      'lon': 'Longitud'}, inplace=True)
data.head()
#Descriptive or summary statistics of numeric columns

data.describe()
#Concise summary of the dataframe

data.info()
#Returns the dimensions of the array

#Rows & Columns

data.shape
#Representation of the distribution of data

data.hist(figsize=[10,10]);
#Drop rows that have min of 9 NAN values

data.dropna(thresh=9,inplace=True)
#Returns the dimensions of the array

#Rows & Columns

data.shape
#Percentage of NAN Values

data.isna().mean().round(4) * 100
#Number of NAN Values

data.isna().sum()
#Checking the uniqueness in the columns

len(data['ID'].unique())                  
#Checking the uniqueness in the columns

len(data['CauseOfDeath'].unique())
data['CauseOfDeath'].isna().sum()
#Unknown instead of NAN as cause of Death

data['CauseOfDeath'].fillna('Unknown', inplace = True) 
data['CauseOfDeath'].head()
#Outline different types of Cause of Death

set(data['CauseOfDeath'])
data['CauseOfDeath'] = data['CauseOfDeath'].replace({'AH1N1 influenza virus, while stuck at border': 'Virus',

                                                     'Asphyxiation': 'Asphyxiation',

                                                     'Asphyxiation (Silica sand inhalation)': 'Asphyxiation',

                                                     'Asphyxiation and crushing': 'Asphyxiation + Others',

                                                     'Assaulted by smugglers': 'Criminal Violence',

                                                     'Attacked by hippopotamus': 'Attacked by Animal',

                                                     'Beat-up and killed': 'Beat-up',

                                                     'Beat-up and thrown into river': 'Beat-up',

                                                     'Beaten to death on train': 'Beat-up',

                                                     'Beating/shot by traffickers': 'Beat-up',

                                                     'Blunt force head trauma': 'Blunt Force',

                                                     'Blunt force injuries': 'Blunt Force',

                                                     'Boat fire': 'Burned to death',

                                                     'Bronchial aspiration': 'Respiratory Complication',

                                                     'Bronchopneumonia': 'Respiratory Complication',

                                                     'Burned to death hiding in truck': 'Burned to death',

                                                     'Burns and Suffocation': 'Burned to death',

                                                     'Burns from cooking gas explosion in connection house in Libya': 'Burned to death',

                                                     'Car accident': 'Car Accident',

                                                     'Cervix cancer': 'Cancer',

                                                     'Clubbed/beaten to death': 'Beat-up',

                                                     'Criminal Violence': 'Criminal Violence',

                                                     'Crushed': 'Crushed',

                                                     'Crushed / drowning': 'Crushed',

                                                     'Crushed by bus on ferry': 'Crushed',

                                                     'Crushed by pallets': 'Crushed',

                                                     'Crushed to death': 'Crushed',

                                                     'Cut in half by train': 'Accident',

                                                     'Dehydration': 'Dehydration',

                                                     'Dehydration Harsh_weather_lack_of_adequate_shelter': 'Dehydration + Others',

                                                     'Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse Sexual_abuse': 'Dehydration + Others',

                                                     'Dehydration Suffocation Vehicle_Accident': 'Dehydration + Others',

                                                     'Dehydration Vehicle_Accident Excessive_physical_abuse': 'Dehydration + Others',

                                                     'Dehydration and exposure to the elements': 'Dehydration + Others',

                                                     'Dehydration, Asphyxiation': 'Dehydration + Others',

                                                     'Dehydration, Presumed drowning': 'Dehydration + Others',

                                                     'Dehydration, Starvation': 'Dehydration + Others',

                                                     'Died of unknown cause in hospital shortly after rescue': 'Unknown Situation',

                                                     'Digestive bleeding': 'Digestive Bleeding',

                                                     'Drowning': 'Drowning',

                                                     'Drowning after being thrown overboard by other passengers': 'Drowning',

                                                     'Drowning due to forced disembarcation': 'Drowning',

                                                     'Drowning or suffocation in hull': 'Drowning + Others',

                                                     'Drowning, Asphyxiation': 'Drowning + Others',

                                                     'Drowning, Other': 'Drowning + Others',

                                                     'Drowning, Trampling': 'Drowning + Others',

                                                     'Drowning. Boat collided with ferry': 'Drowning',

                                                     'Electrocuted on train': 'Electrocution',

                                                     'Electrocution': 'Electrocution',

                                                     'Electrocution on railway': 'Electrocution',

                                                     'Excessive_physical_abuse': 'Excessive_physical_abuse',

                                                     'Excessive_physical_abuse Sexual_abuse': 'Excessive_physical_abuse + Others',

                                                     'Excessive_physical_abuse Shot_or_Stabbed': 'Excessive_physical_abuse + Others',

                                                     'Exposure': 'Exposure',

                                                     'Exposure, Hyperthermia': 'Exposure + Others',

                                                     'Exposure, Hypothermia': 'Exposure + Others',

                                                     'Exposure, hypothermia': 'Exposure + Others',

                                                     'Exposure. Died upon entry to refugee camp.': 'Exposure',

                                                     'Fall from cliff': 'Fall',

                                                     'Fell from boat': 'Fall',

                                                     'Fell from train': 'Fall',

                                                     'Fell from truck': 'Fall',

                                                     'Fell from wall': 'Fall',

                                                     'Found hanged': 'Suicide',

                                                     'Fuel Inhalation': 'Respiratory Complication',

                                                     'Fuel burns': 'Burned to death',

                                                     'Glycemic crisis (Lack of Insuline Treatment)': 'Glycaemic crisis',

                                                     'Gylcemic crisis (Diabetic, medicine thrown overboard)': 'Glycaemic crisis',

                                                     'Harsh conditions': 'Harsh Conditions',

                                                     'Harsh_weather_lack_of_adequate_shelter': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse Sexual_abuse': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Other': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Suffocation': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse Sexual_abuse': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Suffocation Vehicle_Accident': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter Vehicle_Accident': 'Harsh Conditions + Others',

                                                     'Harsh_weather_lack_of_adequate_shelter, Suffocation, Excessive_physical_abuse, Sexual_abuse': 'Harsh Conditions + Others',

                                                     'Head injury': 'Blunt Force',

                                                     'Head injury from fall': 'Blunt Force',

                                                     'Head trauma (hit by boat propeller)': 'Blunt Force',

                                                     'Heart Attack': 'Heart Attack',

                                                     'Heart attack': 'Heart Attack',

                                                     'Hi by truck': 'Hit By Automotive Vehicle',

                                                     'Hit by Vehicle': 'Hit By Automotive Vehicle',

                                                     'Hit by car': 'Hit By Automotive Vehicle',

                                                     'Hit by train': 'Hit By Automotive Vehicle',

                                                     'Hit by truck': 'Hit By Automotive Vehicle',

                                                     'Hit by vehicle': 'Hit By Automotive Vehicle',

                                                     'Homicide, likely by asphyxiation': 'Asphyxiation',

                                                     'Hunger, fatigue': 'Starvation',

                                                     'Hyperthermia': 'Hyperthermia',

                                                     'Hyperthermia, Abandoned by smugglers in the desert': 'Hyperthermia',

                                                     'Hyperthermia, Dehydration': 'Hyperthermia + Others',

                                                     'Hyperthermia, starvation': 'Hyperthermia + Others',

                                                     'Hypothermia': 'Hypothermia',

                                                     'Hypothermia, Exhaustion': 'Hypothermia + Others',

                                                     'Hypothermia, Malnutrition': 'Hypothermia + Others',

                                                     'Inhalation of toxic fumes from boat engine': 'Respiratory Complication',

                                                     'Injured from a fight': 'Blunt Force',

                                                     'Injuries caused by boat motor': 'Blunt Force',

                                                     'Killed': 'Murdered',

                                                     'Landmine': 'Landmine',

                                                     'Likely drowning': 'Drowning',

                                                     'Likely suffocation (found dead in a truck)': 'Suffocation',

                                                     'Lung infection': 'Respiratory Complication',

                                                     'Meningitis': 'Meningitis',

                                                     'Mixed': 'Unknown Situation',

                                                     'Mixed - mostly drownings or shootings': 'Drowning + Others',

                                                     'Mixed. Migrants were stranded on boats': 'Unknown Situation',

                                                     'Mostly starvation, dehydration, and beatings by crew members': 'Starvation + Others',

                                                     'Multiple blunt force injuries': 'Blunt Force',

                                                     'Murdered': 'Murdered',

                                                     'Murdered (bandits)': 'Murdered',

                                                     'Murdered (head wound)': 'Murdered',

                                                     'Murdered (militia)': 'Murdered',

                                                     'Murdered by gang members': 'Murdered',

                                                     'On board violence': 'Criminal Violence',

                                                     'Other': 'Unknown Situation',

                                                     'Other Shot_or_Stabbed': 'Shot',

                                                     'Pending': 'Unknown Situation',

                                                     'Pima County (see spreadsheet for exact location)': 'Unknown Situation',

                                                     'Plane Stowaway': 'Plane Stowaway',

                                                     'Plane stowaway': 'Plane Stowaway',

                                                     'Poison': 'Poison',

                                                     'Presumed Dehydration': 'Dehydration',

                                                     'Presumed Drowning': 'Drowning',

                                                     'Presumed asphyxiation': 'Asphyxiation',

                                                     'Presumed dehydration': 'Dehydration',

                                                     'Presumed drowning': 'Drowning',

                                                     'Presumed exposure': 'Exposure',

                                                     'Presumed hyperthermia': 'Hyperthermia',

                                                     'Presumed hypothermia': 'Hypothermia',

                                                     'Presumed shot': 'Shot',

                                                     'Presumed violence': 'Criminal Violence',

                                                     'Probable drowning': 'Drowning',

                                                     'Pulmonary complications': 'Respiratory Complication',

                                                     'Pulmonary edema': 'Respiratory Complication',

                                                     'Pulmonary edema, Kidney failure': 'Respiratory Complication',

                                                     'Raped and Murdured': 'Raped & Murdered',

                                                     'Raped and murdered': 'Raped & Murdered',

                                                     'Renal insufficiency': 'Renal Insufficiency',

                                                     'Renal insufficiency and pulmonary edema': 'Renal Insufficiency',

                                                     'Respiratory illness': 'Respiratory Complication',

                                                     'Respiratory problem': 'Respiratory Complication',

                                                     'Road accident': 'Accident',

                                                     'Severe exhaustion and dehydration': 'Exhaustion',

                                                     'Sexual_abuse': 'Sexual Abuse',

                                                     'Shot': 'Shot',

                                                     'Shot by Apache helicopter': 'Shot',

                                                     'Shot_or_Stabbed': 'Shot',

                                                     'Sickness': 'Sickness',

                                                     'Sickness, Harsh conditions': 'Sickness + Others', 

                                                     'Sickness_and_lack_of_access_to_medicines': 'Sickness & No Medicines',

                                                     'Sickness_and_lack_of_access_to_medicines, Dehydration': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Harsh_weather_lack_of_adequate_shelter': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Shot_or_Stabbed': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Suffocation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration Vehicle_Accident': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Excessive_physical_abuse Sexual_abuse Shot_or_Stabbed': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Excessive_physical_abuse Shot_or_Stabbed': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter Other': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter Suffocation': 'Sickness & No Medicines + Others', 'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter Suffocation Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Harsh_weather_lack_of_adequate_shelter, Vehicle_Accident':'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Shot_or_Stabbed': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Starvation, Dehydration, Excessive_physical_abuse, Sexual_abuse':'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Harsh_weather_lack_of_adequate_shelter': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Vehicle_Accident Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Dehydration Suffocation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Excessive_physical_abuse Shot_or_Stabbed': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Suffocation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Suffocation Vehicle_Accident Excessive_physical_abuse Sexual_abuse':'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Vehicle_Accident Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Starvation, Harsh_weather_lack_of_adequate_shelter, Vehicle_Accident, Excessive_physical_abuse':'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Suffocation Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Suffocation Excessive_physical_abuse Shot_or_Stabbed': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Suffocation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Suffocation Vehicle_Accident': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Vehicle_Accident': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Dehydration': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Dehydration, Harsh_weather_lack_of_adequate_shelter': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Harsh_weather_lack_of_adequate_shelter Vehicle_Accident': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Starvation': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Starvation, Dehydration Excessive_physical_abuse, Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Starvation, Excessive_physical_abuse, Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines, Starvation, Harsh_weather_lack_of_adequate_shelter, Excessive_physical_abuse, Sexual_abuse': 'Sickness & No Medicines + Others',

                                                     'Sickness_and_lack_of_access_to_medicines Starvation Harsh_weather_lack_of_adequate_shelter Vehicle_Accident Excessive_physical_abuse': 'Sickness & No Medicines + Others',

                                                     'Stabbed': 'Stabbed',

                                                     'Starvation': 'Starvation',

                                                     'Starvation, Dehydration': 'Starvation + Others',

                                                     'Starvation Dehydration Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Dehydration Excessive_physical_abuse Sexual_abuse Shot_or_Stabbed': 'Starvation + Others',

                                                     'Starvation Dehydration Excessive_physical_abuse Shot_or_Stabbed': 'Starvation + Others',

                                                     'Starvation Dehydration Harsh_weather_lack_of_adequate_shelter': 'Starvation + Others',

                                                     'Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation': 'Starvation + Others',

                                                     'Starvation Dehydration Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Dehydration Other': 'Starvation + Others',

                                                     'Starvation Dehydration Suffocation Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Dehydration Suffocation Excessive_physical_abuse Sexual_abuse Shot_or_Stabbed': 'Starvation + Others',

                                                     'Starvation Dehydration Vehicle_Accident': 'Starvation + Others',

                                                     'Starvation Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Excessive_physical_abuse Sexual_abuse': 'Starvation + Others',

                                                     'Starvation Harsh_weather_lack_of_adequate_shelter': 'Starvation + Others',

                                                     'Starvation Harsh_weather_lack_of_adequate_shelter Suffocation': 'Starvation + Others', 

                                                     'Starvation Harsh_weather_lack_of_adequate_shelter Suffocation Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Suffocation Excessive_physical_abuse': 'Starvation + Others',

                                                     'Starvation Suffocation Excessive_physical_abuse Sexual_abuse': 'Starvation + Others',

                                                     'Starvation Suffocation Shot_or_Stabbed': 'Starvation + Others',

                                                     'Starvation Dehydration': 'Starvation + Others',

                                                     'Starvation, Dehydration, Exhaustion': 'Starvation + Others',

                                                     'Starvation, Dehydration, Suffocation': 'Starvation + Others',

                                                     'Starvation, Suffocation': 'Starvation + Others',

                                                     'Starvation, sickness': 'Starvation + Others',

                                                     'Suffocation': 'Suffocation',

                                                     'Suffocation Excessive_physical_abuse': 'Suffocation + Others',

                                                     'Suffocation Vehicle_Accident': 'Suffocation + Others',

                                                     'Suffocation, Other': 'Suffocation + Others',

                                                     'Suffocation, Trampled': 'Suffocation + Others',

                                                     'Suicide': 'Suicide',

                                                     'Tekeze River, near Himora, Ethiopia': 'Unknown Situation',

                                                     'Tortured to death': 'Criminal Violence',

                                                     'Toxic fumes/asphyxiation': 'Asphyxiation',

                                                     'Train accident': 'Accident',

                                                     'Truck crash (was clinging to bottom of truck)': 'Blunt Force',

                                                     'Unclear, possibly related to earlier beating by truck driver': 'Blunt Force',

                                                     'Undernourished, dehydration': 'Starvation + Others',

                                                     'Unknow (skeletal remains)': 'Unknown Situation',

                                                     'Unknown': 'Unknown Situation',

                                                     'Unknown (Mummified remains)': 'Unknown Situation',

                                                     'Unknown (Skeletal remains)': 'Unknown Situation',

                                                     'Unknown (bodies found decomposed)': 'Unknown Situation',

                                                     'Unknown (bodies recovered from boat)': 'Unknown Situation',

                                                     'Unknown (bodies recovered on boat)': 'Unknown Situation',

                                                     'Unknown (body found in advanced state of decomposition)': 'Unknown Situation',

                                                     'Unknown (body found in desert)': 'Unknown Situation',

                                                     'Unknown (body recovered from boat)': 'Unknown Situation',

                                                     'Unknown (decomposed remains)': 'Unknown Situation',

                                                     'Unknown (found dead near train tracks)': 'Unknown Situation',

                                                     'Unknown (found dead on dinghy)': 'Unknown Situation',

                                                     'Unknown (found dead on top of train)': 'Unknown Situation',

                                                     'Unknown (found on motorway)': 'Unknown Situation',

                                                     'Unknown (mummified remains)': 'Unknown Situation',

                                                     'Unknown (postmortem decomposition)': 'Unknown Situation',

                                                     'Unknown (remains)': 'Unknown Situation',

                                                     'Unknown (skeletal and mummified remains)': 'Unknown Situation',

                                                     'Unknown (skeletal remains and mummified)': 'Unknown Situation',

                                                     'Unknown (skeletal remains)': 'Unknown Situation',

                                                     'Unknown, Skeletal Remains': 'Unknown Situation',

                                                     'Unknown, plane stowaway': 'Plane Stowaway',

                                                     'Unknown, torture involved': 'Torture',

                                                     "Unspecified deaths on 'La Bestia' train": 'Unknown Situation',

                                                     'Unspecified location between North Africa and Italy. Body brought to Calabria.': 'Unknown Situation',

                                                     'Van accident': 'Vehicle Accident',

                                                     'Various': 'Unknown Situation',

                                                     'Vehicle Accident': 'Vehicle Accident',

                                                     'Vehicle accident': 'Vehicle Accident',

                                                     'Vehicle incident': 'Vehicle Accident',

                                                     'Vehicle_Accident': 'Vehicle Accident',

                                                     'Vehicle_Accident Other': 'Vehicle Accident + Others',

                                                     'Vehicle_Accident Shot_or_Stabbed': 'Vehicle Accident + Others',

                                                     'Violence': 'Criminal Violence',

                                                     'Violence during riot': 'Criminal Violence',

                                                     'Violent robbery': 'Criminal Violence',

                                                     'drowning': 'Drowning',

                                                     'gang violence (body dismembered)': 'Criminal Violence',

                                                     'heart attack': 'Heart Attack',

                                                     'mixed': 'Unknown Situation',

                                                     'shot': 'Shot',

                                                     'unknown (corpses recovered from boats)': 'Unknown Situation'})
set(data['CauseOfDeath'])
len(data['CauseOfDeath'].unique())
len(data['RegionOfOrigin'].unique())   
#Outline different regions of Origin

set(data['RegionOfOrigin'])             
#Unknown instead of NAN as RegionOfOrigin

data['RegionOfOrigin'].isna().sum()
#Unknown instead of NAN as Region of Origin

data['RegionOfOrigin'].fillna('Unknown', inplace = True) 
data['RegionOfOrigin'] = data['RegionOfOrigin'].replace({'Horn of Africa (P)':'Horn of Africa',

                                                         'MENA': 'Middle East / North Africa ',

                                                         'Central America & Mexico': 'Central America'})
#Grab DataFrame rows where column has certain values

value_list = ['Unknown', 'Mixed']

data[data['RegionOfOrigin'].isin(value_list)]
data.loc[data.IncidentRegion == 'Mediterranean', ['RegionOfOrigin']] = 'Africa'
set(data['RegionOfOrigin'])  
value_list = ['Unknown', 'Mixed']

#Grab DataFrame rows where column has certain values

data[data['RegionOfOrigin'].isin(value_list)]
len(data['RegionOfOrigin'].unique())   
len(data['Nationality'].unique())   
set(data['Nationality']) 
#NAN counts

data['Nationality'].isna().sum()
#Unknown instead of NAN as Nationality

data['Nationality'].fillna('Unknown', inplace = True) 
data['Nationality'] = data['Nationality'].replace({'1 Honduran, 3 Mexican': 'Honduras, Mexico',

                                                    '1 Nigerian, others unknown. Survivors all from Sub-Saharan Africa': 'Nigeria',

                                                    '1 Venezuelan, 1 unknown': 'Venezuela',

                                                    '13 Cuba, 1 Dominican Republic, 1 Colombia': 'Cuba, Dominican Republic, Colombia',

                                                    '15 dead from Palestine. Missing are from Palestine, Syria, and Egypt': 'Palestine',

                                                    '2 from Niger': 'Niger',

                                                    '2 Senegal, 2 Guinea, 1 Ghana': 'Senegal, Guinea, Ghana',

                                                    '20 Unknown, 1 Bangladesh, 1 Senegal': 'Bangladesh, Senegal',

                                                    'Afghan': 'Afghanistan',

                                                    'Afghanistan, Iran': 'Afghanistan, Iran',

                                                    'Afghanistan, Iraq, Syria': 'Afghanistan, Iraq, Syria',

                                                    'Afghanistan, Pakistan': 'Afghanistan, Pakistan',

                                                    'Afghanistan, Syria': 'Afghanistan, Syria',

                                                    'Afghanistan, Syrian Arab Republic': 'Afghanistan, Syria',

                                                    'Afghanistan': 'Afghanistan',

                                                    'Africa (5), Morocco': 'Africa, Morocco',

                                                    'Africa': 'Africa',

                                                    'African': 'Africa',

                                                    'Albania': 'Albania',

                                                    'Algeria': 'Algeria',

                                                    'Algerian': 'Algeria',

                                                    'Argelia': 'Argelia',

                                                    'At least 1 from Mexico': 'Mexico',

                                                    'at least 4 Syrian': 'Syria',

                                                    'Bangladesh, Burma, Myanmar': 'Bangladesh, Myanmar',

                                                    'Bangladesh': 'Bangladesh',

                                                    'Bangladeshi, Rohingya': 'Bangladesh, Myanmar',

                                                    'Brazil (12), Dominican Republic (5), Cuba (2)': 'Brazil, Dominican Republic, Cuba',

                                                    'Brazil': 'Brazil',

                                                    'Brazilian': 'Brazil',

                                                    'Burkina Faso, Malia, Guinea, Ivory Coast': 'Burkina Faso, Malia, Guinea, Ivory Coast',

                                                    'Cambodia': 'Cambodia',

                                                    'Cameroon (2) Gambia (1), Mauritius (1), Ivory Coast (2)': 'Cameroon, Gambia, Mauritius, Ivory Coast',

                                                    'Cameroon, Democratic Republic of Congo, Syria, Turkey': 'Cameroon, Democratic Republic of Congo, Syria, Turkey',

                                                    'Cameroon, Guinea': 'Cameroon, Guinea',

                                                    'Cameroon, Senegal and Ivory Coast': 'Cameroon, Senegal, Ivory Coast',

                                                    'Cameroon': 'Cameroon',

                                                    'Cameroonian': 'Cameroon',

                                                    'Central African Republic': 'Central African Republic',

                                                    'Central America': 'Central America',

                                                    'China (ethnic Uighur)': 'China',

                                                    'China': 'China',

                                                    'Comoran': 'Comoros',

                                                    'Comoros': 'Comoros',

                                                    'Congo': 'Democratic Republic of Congo',

                                                    'Costa rica': 'Costa Rica',

                                                    'Cuba or Dominican Republic': 'Cuba',

                                                    'Cuba': 'Cuba',

                                                    'Cuban': 'Cuba',

                                                    'Democratic Republic of Congo': 'Democratic Republic of Congo',

                                                    'Democratic Republic of the Congo': 'Democratic Republic of Congo',

                                                    'Domican Republic': 'Dominican Republic',

                                                    'Dominican Republic, Haiti': 'Dominic Republic, Haiti',

                                                    'Dominican Republic': 'Dominic Republic',

                                                    'Ecuador': 'Ecuador',

                                                    'Ecuadorian': 'Ecuador',

                                                    'Ecuator': 'Ecuador',

                                                    'Ecuator': 'Ecuador',

                                                    'Egypt (est.80), Ethiopia (est.150), Somalia (est.190), Sudan, Syria': 'Egypt, Ethiopia, Somalia, Sudan, Syria',

                                                    'Egypt, Ethiopia, Eritrea, Sudan, Comoros': 'Egypt, Ethiopia, Eritrea, Sudan, Comoros',

                                                    'Egypt, Syrian Arab Republic': 'Egypt, Syria',

                                                    'Egypt': 'Egypt',

                                                    'Egyptian, Eritrean, Sudanese, Syrian': ' Egypt, Eritrea, Sudan, Syria',

                                                    'El Salvador and Honduras': 'El Salvador, Honduras',

                                                    'El Salvador': 'El Salvador',

                                                    'Eritrea (2), Syria (1)': 'Eritrea, Syria',

                                                    'Eritrea or Somalia': 'Eritrea',

                                                    'Eritrea, Ethiopia, Somalia': 'Eritrea, Ethiopia, Somalia',

                                                    'Eritrea, Sudan': 'Eritrea, Sudan',

                                                    'Eritrea': 'Eritrea',

                                                    'Eritrean': 'Eritrea',

                                                    'Eritria': 'Eritrea',

                                                    'Ethiopia and Somalia': 'Ethiopia, Somalia',

                                                    'Ethiopia, Somalia': 'Ethiopia, Somalia',

                                                    'Ethiopia': 'Ethiopia',

                                                    'Ethiopian': 'Ethiopia',

                                                    'Ethnic Rohingya': 'Myanmar',

                                                    'Ethnic Yazidis (Iraq)': 'Iraq',

                                                    'Gambia (2) Guinea  Bissau (1)': 'Gambia, Guinea Bissau',

                                                    'Gambia, Ghana, Mali': 'Gambia, Ghana, Mali',

                                                    'Gambia, Nigeria and Senegal': 'Gambia, Nigeria, Senegal',

                                                    'Gambia': 'Gambia',

                                                    'Ghana, Nigeria': 'Ghana, Nigeria',

                                                    'Ghana': 'Ghana',

                                                    'Guatemala and El Salvador': 'Guatemala, El Salvador',

                                                    'Guatemala, Ecuador': 'Guatemala, Ecuador',

                                                    'Guatemala, Honduras, El Salvador': 'Guatemala, Honduras, El Salvador',

                                                    'Guatemala': 'Guatemala',

                                                    'Guatemalan': 'Guatemala',

                                                    'Guinea Conakry': 'Guinea',

                                                    'Guinea': 'Guinea',

                                                    'Guinean': 'Guinea',

                                                    'Haiti': 'Haiti',

                                                    'Haitian': 'Haiti',

                                                    'Honduran': 'Honduras',

                                                    'Honduras and Guatemala': 'Honduras, Guatemala',

                                                    'Honduras or El Salvador': 'Honduras',

                                                    'Honduras': 'Honduras',

                                                    'India': 'India',

                                                    'Indonesia': 'Indonesia',

                                                    'Iran and Iraq': 'Iran, Iraq',

                                                    'Iran': 'Iran',

                                                    'Iraq, Algeria and Syria (initial reports)': 'Iraq, Algeria, Syria',

                                                    'Iraq, Algeria, Syria': 'Iraq, Algeria, Syria',

                                                    'Iraq, Syria, Afghanistan': 'Iraq, Syria, Afghanistan',

                                                    'Iraq': 'Iraq',

                                                    'Iraqi': 'Iraq',

                                                    'Iraqui': 'Iraq',

                                                    'Ivory Coast (15), Mali (7), Senegal (5), Guinea (1), Mauritania (1)': 'Ivory Coast, Mali, Senegal, Guinea, Mauritania',

                                                    'Ivory Coast and Guinea Conakry': 'Ivory Coast, Guinea',

                                                    'Ivory Coast': 'Ivory Coast',

                                                    'Kurdistan': 'Kurdistan',

                                                    'Lebanese': 'Lebanon',

                                                    'Libya, Morocco, Syria': 'Libya, Morocco, Syria',

                                                    'Likely Comorian': 'Comoros',

                                                    'likely Comoros': 'Comoros',

                                                    'Likely Eritrea': 'Eritrea',

                                                    'Likely Rohingya': 'Myanmar',

                                                    'Madagascar': 'Madagascar',

                                                    'Maghreb': 'Maghreb',

                                                    'Mahgreb': 'Maghreb',

                                                    'Malagasy': 'Madagascar',

                                                    'Malawian': 'Malawi',

                                                    'Malaysia': 'Malaysia',

                                                    'Mali': 'Mali',

                                                    'Mexican': 'Mexico',

                                                    'Mexico': 'Mexico',

                                                    'Mixed': 'Unknown Nationality',

                                                    'Moroccan': 'Morocco',

                                                    'Morocco': 'Morocco',

                                                    'Mostly Ethiopian. Others were Somalian and 2 Yemeni crew': 'Ethiopia, Somalia, Yemen',

                                                    'mostly from Sudan, Bangladesh': 'Sudan, Bangladesh',

                                                    'Mostly from Syria, possibly some from Iraq': 'Syria, Iraq',

                                                    'Mozambican': 'Mozambique',

                                                    'Mozambique': 'Mozambique',

                                                    'Myanmar (Rohingya)': 'Myanmar',

                                                    'Myanmar, Bangladesh': 'Myanmar, Bangladesh',

                                                    'Myanmar': 'Myanmar',

                                                    'Nepal': 'Nepal',

                                                    'New Guinea': 'New Guinea',

                                                    'Nicaragua': 'Nicaragua',

                                                    'Niger (3), Mali(3), Senegal(3), Guinea(3), Ivory Coast(2), CAR(1), Liberia(1)': 'Niger, Mali, Senegal, Guinea, Ivory Coast, Central African Republic, Liberia',

                                                    'Niger': 'Niger',

                                                    'Nigeria (2) Cameroon (2)': 'Nigeria, Cameroon',

                                                    'Nigeria, Eritrea, Guinea, Gambia, Sudan, Ivory Coast, Somalia': 'Nigeria, Eritrea, Guinea, Gambia, Sudan, Ivory Coast, Somalia',

                                                    'Nigeria, Ghana, Niger': 'Nigeria, Ghana, Niger',

                                                    'Nigeria, Ivory Coast, Guinea, Sudan, Mali': 'Nigeria, Ivory Coast, Guinea, Sudan, Mali',

                                                    'Nigeria, others': 'Nigeria',

                                                    'Nigeria, Senegal': 'Nigeria, Senegal',

                                                    'Nigeria': 'Nigeria',

                                                    'Nigerian': 'Nigeria',

                                                    'Pakistan': 'Pakistan',

                                                    'Pakistani': 'Pakistan',

                                                    'Palestine': 'Palestine',

                                                    'Peru (1), Unknown (1)': 'Peru',

                                                    'Peru': 'Peru',

                                                    'Reported as "mostly" Senegal': 'Senegal',

                                                    'Reported as unspecified national of Africa': 'Africa',

                                                    'Reported as unspecified national of Central America': 'Central America',

                                                    'Reported as unspecified national of Sub-Saharan Africa': 'Africa',

                                                    'Reported as unspecified nationals of Africa': 'Africa',

                                                    'Reported as unspecified nationals of Central America' : 'Central America',

                                                    'Reported as unspecified nationals of Honduras and Guatemala' :'Honduras, Guatemala',

                                                    'Reported as unspecified nationals of Honduras' : 'Honduras',

                                                    'Reported as unspecified nationals of Horn of Africa' :'Africa',

                                                    'Reported as unspecified nationals of Rohingya'  : 'Myanmar',

                                                    'Reported as unspecified nationals of Somalia, Sudan and Nigeria' : 'Somalia, Sudan, Nigeria',

                                                    'Reported as unspecified nationals of Sub-Saharan Africa' : 'Africa',

                                                    'Reported as unspecified nationals of Syria, Afghanistan, Iraq, Iran' : 'Syria, Afghanistan, Iraq, Iran',

                                                    'Reported as unspecified nationals of the Horn of Africa' : 'Africa',

                                                    'Reported as unspecified nationals of West Africa' : 'Africa',

                                                    'Rohingya' : 'Myanmar',

                                                    'Salvadoran' : 'El Salvador',

                                                    'Senegal, Ivory Coast, Gambia, Guinea, Niger, Mali and Mauritania': 'Senegal, Ivory Coast, Gambia, Guinea, Niger, Mali, Mauritania',

                                                    'Senegal, Mali, Guinea': 'Senegal, Mali, Guinea',

                                                    'Senegal': 'Senegal',

                                                    'Somalia (1) Eritrea (1)': 'Somalia, Eritrea',

                                                    'Somalia, Afghanistan': 'Somalia, Afghanistan',

                                                    'Somalia, Eritrea, Benin, Mali': 'Somalia, Eritrea, Benin, Mali',

                                                    'Somalia, Ethiopia': 'Somalia, Eritrea',

                                                    'Somalia, Sudan and Nigeria': 'Somalia, Sudan, Nigeria',

                                                    'Somalia': 'Somalia',

                                                    'Sub-Saharan Africa' : 'Africa',

                                                    'Sudan (6), Ethiopia (2), Eritrea (1), Unknown (1)': 'Sudan, Ethiopia, Eritrea',

                                                    'Sudan, Bangladesh': 'Sudan, Bangladesh',

                                                    'Sudan': 'Sudan',

                                                    'Sudanese' : 'Sudan',

                                                    'Survivors from Burkina Faso, Malia, Guinea, and the Ivory Coast' : 'Burkina Faso, Malia, Guinea, Ivory Coast',

                                                    'Survivors from Nigeria, Ghana, Niger' : 'Nigeria, Ghana, Niger',

                                                    'Survivors from Sub-Saharan Africa' : 'Africa',

                                                    'Survivors from Syria, Iraq, and Somalia' : 'Syria, Iraq, Somalia',

                                                    'Survivors were from Gambia, Ghana and Mali' : 'Gambia, Ghana, Mali',

                                                    'Syria (mostly)': 'Syria',

                                                    'Syria, Afghanistan, Iraq, Iran': 'Syria, Afghanistan, Iraq, Iran',

                                                    'Syria, Afghanistan': 'Syria, Afghanistan',

                                                    'Syria, Iraq, Somalia': 'Syria, Iraq, Somalia',

                                                    'Syria, Iraq': 'Syria, Iraq',

                                                    'Syria, Somalia, Gambia, Ivory Coast, Mali, Tunisia, Sierra Leone, Bangladesh, Algeria, Egypt, Niger, Zambia, Ghana': 'Syria, Somalia, Gambia, Ivory Coast, Mali, Tunisia, Sierra Leone',

                                                    'Syria': 'Syria',

                                                    'Syrian and Iraqi': 'Syria, Iraq',

                                                    'Syrian Arab Republic, Egypt, Sudan': 'Syria, Egypt, Sudan',

                                                    'Syrian Arab Republic, Eritrea, Somalia, Cameroon': 'Syria, Eritrea, Somalia, Cameroon',

                                                    'Syrian Arab Republic' : 'Syria',

                                                    'Syrian Kurds': 'Syria',

                                                    'Syrian, Egyptian, other African': 'Syria, Egypt',

                                                    'Syrian, Iraqi, Afghan': 'Syria, Iraq, Afghanistan',

                                                    'Syrian. An infant was among the dead.': 'Syria',

                                                    'Syrian': 'Syria',

                                                    'Tunisia': 'Tunisia',

                                                    'Uknown': 'Unknown Nationality',

                                                    'Unknown (5 reported as nationals from Sub-Saharan Africa and 1 from Morocco)': 'Africa, Morocco',

                                                    'Unknown (Kurdish)' : 'Iran',

                                                    'Unknown (skeletal remains)': 'Unknown Nationality',

                                                    'Unknown Nationality': 'Unknown Nationality',

                                                    'Unknown. Survivors all from Sub-Saharan African nations.': 'Africa',

                                                    'Unknown. Survivors from Bangladesh, Burma, or are ethnic Rohingya': 'Bangladesh, Myanmar',

                                                    'Unkown': 'Unknown Nationality',

                                                    'Unnknown': 'Unknown Nationality',

                                                    'Unspecfied nationals of Western Africa': 'Africa',

                                                    'Unspecified national of Sub-Saharan Africa': 'Africa',

                                                    'Unspecified nationalities of Sub-Saharan Africa': 'Africa',

                                                    'Unspecified nationalities of Subsaharan Africa': 'Africa',

                                                    'Unspecified nationality of North Africa': 'Africa',

                                                    'Unspecified nationality of Sub-Saharan Africa': 'Africa',

                                                    'Unspecified nationals of Africa': 'Africa',

                                                    'Unspecified nationals of Sub-Saharan Africa': 'Africa',

                                                    'Unspecified nationals of West Africa': 'Africa',

                                                    'Various': 'Unknown Nationality',

                                                    'Venezuela': 'Venezuela',

                                                    'Venezulean': 'Venezuela',

                                                    'Zimbabwe': 'Zimbabwe',

                                                    "'African'": 'Africa',

                                                    "'Mostly African'": 'Africa',

                                                    "'Sub-Saharan African'": 'Africa'})
len(data['Nationality'].unique()) 
set(data['Nationality']) 
len(data['MissingNumberOfPeople'].unique())
data['MissingNumberOfPeople'].head()
print ('Percentage of NAN in Missing Number of People: ', data['MissingNumberOfPeople'].isna().mean().round(4) * 100,'%')

print ('Total Data points of NAN in Missing Number of People: ', data['MissingNumberOfPeople'].isna().sum())
data['MissingNumberOfPeople'] = data['MissingNumberOfPeople'].fillna(0)

data['MissingNumberOfPeople'] = data['MissingNumberOfPeople'].astype(int)
data['MissingNumberOfPeople'].head()
data['DeadNumberOfPeople'].head()
len(data['DeadNumberOfPeople'].unique())   
print ('Percentage of NAN in Dead Number of People: ', data['DeadNumberOfPeople'].isna().mean().round(4) * 100,'%')

print ('Total Data points of NAN in Dead Number of People: ', data['DeadNumberOfPeople'].isna().sum())
data['DeadNumberOfPeople'].fillna(data['DeadNumberOfPeople'].mode()[0], inplace=True)
data['DeadNumberOfPeople'].head()
data['DeadNumberOfPeople'] = data['DeadNumberOfPeople'].astype(int)
len(data['IncidentRegion'].unique()) 
set(data['IncidentRegion'])           
data['IncidentRegion'] = data['IncidentRegion'].replace({'Central America incl. Mexico':'Central America',

                                                         'Middle East ': 'Middle East',

                                                         'Southeast Asia': 'South East Asia'})

data['IncidentRegion'].fillna('Unknown', inplace = True) 
set(data['IncidentRegion'])    
len(data['IncidentRegion'].unique()) 
len(data['Date'].unique())                       
data['Date'] = pd.to_datetime(data['Date'])

data['Date'].head()
print (data['Date'].min())

print (data['Date'].max())
len(data['Source'].unique())                     
set(data['Source'])   
data['Source'].fillna('Unknown', inplace = True) 
data['Source'] = data['Source'].replace({'AFP and Sofia News Agency': 'AFP',

                                         'AFP and the Guardian': 'AFP',

                                         'AFP, Reuters': 'AFP',

                                         'AM León (Mexico)': 'AM Leon',

                                         'ANSA English': 'ANSA',

                                         'ANSA Italy': 'ANSA',

                                         'ANSA Med': 'ANSA',

                                         'ANSA Med and Relief Web': 'ANSA',

                                        'AP, BBC': 'AP',

                                         'Afghan immigrant dies in back of truck': 'Unknown Source',

                                        'Agence France Presse': 'AFP',

                                         'Al Jazeera, AFP, BBC': 'Al Jazeera',

                                        'Aljazeera, Telegraph, IOM, UNHCR': 'Al Jazeera',

                                         'Andalucía Informacion and Europa Sur': 'Andalucía Informacion',

                                         'Andulucia Information': 'Andalucía Informacion',

                                         'Armada Espagnola, Diario Sur': 'Armada Espagnola',

                                         'Asociación Pro Derechos Humanos De Andalucía (APDHA)': 'APDHA',

                                         'Asociación Pro Derechos Humanos de Andalucía': 'APDHA',

                                         'Asociación Pro Derechos Humanos de Andalucía (APDHA)': 'APDHA',

                                         'Associated Press': 'AP',

                                         'Associated Press and Hellenic Coast Guard': 'AP',

                                         'Associated Press and IOM Ankara': 'AP',

                                         'BBC (UNHCR)': 'BBC',

                                         'BBC and CBC': 'BBC',

                                         'BBC, UNHCR': 'BBC',

                                         'BBC/CBC': 'BBC',

                                          "Blog de veille sur les droits de l'Homme en Serbie, B92": 'B92',

                                         'Central Noticias Imagen del Golfo': 'Central Noticias',

                                         'China Central Television (eng); China News Service (eng) (ECNS)': 'ECNS',

                                         'Daily Sabah and UNHCR': 'Daily Sabah',

                                         'Daily Sabha': 'Daily Sabah',

                                         'Departmento 19': 'Departamento 19',

                                        'Desaparecidos y sin reclamar en la frontera': 'Facebook',

                                         'Desaparecidos y sin reclamar en la frontera via Facebook': 'Facebook',

                                        'Diario de Cuba & El Colombiano': 'Diario de Cuba',

                                         'EUBusiness (Agence France-Presse)': 'AFP',

                                         'El Manana': 'El Mañana',

                                         'El Mexico, Record Chiapas': 'El Mexico',

                                         'El Pulmondelademocracia': 'El Pulmon de la Democracia',

                                         'El manana': 'El Mañana',

                                         'El mañana': 'El Mañana',

                                         'El mundo': 'El Mundo',

                                         'El norte and El Diario': 'El Norte', 

                                         'ElBuscapersonas David Nostas (via Facebook)': 'El Buscapersonas David Nostas',

                                         'Elbuscapersonas David Nostas (via Facebook)': 'El Buscapersonas David Nostas',

                                         'Euronews, Hurriyet Daily News': 'Euronews',

                                        'Expreso.press': 'Expreso',

                                          'Facebook Desaparecidos y sin reclamar en la frontera': 'Facebook',

                                          'Ghana WEb and Critica .com': 'Ghana Web',

                                          'Global post': 'Global Post',

                                          'Greek Reporter, AFP': 'Greek Reporter',

                                        'Grupo Sieno, Twitter': 'Grupo Sieno',

                                         'Guanajuato Informa and Facebook': 'Guanajuato Informa',

                                          'Hellenic Coast Guard': 'Hellenic Coast Guard',

                                         'Hellenic Coast Guard via IOM Athens': 'Hellenic Coast Guard',

                                         'Hellenic Coast Guard via IOM Greece': 'Hellenic Coast Guard',

                                         'Hellenic Coast Guard, via IOM Athens': 'Hellenic Coast Guard',

                                          'Horacero and El Salvador Times': 'Horacero',

                                          'Hurriyet Daily News, Anatolu Agency': 'Hurriyet Daily News',

                                          'ICRC via IOM Libya': 'IOM Libya',

                                          'IOM Athens and Migrant Report': 'IOM Athens',

                                         'IOM Athens and Reuters': 'IOM Athens',

                                         'IOM Greece according to Greek government sources': 'IOM Greece',

                                         'IOM Greece and AP': 'IOM Greece',

                                         'IOM Greece and Hellenic Coast Guard': 'IOM Greece',

                                         'IOM Greece, AP': 'IOM Greece',

                                         'IOM Greece, Reuters': 'IOM Greece',

                                         'IOM Greece/AFP': 'IOM Greece',

                                         'IOM Italy, BBC': 'IOM Italy',

                                         'IOM Italy, UNHCR': 'IOM Italy',

                                         'IOM Libya, Libyan Red Crescent': 'IOM Libya',

                                         'IOM Madagascar, LINFO.re': 'IOM Madagascar',

                                          'IOM Morroco': 'IOM Morocco',

                                         'IOM Rome Office': 'IOM France',

                                         'IOM Rome and Reuters': 'IOM Rome',

                                         'IOM Rome and Yahoo! News': 'IOM Rome',

                                         'IOM Rome, UNHCR': 'IOM Rome',

                                          'IOM Tunisa': 'IOM Tunisia',

                                          'IOM Tunisia via Tunisian Coast Guard': 'IOM Tunisia',

                                         'IOM Tunisia, Espace Manager': 'IOM Tunisia',

                                          'IOM Turkey via TCGC': 'IOM Turkey',

                                          'IOM Yemen, UNHCR': 'IOM Yemen',

                                          'IOM and Daily Star': 'IOM',

                                         'IOM, Italian Coast Guard, Al Wasat': 'IOM Italy',

                                         'IOM/Coast Guard Command/Indian Express': 'IOM',

                                         'IOMLybia': 'IOM Libya',

                                         'IOm Djibouti': 'IOM Djibouti',

                                          'Infonogales': 'InfoNogales',

                                         'LRC Az-Zawiyah via IOM Libya': 'IOM Libya',

                                         'LRC Sabratha via IOM Libya': 'IOM Libya',

                                         'La Jornada': 'La Jornada',

                                         'La Jornada, Líder Informativo': 'La Jornada',

                                         'La Journada': 'La Jornada',

                                         'La Policiaca': 'La Policiaca',

                                         'La Policiaca and Cuarto de Guerra': 'La Policiaca',

                                         'La Prensa and NAM  News Network': 'La Prensa',

                                        'Libyan Red Crescent Az Zawiyah via IOM Libya': 'Libyan Red Crescent',

                                          'Lider Web Infromativo': 'Lider Web Informativo',

                                        'Local 10 - ABC News': 'Local 10',

                                        'MiMorelia': 'Mi Morelia',

                                         'Minitry of External Relations, Government of Guatemala': 'Ministry of External Relations, Government of Guatemala',

                                          'Nogales International and PCOME': 'Nogales International',

                                          'OHCHR - Report of OHCHR mission to Bangladesh': 'OHCHR',

                                        'Periodico Se manalo Frontera': 'Periodico Se Manalo Frontera',

                                         'Periodico de manano Frontera': 'Periodico Se Manalo Frontera',

                                          'Phuketwan': 'Phuket Wan',

                                         'Pima County Office of the Medical Examinder': 'Pima County Office of the Medical Examiner',

                                          'RTE News and Turkish Coast Guard': 'RTE News',

                                         'Radio Cadena Voces (via Facebook)': 'Radio Cadena Voces',

                                         'Radio Eco Digital via Facebook': 'Radio Eco Digital',

                                          'Relief Web/UNHCR': 'Relief Web',

                                         'Report of OHCHR mission to Bangladesh': 'OHCHR',

                                         'Reporte Tamaulipas via Facebook and Contacto': 'Reporte Tamaulipas',

                                        'Reuters EspaÃ±a': 'Reuters',

                                         'Reuters and Al Jazeera': 'Reuters',

                                         'Reuters and Middle East eye': 'Reuters',

                                         'Reuters, Hellenic Coast Guard': 'Reuters',

                                         'Reuters, IOM Bulgaria': 'Reuters',

                                         'Reuters, IOM Italy': 'Reuters',

                                         'Reuters, New York Times': 'Reuters',

                                         'Reuters, SBS': 'Reuters',

                                         'Reynosa Codigo Rojo via Facebook': 'Reynosa Codigo Rojo',

                                        'SOS Mediterranee, UNHCR': 'SOS Mediteranee',

                                         'Salvamento Maritimo via Twitter and Entre Fronteras': 'Salvamento Marítimo',

                                         'Salvamento Maritimo via Twitter and La Voz Libre': 'Salvamento Marítimo',

                                         'Salvamento Marítimo': 'Salvamento Marítimo',

                                        'Santa Fe (radio) and Noticias RCN': 'Santa Fe (Radio)',

                                        'South Austin Patch, Guanajuato Informa': 'South Austin Patch',

                                        'Spanish Ministry of the Interior via IOM Spain': 'IOM Spain',

                                        'Still under investigation, but reports passed on via IOM Italy. Also the Guardian': 'IOM Italy',

                                        'Sun Sentinel, Miami Herald': 'Sun Sentinel',

                                        'Super Channel 12 via Facebook': 'Super Channel 12',

                                        'The World Post (Huffington and BIG)': 'The World Post',

                                        'Turkish Coast Guard via IOM Ankara': 'Turkish Coast Guard',

                                         'Turkish Coast Guard via IOM Turkey': 'Turkish Coast Guard',

                                         'Twitter and Chanel 6': 'Chanel 6',

                                         'UBAlert, Chronicle.co.zw': 'UBAlert',

                                        'UNHCR News Today': 'UNHCR',

                                         'UNHCR and Migrant Report': 'UNHCR',

                                         'UNHCR, El Confidencial': 'UNHCR',

                                         'UNHCR, IOM Italy': 'UNHCR',

                                        'Xinhua, The Pappas Post': 'Xinhua',

                                        'Zegabi, IOM Zambia': 'Zegabi',

                                        'Zocalo, Maverick County Sheriff': 'Zocalo'})
len(data['Source'].unique())          
data['Source'].head()
set(data['Reliability'])                
data['Reliability'] = data['Reliability'].replace({'Partially verified':'Partially Verified'})

data['Reliability'].fillna('Unknown', inplace = True) 
set(data['Reliability'])  
len(data['Latitud'].unique())                    
data['Latitud'].isna().sum()
len(data['Longitud'].unique())   
data['Longitud'].isna().sum()
data.isna().mean().round(4) * 100
data.head()
data.shape
data.info()
#Save clean datas set

data.to_csv('data.csv')
#Load data set

immigrants = pd.read_csv('data.csv', sep=',', index_col=0)

immigrants.head()
immigrants.info()
# Create new columns

immigrants['Date']  = pd.to_datetime(immigrants['Date'], errors='coerce')

immigrants['Day']   = immigrants['Date'].dt.day

immigrants['Month'] = immigrants['Date'].dt.month

immigrants['Year']  = immigrants['Date'].dt.year
immigrants.head()
sns.heatmap(immigrants.corr(), annot=True);
sns.set_color_codes("bright")

sns.barplot(x="Year", y="DeadNumberOfPeople", data=immigrants,

            label="Year Suicides", palette="BuGn_r")

plt.xticks(rotation=90);
sns.set_color_codes("bright")

sns.barplot(x="Month", y="DeadNumberOfPeople", data=immigrants,

            label="Year Suicides", palette="GnBu_d")

plt.xticks(rotation=90);
sns.set_color_codes("muted")

sns.barplot(x="Day", y="DeadNumberOfPeople", data=immigrants,

            label="Year Suicides", palette='husl')

plt.xticks(rotation=90);
sns.barplot(x='Reliability', y='DeadNumberOfPeople', hue='Year', data=immigrants);
sns.lineplot(x='Month', y='MissingNumberOfPeople', data=immigrants)

sns.lineplot(x='Month', y='DeadNumberOfPeople', data=immigrants)



_ = plt.legend(['MissingNumberOfPeople', 'DeadNumberOfPeople'])
immigrants['RegionOfOrigin'].value_counts().plot(kind = "bar", title = "Region of Origin");
immigrants['CauseOfDeath'].value_counts().head(15).plot(kind = "bar", title = "Reason for Death");
immigrants['IncidentRegion'].value_counts().plot(kind = "bar", title = "Incident Region");
immigrants.groupby('RegionOfOrigin')['DeadNumberOfPeople'].sum().to_frame()
immigrants.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
#Get Label Encoder variables per categorical variables in the dataset

immigrants_rates = immigrants

immigrants_rates['CauseOfDeath'] = le.fit_transform(immigrants.CauseOfDeath)

immigrants_rates['RegionOfOrigin'] = le.fit_transform(immigrants.RegionOfOrigin)

immigrants_rates['Nationality'] = le.fit_transform(immigrants.Nationality)

immigrants_rates['IncidentRegion'] = le.fit_transform(immigrants.IncidentRegion)

immigrants_rates['Source'] = le.fit_transform(immigrants.Source)

immigrants_rates['Reliability'] = le.fit_transform(immigrants.Reliability)

immigrants_rates.head()
immigrants_rates.info()
from sklearn.model_selection import train_test_split

import lightgbm as lgb # Light GBM is a gradient boosting framework that uses tree based learning algorithm

import shap



# print the JS visualization code to the notebook

shap.initjs()
X = immigrants_rates[['CauseOfDeath', 'RegionOfOrigin', 'Nationality',

       'MissingNumberOfPeople', 'DeadNumberOfPeople', 'IncidentRegion', 

       'Source', 'Latitud', 'Longitud', 'Day', 'Month', 'Year']]

y = immigrants_rates[[ 'Reliability']]



# create a train/test split

train_x, test_x, train_y, test_y = train_test_split(X,y, test_size = 0.2,random_state=7)

print(train_x.shape, train_y.shape)

print(test_x.shape, test_y.shape)
d_train = lgb.Dataset(train_x, label=train_y)

d_test = lgb.Dataset(test_x, label=test_y)
params = {

    "max_bin": 512,

    "learning_rate": 0.05,

    "boosting_type": "gbdt",

    "objective": "binary",

    "metric": "binary_logloss",

    "num_leaves": 10,

    "verbose": -1,

    "min_data": 100,

    "boost_from_average": True

}



model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[:1000,:], X.iloc[:1000,:])
shap.summary_plot(shap_values, X)