import re

import math

import calendar

import numpy as np

import pandas as pd



from pprint import pprint

from datetime import datetime, date



csv = pd.read_csv("../input/PakistanDroneAttacksWithTemp Ver 4.csv", encoding="latin1")



csv.head()
a = list(set(csv["Latitude"]))[0]

b = list(set(csv["Latitude"]))[1]



#remove rows with empty latitude or empty longitude

clean_data_set = csv[csv["Latitude"].isnull() != True]

clean_data_set = clean_data_set[clean_data_set["Longitude"].isnull() != True]
#set the timestamp at the date place

def get_day_name(s):

    return s.split(', ')[0]



month = {

    "January": 1,

    "February": 2,

    "March": 3,

    "April": 4,

    "May": 5,

    "June": 6,

    "July": 7,

    "August": 8,

    "September": 9,

    "October": 10,

    "November": 11,

    "December": 12

}



def get_month(s):

    try:

        return month[(s.split(" ")[1].split()[0])]

    except:

        return -1



def get_day(s):

    try:

        return int(s.split(", ")[1].split()[1])

    except:

        try:

            return int(s.split(", ")[2])

        except:

            return -1

    

def get_year(s):

    try:

        return int(s.split(" ")[-1])

    except:

        return -1



def get_timestamp(s):

    try:

        return int(date(get_year(s), get_month(s), get_day(s)).strftime("%s"))

    except:

        return -1





clean_data_set["day"] = clean_data_set["Date"].apply(get_day_name)

clean_data_set["timestamp"] = clean_data_set["Date"].apply(get_timestamp)



clean_data_set = clean_data_set.drop(["Date", "Time", "Temperature(F)", "Province"], axis=1)



clean_data_set.head()
clean_data_set.keys()
#fill the nan values of nb killed: estimate the number of persons killed



min_number_killed_average = clean_data_set["Total Died Min"].mean()

max_number_killed_average = clean_data_set["Total Died Mix"].mean()



def estimate_number_killed(row):

    if math.isnan(row["Total Died Min"]) and math.isnan(row["Total Died Mix"]):

        return 0

    elif math.isnan(row["Total Died Min"]):

        if min_number_killed_average < row["Total Died Mix"]:

            return (min_number_killed_average + row["Total Died Mix"] * 2) / 3. #set bigger coeff to the know value

        else:

            return row["Total Died Mix"] #if the average is bigger than the maxx kill, return the max kill

    elif math.isnan(row["Total Died Mix"]):

        if max_number_killed_average > row["Total Died Min"]:

            return (max_number_killed_average + row["Total Died Min"] * 2) / 3.

        else:

            return row["Total Died Min"]

    else:

        return (row["Total Died Mix"] + row["Total Died Min"]) / 2.





clean_data_set["nb_killed"] = clean_data_set.apply(lambda r: int(estimate_number_killed(r)), axis=1)

clean_data_set = clean_data_set.drop(["Civilians Min", "Civilians Max", "Foreigners Min", "Foreigners Max", "Total Died Min", "Total Died Mix"], axis=1)



clean_data_set.head()
#estimate the nb of injured people

min_number_inj_average = clean_data_set["Injured Min"].mean()

max_number_inj_average = clean_data_set["Injured Max"].mean()



def estimate_number_injured(row):

    if math.isnan(row["Injured Min"]) and math.isnan(row["Injured Max"]):

        return 0

    elif math.isnan(row["Injured Min"]):

        if min_number_inj_average < row["Injured Max"]:

            return (min_number_inj_average + row["Injured Max"] * 2) / 3. 

        else:

            return row["Injured Max"]

    elif math.isnan(row["Injured Max"]):

        if max_number_inj_average > row["Injured Min"]:

            return (max_number_inj_average + row["Injured Min"] * 2) / 3.

        else:

            return row["Injured Min"]

    else:

        return (row["Injured Max"] + row["Injured Min"]) / 2.





clean_data_set["nb_injured"] = clean_data_set.apply(lambda r: int(estimate_number_injured(r)), axis=1)

clean_data_set = clean_data_set.drop(["Injured Min", "Injured Max"], axis=1)



clean_data_set.head()
#estimate the nb of terro

al_qaeda = clean_data_set["Al-Qaeda"].mean()

taliban = clean_data_set["Taliban"].mean()



def number_terro(row):

    if math.isnan(row["Al-Qaeda"]) and math.isnan(row["Taliban"]):

        return 0

    elif math.isnan(row["Al-Qaeda"]):

        return row["Taliban"]

    elif math.isnan(row["Taliban"]):

        return row["Al-Qaeda"]

    else:

        return (row["Al-Qaeda"] + row["Taliban"])





clean_data_set["nb_terro_estimate"] = clean_data_set.apply(lambda r: int(number_terro(r)), axis=1)

clean_data_set = clean_data_set.drop(["Al-Qaeda", "Taliban", "Special Mention (Site)", "Comments", "References"], axis=1)



clean_data_set.head()
clean_data_set = clean_data_set.drop(["Women/Children  "],axis=1)

clean_data_set.head()
def civil_killed(row):

    return (row["nb_killed"] - row["nb_terro_estimate"])



clean_data_set["nb_killed"] = clean_data_set.apply(lambda r: int(civil_killed(r)), axis=1)

clean_data_set.head()