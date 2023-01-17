import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

# from control import *

from  clean import *

from  graphs import *

from jengineer import *

from python_expansion import *
data = pd.read_csv("/kaggle/input/build-characteristics-corona-patients-part-a/Characteristics_Corona_patients - 06.10.2020 2.csv",

                   parse_dates=['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"])
data.shape
%%time

# 2min 44s

for j in data.columns:

    data[j] = data[j].apply(lambda x: x if x==x else np.nan)
data
%%time

for j in ['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"]:

    name = j+"_D"

    zero = '1.1.2019'

    JEngineer.interval_date_datetime_col(data, j, zero , name )
%%time

JEngineer.interval_datatime_cols(data,"confirmed_date","return_date", "return_date_until_confirmed_date")

print(3)

JEngineer.interval_datatime_cols(data, "confirmed_date", "date_onset_symptoms", "date_onset_symptoms_until_confirmed_date")

print(3)

JEngineer.interval_datatime_cols(data,"released_date","confirmed_date", "confirmed_date_until_released_date")

print(3)

JEngineer.interval_datatime_cols(data,"deceased_date","confirmed_date", "confirmed_date_until_deceased_date")

print(3)

 
%%time

# 6min 30s

# togther Wall time: 10min 14s

indexs = data.index[data.infected_by.notnull()]



data["len_people_infected_by_patient"] =  0 

data["severity_illness_infectious_person"] = np.nan



for indx in indexs:

    i  = data.infected_by[indx]

    i = i.split(",")

    i = [x.strip() for x in i]

    i = Clean.remove_from_ls(i , "")

    

    for x in i:

        ind = float(x)

        data.loc[ind, "len_people_infected_by_patient"] =  data.loc[ind, "len_people_infected_by_patient"] +1 

    

    if len(i) == 1:

        ind = float(i[0])

        data.loc[indx, "severity_illness_infectious_person"] =  data.severity_illness[ind]

    

    elif len(i) > 1:

        pass

    

    

# make 0 to np.nan

data.loc[data["len_people_infected_by_patient"] ==0, "len_people_infected_by_patient"] = np.nan
data.to_csv("Characteristics_Corona_patients version 5 6-10-20.csv", index = False)

data.to_csv()