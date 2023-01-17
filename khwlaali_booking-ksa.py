import pandas as pd
import numpy as np
df = pd.read_csv("../input/bookingcom-saudi-cities/booking_saudi_cities_cleaned.csv")
df.head()


df = df.iloc[:, 1:] # only need my data colum 
df.head()
df.isnull().sum() # I found null values and I want to fill them
# clean non null values to get the average and fill null values with average
for i in range(len(df[" Review"])):
    try:
        df[" Review"][i] = df[" Review"][i].split(">")[1].split("</div")[0]
    except:
        pass
#df[" Review"]
# get the average
df[" Review"].astype(float).sum()/df[" Review"].count() # the average = 7.6
# fill the average
df[" Review"].fillna("7.6", inplace=True)
# clean Area
df["Are"] = df["Are"].apply(lambda x: x.split("\n")[-1]) 
# clean price
df[" price"] = df[" price"].apply(lambda x:x.split()[-1])
hotels = [] # to storge
for hotel in range(len(df["facilities"])):
    hotel_facility = [] # to storge as list 
    for i in range(len(df["facilities"][hotel].split("<div"))):
        try: # put try for if there are problem inside it  and except to pass the error 
            f = df["facilities"][hotel].split("<div")[i].split("svg")[-1].split("\n")[1]
            if f[0] != ">" and f[0] != "<": # facilities has a  > < , i put this condition 
                hotel_facility.append(f)
        except:
            pass
    hotels.append(hotel_facility) # to add in hotel_facility
    hotel_facility = [] # for cleaning  
hotels = []
for facility in range(len(df["facilities"])):
    hotel_facility = []
    for i in range(len(df["facilities"][facility].split("<div"))):
        try: 
            fac = df["facilities"][facility].split("<div")[i].split("svg")[-1].split("\n")[1]
            if fac[0] != "<" and fac[0] != ">":
                hotel_facility.append(fac)
        except:
            pass
        hotels.append(hotel_facility)
        hotel_facility = []

df["facilities"] = pd.DataFrame({"facilities":hotels}) # here make replace for the old colum to anew colum (means clean colum)
hotel_check = []
for n in range(len(df["checklist_facilities"][0].split("<div"))):
    z= "" # empty
    try:
        l = df["checklist_facilities"][0].split("<div")[n].split("svg")[-1].split("\n")
        for item in l :
            if ">" not in item and ">" not in item:
                if len(item) > 1:
                    z+=item+":" # to add iteam (varible :value )
        if len(z)> 0:
            hotel_check.append(z)
        z = ""
    except:
        pass
    
#print(hotel_check)
    
hotels_checklist = []
for i in range(len(df["checklist_facilities"])):
    hotel_check = []
    for n in range(len(df["checklist_facilities"][i].split("<div"))):
        z = ""
        try:
            line = df["checklist_facilities"][i].split("<div")[n].split("svg")[-1].split("\n")
            for item in line:
                if "<" not in item and ">" not in item:
                    if len(item) > 1:
                        z+= item+":"
            if len(z)> 0:
                hotel_check.append(z)
            z = ""
        except:
            pass
    hotels_checklist.append(hotel_check)
    hotel_check= []
df["checklist_facilities"] = pd.DataFrame({"checklist_facilities":hotels_checklist}) # here make replace for the old colum to anew colum (means clean colum)
df.head()
#df
df.to_csv("booking_saudi_cities_cleaned.csv")