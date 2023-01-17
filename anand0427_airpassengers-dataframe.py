
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/AirPassengers.csv")
df.head()
df1 = pd.DataFrame() 
for i in df.items():
    print(i)
dict = {"Jan":[],"Feb":[],"Mar":[],"Apr":[],"May":[],"Jun":[],"Jul":[],"Aug":[],"Sep":[],"Oct":[],"Nov":[],"Dec":[]}
for i in df.itertuples():
    if i[1].split("-")[1] == "01":
        dict["Jan"].append(i[2])
    elif i[1].split("-")[1] == "02":
        dict["Feb"].append(i[2])
    elif i[1].split("-")[1] == "03":
        dict["Mar"].append(i[2])
    elif i[1].split("-")[1] == "04":
        dict["Apr"].append(i[2])
    elif i[1].split("-")[1] == "05":
        dict["May"].append(i[2])
    elif i[1].split("-")[1] == "06":
        dict["Jun"].append(i[2])
    elif i[1].split("-")[1] == "07":
        dict["Jul"].append(i[2])
    elif i[1].split("-")[1] == "08":
        dict["Aug"].append(i[2])
    elif i[1].split("-")[1] == "09":
        dict["Sep"].append(i[2])
    elif i[1].split("-")[1] == "10":
        dict["Oct"].append(i[2])
    elif i[1].split("-")[1] == "11":
        dict["Nov"].append(i[2])
    elif i[1].split("-")[1] == "12":
        dict["Dec"].append(i[2])
df = pd.DataFrame(dict) 
df
df.to_csv("AirPassengers.csv",index = False)
