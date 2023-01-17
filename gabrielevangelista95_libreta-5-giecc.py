import requests
import json
import pandas as pd
from matplotlib import pyplot as plt
API_URL = 'https://4ps63tr0b0.execute-api.us-east-2.amazonaws.com/thisStage/skn?'
device_ID = "dev002"
query_Date = "2019-06-20"

api_full_url = API_URL + "device=" + device_ID + "&date=" + query_Date

print("Full API URL : {}".format(api_full_url))
r = requests.get(api_full_url)
r.text
json_response = r.json()

print("Respuesta en formato JSON : \n\n{}".format(json_response))
for response in json_response:
    print(response,"\n\n")
table_response = pd.DataFrame(json.loads(json_response[0]))
table_response
num_of_jsons = len(json_response)

print("Numero de objetos JSON regresados : {}".format(num_of_jsons))
table_response = []

for j_response in json_response:
    new_table = pd.DataFrame(json.loads(j_response))
    table_response.append(new_table)
pd.concat(table_response)
pd.concat([pd.DataFrame(json.loads(j_response)) for j_response in json_response], ignore_index=True)
table_response = pd.concat([pd.DataFrame(json.loads(j_response)) for j_response in json_response], ignore_index=True)
table_response.sort_values(by ='publishTime' , ascending=True)
table_response.sort_values(by ='recordTime' , ascending=False)
table_response_sorted = table_response.sort_values(by ='recordTime' , ascending=False)
table_response_sorted["recordTime"]
real_times = []
for elmnt in table_response_sorted["recordTime"]:
    digits = len(elmnt)
    if digits == 4:
        real_time = "0:"+elmnt[0:2]+":"+elmnt[2:4]
    elif digits == 5:
        real_time = elmnt[0]+":"+elmnt[1:3]+":"+elmnt[3:5]
        
    real_times.append(real_time)
    print(real_time)

real_times = []

for elmnt in table_response["recordTime"]:
    digits = len(elmnt)
    if digits == 4:
        real_time = "0:"+elmnt[0:2]+":"+elmnt[2:4]
    elif digits == 5:
        real_time = elmnt[0]+":"+elmnt[1:3]+":"+elmnt[3:5]
        
    real_times.append(real_time)
real_time = pd.DataFrame(real_times,columns=["real_times"])
real_time
table_response_2 = table_response.join(real_time)
table_response_2
table_response_sorted = table_response_2.sort_values(by ='real_times' , ascending=True)
table_response_sorted
table_response_sorted.columns
time_vals = table_response_sorted["real_times"]
mafs_vals = table_response_sorted["maf"]
rpms_vals = table_response_sorted["rpm"]
vels_vals = table_response_sorted["vel"]
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:%M')
import datetime
date_times = [datetime.datetime.strptime(time_val,"%H:%M:%S") for time_val in time_vals]
plt.figure(figsize=(15,6))
plt.plot(date_times[0:-2],vels_vals[0:-2])
plt.title("Velocidad en km/h")
plt.xlabel("Time")
plt.ylabel("km / h")

plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(myFmt)

plt.show()
plt.figure(figsize=(15,6))
plt.plot(date_times[0:-2],mafs_vals[0:-2])
plt.title("MAF en grams / sec")
plt.xlabel("Time")
plt.ylabel("grams / sec")

plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(myFmt)

plt.show()
plt.figure(figsize=(15,6))
plt.plot(date_times[0:-2],rpms_vals[0:-2])
plt.title("RPM")
plt.xlabel("Time")
plt.ylabel("rpm")

plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(myFmt)

plt.show()
