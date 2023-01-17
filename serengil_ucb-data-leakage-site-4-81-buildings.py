import numpy as np

import pandas as pd

import requests

from tqdm import tqdm

from datetime import date, datetime, time, timedelta

import matplotlib.pyplot as plt



pd.set_option('display.max_rows',100)
locations = []

for i in range(173, 292+1):

    locations.append(i)



locations.append(3727)

locations.append(5824)

locations.append(5992)
if False:

    metadata_list = []



    pbar = tqdm(range(0, len(locations)))

    for index in pbar:

        location = locations[index]

        location_metadata = []

        http_address = "http://engagementdashboard.com/a/location/metadata?locationIds=%s" % str(location)

        #print("Calling ",http_address)



        #try max 3 times

        try:

            resp = requests.get(http_address, timeout=120)

        except:

            try:

                resp = requests.get(http_address, timeout=120)

            except:

                resp = requests.get(http_address, timeout=120)



        if resp.status_code == 200:

            resp_json = resp.json()[0]



            #print(resp_json)



            squareFeet = -1; year = -1

            name = ""; spaceUse = ""; address = ""; timezone = ""; resources = ""



            try: squareFeet = resp_json['areaNumber'] 

            except: pass



            try: year = resp_json['yearConstructed']

            except: pass



            try: name = resp_json["name"]

            except: pass



            try: spaceUse = resp_json["spaceUse"]

            except: pass



            try: address = resp_json["address"]

            except: pass



            try: timezone = resp_json["timeZone"]

            except: pass



            try: resources = resp_json["resources"]

            except: pass



            location_metadata.append(location)

            location_metadata.append(squareFeet)

            location_metadata.append(year)

            location_metadata.append(name)

            location_metadata.append(spaceUse)

            location_metadata.append(address)

            location_metadata.append(timezone)

            location_metadata.append(resources)



            metadata_list.append(location_metadata)
#source = pd.DataFrame(metadata_list, columns = ['source_id', 'square_feet', 'year', 'name', 'spaceUse', 'address', 'timezone', 'resources'])
#source.head(10)
target = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
target.head()
#we believe that UC Berkeley is site id 4. Filter target data frame for just site 4.

target = target[target['site_id'] == 4]
print("There are ",len(target.building_id.unique())," buildings in ASHRAE data set")
ashrae_buildings = target.building_id.unique()
train_df = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

train_df = train_df[train_df.building_id.isin(ashrae_buildings)]
train_df.head()
def retrieveConsumptions(locations): 

    consumption_list = []

    for index in range(0, len(locations)):

        location = locations[index]

        print(index+1,'/',len(locations), ".",int(location), ".", end='')

        for year in range(2016, 2020):

            print(year, "",end= '')

            for half in range(0, 2):

                if year == 2019 and half == 1: #2019 2nd period is not completed yet (today is 2019-11-23)

                    break

                if half == 0:

                    print("1/2 ", end='')

                    if year == 2019:

                        http_address = "https://engagementdashboard.com/a/consumption?endTime="+str(year)+"-01-01T23:59:59.000Z&granularity=HOUR&locationIds="+str(int(location))+"&resource=Electricity&startTime="+str(year)+"-01-01T00:00:00.000Z"

                    else:

                        http_address = "https://engagementdashboard.com/a/consumption?endTime="+str(year)+"-06-31T23:59:59.000Z&granularity=HOUR&locationIds="+str(int(location))+"&resource=Electricity&startTime="+str(year)+"-01-01T00:00:00.000Z"

                else:

                    print("2/2. ", end='')

                    http_address = "https://engagementdashboard.com/a/consumption?endTime="+str(year)+"-12-31T23:59:59.000Z&granularity=HOUR&locationIds="+str(int(location))+"&resource=Electricity&startTime="+str(year)+"-07-01T00:00:00.000Z"



                #print(http_address)

                

                #try maximum 4 times

                try:

                    resp = requests.get(http_address, timeout=120)

                except:

                    try:

                        resp = requests.get(http_address, timeout=120)

                    except:

                        try:

                            resp = requests.get(http_address, timeout=120)

                        except:

                            resp = requests.get(http_address, timeout=120)

                

                if resp.status_code == 200:

                    resp_json = resp.json()[0]

                    #print(resp_json)

                    consumptions = resp_json["actual"]["data"]

                    #consumptions = resp_json["baseline"]["data"]

                    for i in consumptions:



                        consumption = []



                        value = i["value"]

                        timestamp = i["timestamp"]

                        

                        trx_datetime = datetime.fromtimestamp(timestamp) + timedelta(hours=-8) 

                        consumption.append(location)

                        consumption.append(str(trx_datetime))

                        consumption.append(value)

                        

                        consumption_list.append(consumption)

                else:

                    print("Error code ",resp.status_code," returned for building ",location," ",year," ",half)

        print("")    

    return consumption_list
"""

#you can activate this block to retrieve data from the service

consumptions = retrieveConsumptions(locations)

leak = pd.DataFrame(consumptions, columns=['berkeley_id', 'timestamp', 'meter_reading'])

leak['timestamp'] = pd.to_datetime(leak['timestamp'])

leak = leak[leak.timestamp.dt.year >= 2016]

"""

leak = pd.read_csv("../input/uc-berkeley-consumptions/berkeley_consumptions.csv")

leak['timestamp'] = pd.to_datetime(leak['timestamp'])
leak.head()
# california daylight savings

idx = leak[(( leak['timestamp'] >= "2016-03-13 02:00:00") & (leak['timestamp'] <= "2016-11-06 02:00:00") )

    | ( (leak['timestamp'] >= "2017-03-12 02:00:00") & (leak['timestamp'] <= "2017-11-05 02:00:00") )

    | ( (leak['timestamp'] >= "2018-03-11 02:00:00") & (leak['timestamp'] <= "2018-11-05 02:00:00") ) ].index



#GMT-7 for summer days instead of GMT-8

leak.loc[idx, 'timestamp'] = leak.iloc[idx]['timestamp'] + timedelta(hours=1)
leak[leak['timestamp'] >= "2016-03-13"].head()
berkeley_buildings = list(leak.berkeley_id.unique())
correlation_threshold = 0.79 # we will expect correlation coefficient higher than this value

mae_correlation = 10 #mean absolute error of berkeley and ashrae data should have less than this value
lookup_list = [] #this will store ashrae id and berkeley id matching



leak_validation = leak[leak.timestamp.dt.year == 2016] #We can confirm Berkeley data with ASHRAE 2016 data



matched = 0; index = 0



for i in berkeley_buildings:

    for j in ashrae_buildings:

        

        df1 = leak_validation[(leak_validation.berkeley_id == i)]

        df2 = train_df[train_df.building_id == j]

        

        tmp = df1.merge(df2, on = ['timestamp'], how='left')

        tmp = tmp.dropna()

        correlation = tmp[['meter_reading_x', 'meter_reading_y']].corr(method ='pearson').values[0,1]

        

        mean = tmp.meter_reading_y.mean()

        mae = abs(tmp.meter_reading_x - tmp.meter_reading_y).sum()/tmp.shape[0]

        mae_over_mean = 100*mae/mean

        

        #print(i," ",j," (",correlation,")")

        

        if correlation >= correlation_threshold and mae_over_mean <= mae_correlation:

            matched = matched + 1

            print(matched,". berkeley ",i," is highly correlated to ashrae ",j," with score ",correlation)

            print("mae: ",mae," whereas mean: ",mean," mae / mean: ",100*mae/mean,"%")

            

            lookup_item = []

            lookup_item.append(i)

            lookup_item.append(j)

            lookup_item.append(mae)

            lookup_list.append(lookup_item)

            

            fig, ax = plt.subplots(figsize=(24, 3))

            plt.title("ASHRAE %s - BERKELEY %s (Correlation: %s)"%(j, i,round(correlation, 2)))

            

            if tmp.meter_reading_x.values[0:800].shape[0] > 0:

                berkeley_graph = tmp.meter_reading_x.values[0:800]

                ashrae_graph = tmp.meter_reading_y.values[0:800]

            else:

                berkeley_graph = tmp.meter_reading_x.values

                ashrae_graph = tmp.meter_reading_y.values

                

            plt.plot(berkeley_graph, label='berkeley')    

            plt.plot(ashrae_graph, label='ashrae')

            plt.legend()

            plt.show()

            

            print("------------------------------------")

            

        index = index + 1
building_lookup = pd.DataFrame(lookup_list, columns=['berkeley_id', 'ashrae_id', 'mae'])
building_lookup.head()
building_lookup[(building_lookup.ashrae_id == 630) | (building_lookup.ashrae_id == 598)]
building_lookup_best = building_lookup.groupby("ashrae_id", as_index=False)["mae"].min()
building_lookup = building_lookup.merge(building_lookup_best, on =["ashrae_id", "mae"], how="inner")
building_lookup.head()
print("There are ",building_lookup.shape[0]," buildings in Berkeley data set matched with ASHRAE data set")
leak = leak.merge(building_lookup, on=['berkeley_id'], how='left')
leak = leak[leak.ashrae_id > 0]
len(leak.ashrae_id.unique())
leak.head()
pd.DataFrame(leak.groupby("ashrae_id")["mae"].mean()).sort_values(by=['mae'])
site4 = leak.copy()
site4 = site4.drop(columns = ['berkeley_id'])

site4 = site4.rename(columns = {"ashrae_id": "building_id", "meter_reading": "meter_reading_scraped"})

site4['building_id'] = site4['building_id'].astype('int32')
site4 = site4[['building_id', 'timestamp', 'meter_reading_scraped']]
site4.sample(10)
site4.shape
print("There are ",site4[site4.timestamp.dt.year > 2016].shape," instances in test set")
site4.to_csv("site4.csv", index=False)