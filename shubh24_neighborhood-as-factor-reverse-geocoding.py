import pandas as pd

import csv

#import reverse_geocoder as rg



train_file =  "../input/train.json"
train_df = pd.read_json(train_file)



train_coords = train_df[["listing_id", "latitude", "longitude"]]
lat_lon = []

listings = []



for i, j in train_coords.iterrows():

    lat_lon.append((j["latitude"], j["longitude"]))

    listings.append(int(j["listing_id"]))
#results = rg.search(lat_lon) #Uncomment this. This is the juice!

results = [] #Comment this :(



nbd = [[listings[i], results[i]['name']] for i in range(0, len(results))] #getting ready to write to csv 
with open("neighborhood_train.csv", "wb") as f:



    writer = csv.writer(f, delimiter = ",")

    writer.writerows(nbd)