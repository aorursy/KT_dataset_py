import hdbscan
from matplotlib import pylab
import pymongo as pm
import numpy as np
import pandas as pd
import json
import csv
from geopy import distance
from descartes import PolygonPatch
connection = pm.MongoClient()
nyc = connection.nyc
taxi_collection = nyc.taxi_data
taxi_collection.drop()
taxi_collection.create_index([("pickup_location",  pm.GEO2D)])
taxi_collection.create_index([("dropoff_location",  pm.GEO2D)])
def import_content(filename):
    print('Reading file...')
    data = open(filename, 'r')
    print('File read...')
    header = True
    keys = []
    step = 0
    batch_data = []
    for line in data:
        line_data = {}
        if header:
            keys = line.split(',')
            print(keys)
            header = False
        else:
            line = line.split(',')
            for k in range(len(keys)):
                line_data[keys[k]] = line[k]
            try:
                line_data['pickup_location'] = line_data['pickup_longitude'] + ',' + line_data['pickup_latitude'] 
                line_data['dropoff_location'] = line_data['dropoff_longitude'] + ',' + line_data['dropoff_latitude']
            except:
                pass # sometimes there are is no data in the pickup or dropoff locs
            if step % 100000 == 0:
                batch_data = load_many_and_report(taxi_collection, batch_data, step)
            else:
                batch_data += [line_data]
        step += 1
    batch_data = load_many_and_report(taxi_collection, batch_data, step)
    print('Finished loading data.')
def load_many_and_report(coll, batch_data, step):
    coll.insert_many(batch_data)
    print('%d lines loaded...' % step)
    batch_data = []
    return batch_data
# import_content('data/train.csv')
import_content('../input/train.csv')
'''
Knowing that you're receiving location points as (long, lat) and you need to swap them.
'''
def getLength(loc1, loc2):
    return distance.distance((loc1[1], loc1[0]), (loc2[1], loc2[0])).miles
cursor = taxi_collection.find()
i = 1
while cursor.alive:
    i += 1
    if i % 100000 == 0:
        print('%d distances calcuated...' % i)
    taxi = cursor.next()
    D = getLength(taxi['pickup_location'].split(','), taxi['dropoff_location'].split(','))
    taxi_collection.update_one( {"_id":taxi['_id']}, {"$set": { "l2_distance": D }} )