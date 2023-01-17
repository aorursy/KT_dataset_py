import sys

import pymongo

from pprint import pprint as pp

from bson.objectid import ObjectId

from PIL import Image

import io
# restore the train.bson file into a mongoDB database named 'cdiscount' and collection named 'images'

# using default/standard ports for mongoDB, use pymongo to retrieve images selectively.

from pymongo import MongoClient

import datetime

#client = MongoClient()

#client = MongoClient('mongodb://localhost:27017/')

client = MongoClient('localhost', 27017)

#db = client.cdiscount

db = client['cdiscount']

#collection = db.images

imagescollection = db['images']

imagescollection.count()

print("collections count -->" + str(imagescollection.count()))

db.collection_names(include_system_collections=False)

cursor = imagescollection.find({"category_id":1000010683})

print("Cursor count -->" + str(cursor.count()))

for rec in cursor:

    #print(rec['imgs'])

    for item in rec['imgs']:

        #print(item['picture'])

        f = io.BytesIO(item['picture'])

        img = Image.open(f)

        print (img.size)

        img.show()



        client.close()