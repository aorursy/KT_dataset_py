# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import json



# Any results you write to the current directory are saved as output.
data = {

    "formatVersion" : 1,

    "passTypeIdentifier" : "pass.com.example.boarding-pass",

    "description" : "Example Boarding Pass",

    "teamIdentifier": "Example",

    "organizationName": "Example",

    "serialNumber" : "123456",

    "foregroundColor": "#866B23",

    "backgroundColor": "#FFD248",

    "boardingPass" : {

        "primaryFields" : [

            {

                "key" : "origin",

                "label" : "Atlanta",

                "value" : "ATL"

            },

            {

                "key" : "destination",

                "label" : "Johannesburg",

                "value" : "JNB"

            }

        ],

        "secondaryFields" : [

            {

                "key" : "boarding-gate",

                "label" : "Gate",

                "value" : "F12"

            }

        ],

        "auxiliaryFields" : [

            {

                "key" : "seat",

                "label" : "Seat",

                "value" : "7A"

            },

            {

                "key" : "passenger-name",

                "label" : "Passenger",

                "value" : "Honey Badger"

            }

        ],

        "transitType" : "PKTransitTypeAir",

        "barcode" : {

            "message" : "DL123",

            "format" : "PKBarcodeFormatQR",

            "messageEncoding" : "iso-8859-1"

        },

        "backFields" : [

            {

                "key" : "terms",

                "label" : "Terms and Conditions",

                "value" : "Valid for date of travel only"

            }

        ]

    }

}
#data below is a string in json format. We have to convert this to json object

data
#This will dump the data to json format

ser_data = json.dumps(data)
#This loads the json data so we can access the keys to get the values

json_data = json.loads(ser_data)
#We just need the fields in boarding pass

json_data['boardingPass']
boarding_pass = []

boarding_pass_fields = {}

#There are multiple primary fields. So we will loop them. We will fetch From and To from this section

primary_fields = json_data['boardingPass']['primaryFields']

for item in primary_fields:

    #Remember this is almost the same format we had in that data

    if item['key'] in ['origin', 'arrival', 'arr', 'From', 'Source']:

        boarding_pass_fields['origin_place'] = item['label']

        boarding_pass_fields['origin_code'] = item['value']

    elif item['key'] in ['destination', 'dest', 'To']:

        boarding_pass_fields['destination_place'] = item['label']

        boarding_pass_fields['destination_code'] = item['value']



#Auxilary fields. We will fetch passenger name from this section

aux_fields = json_data['boardingPass']['auxiliaryFields']

for item in aux_fields:

    if item['key'] in ['passenger-name']:

        boarding_pass_fields['Passenger'] = item['value']

boarding_pass.append(boarding_pass_fields)
#Likewise you can keep adding data and send from JSON. I will pass this to spacy to identify the entities of these

boarding_pass
# Load English tokenizer, tagger, parser, NER and word vectors

nlp = spacy.load('en_core_web_sm')
for item in boarding_pass:

    for key in list(item.keys()):

        if key in ['destination_place', 'origin_place', 'Passenger']:

            doc = nlp(item[key])

            for entity in doc.ents:

                print(entity.text, entity.label_)