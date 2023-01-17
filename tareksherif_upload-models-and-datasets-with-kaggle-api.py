import os
import json


data = {}
data['people'] = []
data['people'].append({
    'name': 'Scott',
    'website': 'stackabuse.com',
    'from': 'Nebraska'
})
data['people'].append({
    'name': 'Larry',
    'website': 'google.com',
    'from': 'Michigan'
})
data['people'].append({
    'name': 'Tim',
    'website': 'apple.com',
    'from': 'Alabama'
})

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)
dataset_name="people"

API={"username":"tareksherif","key":"f4cf963ba526c529b3a9b0ea5058e6f0"}

os.environ['KAGGLE_USERNAME'] = API["username"]
os.environ['KAGGLE_KEY'] = API["key"]
data = {
  "title": dataset_name,
  "id": os.environ['KAGGLE_USERNAME']+"/"+dataset_name,
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
 
with open('dataset-metadata.json', 'w') as outfile:
    json.dump(data, outfile)
!kaggle datasets create -p .
