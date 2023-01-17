from tqdm.notebook import tqdm

import requests

from requests.packages.urllib3.exceptions import InsecureRequestWarning

import json





requests.packages.urllib3.disable_warnings(InsecureRequestWarning)



url = "https://bdtd.ibict.br/vufind/api/v1/search"
stats = requests.get(url + "?lookfor=*%3A*&type=AllFields&facet[]=format&facet[]=institution&limit=0", verify=False).json()



print(f"Documentos: {stats['resultCount']}")

for facet in stats["facets"]["format"]:

    print(f"{facet['translated']}: {facet['count']}")
def get_json_data(metadata, n=1000, fp=10):

    count = metadata["count"]

    records = []

    

    for i in tqdm(range(0, count//n)):

        current_url = url + metadata["href"].replace("limit=0", f"limit={n}") + f"&page={i+1}"

        req = requests.get(current_url, verify=False).json()

        records += req["records"]

        if i % fp == 0:

            print(current_url)

    if (count//n)*n < count:

        i += 1

        print("Last Index:", i+1)

        current_url = url + metadata["href"].replace("limit=0", f"limit={n}") + f"&page={i+1}"

        print("Last URL:", current_url)

        req = requests.get(current_url, verify=False).json()

        records += req["records"]

        

    return records
facets = stats["facets"]["format"]

facets
# Dissertações:

metadata = facets[0]

count = metadata["count"]

thesis_metadata = get_json_data(metadata, fp=20)

print(count, len(thesis_metadata))

json_object = json.dumps(thesis_metadata, indent = 4) 



with open("/kaggle/working/bdtd_master_thesis.json", "w") as f:

    f.write(json_object) 
# Teses:

metadata = facets[1]

count = metadata["count"]

thesis_metadata = get_json_data(metadata, fp=10)

print(count, len(thesis_metadata))

json_object = json.dumps(thesis_metadata, indent = 4) 



with open("/kaggle/working/bdtd_doctoral_thesis.json", "w") as f:

    f.write(json_object) 
from datetime import datetime



print("Extração dos dados terminada em:", datetime.now())