import requests
results = []

base_url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%28AUTH%3A"friston"%29&format=json'

url = base_url

while True:

    r = requests.get(url=url)

    data = r.json()

    

    if len(data['resultList']['result']) == 0:

        break

        

    results += data['resultList']['result']

    print(data['nextCursorMark'])

    url = base_url + "&cursorMark=" + data['nextCursorMark']
len(results)
results[0]
import pandas as pd

df = pd.io.json.json_normalize(results).replace({"N": False, "Y": True}).set_index('id')

df.head()
df.dtypes
df.to_csv('frisons_papers_metadata.csv')