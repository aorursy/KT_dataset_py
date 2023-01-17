url1 = "https://pomber.github.io/covid19/timeseries.json"

url2 = "https://coronanepal.live/"
import requests

import json

from bs4 import BeautifulSoup

import pandas as pd

from matplotlib import pyplot as plt

data = requests.get(url1)

json_data = json.loads(json.dumps(data.json()))

countries = json_data.keys()

print(countries)



# df = pd.DataFrame(columns=["country","date","confirmed","deaths","recovered"])

# for country in list(countries):

#     for i in json_data[country]:

#         df = df.append({"country":country,"date":i["date"],"confirmed":i["confirmed"],"deaths":i["deaths"],"recovered":i["recovered"]},ignore_index=True)



# print(len(df))

# df.head(2)


country = "Nepal"

df = pd.DataFrame(columns=["country","date","confirmed","deaths","recovered"])

for i in json_data[country]:

    df = df.append({"country":country,"date":i["date"],"confirmed":i["confirmed"],"deaths":i["deaths"],"recovered":i["recovered"]},ignore_index=True)



df = df.drop(["country"], axis = 1)



df.head(2)
df.plot()

plt.show()
df['date']= pd.to_datetime(df['date']) 



months_df = df.set_index('date').groupby(pd.Grouper(freq='M')).sum()

months_df.plot()

plt.show()
months_df = df.set_index('date').groupby(pd.Grouper(freq='W')).sum()

months_df.plot(kind="bar")

plt.show()