import numpy as np 

import pandas as pd



Location = "../input/API_ILO_country_YU.csv"

data = pd.read_csv(Location)

columns = data.columns[2:]

indexes = [x for x in columns]

max_values = []

max_info = []



for column in columns:

    max_values.append(data[column].max())

    max_info.append(data.loc[data[column].idxmax()]["Country Name"])

    min_values.append(data[column].min())

    min_info.append(data.loc[data[column].idxmin()]["Country Name"])



max_unemployment_rate = {}



max_unemployment_rate["Max Unemployment Rate"] = max_values

max_unemployment_rate["Country With Max Unemployment Rate"] = max_info
data = pd.DataFrame(max_unemployment_rate, index = indexes)
import numpy as np 

import pandas as pd



Location = "../input/API_ILO_country_YU.csv"

data = pd.read_csv(Location)



columns = data.columns[2:]

indexes = [x for x in columns]

max_values = []

max_info = []



for column in columns:

    max_values.append(data[column].max())

    max_info.append(data.loc[data[column].idxmax()]["Country Name"])



max_unemployment_rate = {}



max_unemployment_rate["Max Unemployment Rate"] = max_values

max_unemployment_rate["Country With Max Unemployment Rate"] = max_info



data = pd.DataFrame(max_unemployment_rate, index = indexes)

print(data)
columns = data.columns[2:]

indexes = [x for x in columns]

min_values = []

min_info = []



for column in columns:

    min_values.append(data[column].min())

    min_info.append(data.loc[data[column].idxmin()]["Country Name"])



min_unemployment_rate = {}



min_unemployment_rate["Min Unemployment Rate"] = min_values

min_unemployment_rate["Country With Min Unemployment Rate"] = min_info

data = pd.DataFrame(min_unemployment_rate, index = indexes)
import numpy as np 

import pandas as pd



Location = "../input/API_ILO_country_YU.csv"

data = pd.read_csv(Location)



columns = data.columns[2:]

indexes = [x for x in columns]

min_values = []

min_info = []



for column in columns:

    min_values.append(data[column].min())

    min_info.append(data.loc[data[column].idxmin()]["Country Name"])



min_unemployment_rate = {}



min_unemployment_rate["Min Unemployment Rate"] = min_values

min_unemployment_rate["Country With Min Unemployment Rate"] = min_info

data = pd.DataFrame(min_unemployment_rate, index = indexes)

print(data)