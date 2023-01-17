import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# files that we can work with; we will focus on countries now

# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

df = pd.DataFrame(data)

print(df.head())
# list of countries

countries = df['Country'].unique()

print(countries)
temperature = df['AverageTemperature'].groupby(df['Country'])