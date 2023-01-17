import csv

import numpy as np

import pandas as pd

import os

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))

infile = '../input/all.data.combined.csv'
df = pd.read_csv(infile)

df.head()
# We'll start with this series id

seriesid = 'CEU0500000001'



# Series ids contain these bits of info:



# Survey types

survey_abbreviation = seriesid[:2]

# Seasonal codes

seasonal_code = seriesid[2]

# Industry codes

industry_code = seriesid [3:11]

# Data type codes

data_type_code = seriesid[11:]



print("\n".join(["seriesid:", seriesid]))

series = [survey_abbreviation, seasonal_code, industry_code, data_type_code]

for thing in series:

    print(thing)
# Random sample of the data

sampl = df.sample(n=20)

sampl = pd.DataFrame(sampl.series_id)

sampl.columns = ["series_id"]

sampl.head()
# Expand out the Series IDs

sampl["survey_abbrevations"] = sampl.series_id.str[:2]

sampl["seasonal_code"] = sampl.series_id.str[2]

sampl["industry_code"] = pd.to_numeric(sampl.series_id.str[3:11])

sampl["data_type_code"] = pd.to_numeric(sampl.series_id.str[11:])



sampl
# Setting up the mapping dictionary thing

file = '../input/ce.datatype.csv'

datatype = pd.read_csv(file)



# "zip" the two columns into a dictionary

datatypemapper = dict(zip(datatype.data_type_code, datatype.data_type_text))



# Replace the codes with the descriptions

sampl = sampl.replace({"data_type_code": datatypemapper})



# Rename the column

sampl = sampl.rename(columns = {"data_type_code": "data_type"})



sampl
file = '../input/ce.industry.csv'

industry = pd.read_csv(file)

industry.head()
# "zip" the two columns into a dictionary

industrymapper = dict(zip(industry.industry_code, industry.industry_name))



# Replace the codes with the descriptions

sampl = sampl.replace({"industry_code": industrymapper})



# Rename the column

sampl = sampl.rename(columns = {"industry_code": "industry_name"})



sampl