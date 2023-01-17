# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import altair as alt # For displaying graphs

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import requests

for dirname, _, filenames in os.walk('/kaggle/input/icu-beds-by-county-in-the-us/'):

    for filename in filenames:

        file = os.path.join(dirname, filename)

        print(file)



# Any results you write to the current directory are saved as output.
demoGraphicAndICUByCounty = pd.read_csv(file)

print("Summary of ICU Beds per State and County with Residents (60+) per ICU Bed")

demoGraphicAndICUByCounty = demoGraphicAndICUByCounty.rename(columns = {"ICU Beds":"ICU_Beds","Total Population":"Total_Population","Population Aged 60+":"Population_Aged_60+","Percent of Population Aged 60+":"Percent_Population_Aged_60","Residents Aged 60+ Per Each ICU Bed":"Residents_Per_ICU_Bed"})

demoGraphicAndICUByCounty
jhuDailyDataSet=pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')

print("Summary of JHU Daily Dataset across the US (Starting Jan 01 2020)")

jhuDailyDataSet
countWithNoICUBeds = demoGraphicAndICUByCounty.query("ICU_Beds==0")

countWithNoICUBeds

countiesWithICUBeds = demoGraphicAndICUByCounty.query("ICU_Beds>0")
print("Most Vulnerable Counties having No ICU Beds")

alt.Chart(countWithNoICUBeds).mark_circle(size=50).encode(

    x='Total_Population',

    y='Population_Aged_60+',

    color='State',

    tooltip=['State', 'County','Percent_Population_Aged_60','Population_Aged_60+','Total_Population']

).interactive()
# Going to concatenate Vulnerable Counties with No ICU Beds and Day wise COVID-19 Cases and Deaths

# Have to aggregate jhuDataSet by Total Cases and Total Deaths



jhuDailyDataSet = jhuDailyDataSet.rename(columns={'state':'State','county':'County'})

jhuAggregateDataSet = jhuDailyDataSet.groupby(['State','County'])

jhuAggregateDataSetAggregate = jhuAggregateDataSet.agg(np.sum)

jhuAggregateDataSetAggregate

mergedDataSet = pd.merge(jhuAggregateDataSetAggregate,countiesWithICUBeds,left_on='State',right_on='State')

alt.data_transformers.disable_max_rows()

print("Vulnerable State and Counties with maximum residents per ICU and higher Cases of COVID19")

alt.Chart(mergedDataSet).mark_circle(size=50).encode(

    x='cases',

    y='Residents_Per_ICU_Bed',

    color='State',

    tooltip=['State', 'County','Percent_Population_Aged_60','Population_Aged_60+','Total_Population','cases','deaths','Residents_Per_ICU_Bed']

).interactive()