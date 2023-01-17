# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
global_temp=pd.read_csv("../input/GlobalTemperatures.csv")

global_temp=global_temp.set_index(global_temp.dt)

global_temp.index=global_temp.index.to_datetime()
yearly_temp=pd.groupby(global_temp,global_temp.index.year)

yearly_temp.LandAverageTemperature.mean().loc[1850:].plot(title="Yearly Average Land Temperature 1850 - 2015")
global_temp_country=pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv")

global_temp_country=global_temp_country.set_index(global_temp_country.dt)

global_temp_country.index=global_temp_country.index.to_datetime()
#The below function calculates the average increases in Average Land Temperature for a specific country 

#and returns the yearly increase rate
def get_country_temp_trend(country):

    country_temps=global_temp_country[global_temp_country.Country==country]

    country_temps=country_temps.ffill() 

    yearly_temps=pd.groupby(country_temps,by=country_temps.index.year).mean()

    model=sm.OLS(yearly_temps.AverageTemperature,yearly_temps.index)

    results=model.fit()

    return (results.params.get_value("x1"))
#We will run the above function against every unique country or region in the temperature dataset
unique_regions=global_temp_country.Country.unique()



temp_trend_list=list()

for region in unique_regions:

    temp_trend=get_country_temp_trend(region)

    temp_trend_list.append(temp_trend)
#Convert the temperature trend list into a pandas Series

temp_trend_series=pd.Series(temp_trend_list,index=unique_regions)

temp_trend_series=temp_trend_series.fillna(0)

temp_trend_series=temp_trend_series.sort_values()
#Lastly, we will plot the yearly average changes
f, ax = plt.subplots(figsize=(5, 50))

sns.barplot(x=temp_trend_series.values, y=temp_trend_series.index, 

            palette=sns.color_palette("coolwarm", len(temp_trend_series.values)), ax=ax)

texts = ax.set(ylabel="", xlabel="Temperature Change", title="Average Yearly Change in Land Temperature")