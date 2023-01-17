# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
df = covid_19_data

df_conf = time_series_covid_19_confirmed

df_death = time_series_covid_19_deaths

df_recovered = time_series_covid_19_recovered
from matplotlib import pyplot as plt

from plotly.offline import iplot, init_notebook_mode

#import plotly.plotly as py

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
df_conf = df_conf.groupby("Country/Region").sum()

df_conf.drop(columns=["Lat", "Long"], inplace=True)
top_country = df_conf.sort_values(by = df_conf.columns[-1], ascending = False).head(3)

top_country.transpose().iplot(title = 'Time series of confirmed cases of top 3 countries')
class period:

    def __init__(self, country):

        self.country = country

        self.day =[]

        self.value = []

    

    def when (self):

        selected = top_country.loc[self.country]

        for a,b in selected.items():

            if b>=500:

                self.day.append(a)

                self.value.append(b)

        country_dict=dict(zip(self.day, self.value))



        country_df = pd.DataFrame.from_dict(country_dict, orient='index')

        return country_df
Italy_df = period("Italy")

US_df = period("US")

China_df = period("China")
Italy = Italy_df.when()

US = US_df.when()

China = China_df.when()
#to put three countries on the same graph, they need to have the same length. So I append nan to the ones 

#that have shorter length 



Italy_to_graph = np.append(Italy, np.repeat(np.nan, len(China)-len(Italy)))

US_to_graph = np.append(US, np.repeat(np.nan, len(China)-len(US)))

China_to_graph = np.append(China, np.repeat(np.nan, len(China)-len(China)))

graph_x = [x for x in range(1, len(China)+1, 1)]



#combine them in one dataframe

only3_df = pd.DataFrame(index = graph_x,data=US_to_graph, columns=["US"])

only3_df["Italy"] = Italy_to_graph

only3_df["China"] = China_to_graph



only3_df
only3_df.iplot(xTitle = "Days", yTitle = "Confirmed Cases", 

                 title = """\n Assuming every country's "day 1" is the fist day when it had more than 500 cases""")
only3_zoom = only3_df[:10]

only3_zoom.iplot(vline=[2, 8], title="When did countries put on strict social distancing measures?")

#Problem: I sitll don't know how to make an annotatio/tag on a specific point of a specific line without the change of the graph

from scipy.optimize import curve_fit

import math
def logistic(x, L, k, x0):

    return L / (1 + np.exp(-k * (x-x0)))
China = only3_df["China"]

p_China = [85000, 0.3, 20]

popt_C, pcov_C = curve_fit(logistic,range(len(China)), China, p0=p_China)

print("In the case of China")

print("Predicted L (the maximum number of confirmed cases): " + str(int(popt_C[0])))

print("Predicted k (growth rate): " + str(float(popt_C[1])))

print("Predicted x0 (the day of the inflection): " + str(int(popt_C[2])))
US = only3_df["US"].dropna()

p_US = [140000, 0.35, 70]

popt_US, pcov_US = curve_fit(logistic,range(len(US)), US, p0=p_US)

print("In the case of US")

print("Predicted L (the maximum number of confirmed cases): " + str(int(popt_US[0])))

print("Predicted k (growth rate): " + str(float(popt_US[1])))

print("Predicted x0 (the day of the inflection): " + str(int(popt_US[2])))
Italy = only3_df["Italy"].dropna()

p_Italy = [120000, 0.30, 45]         

popt_I, pcov_I = curve_fit(logistic,range(len(Italy)), Italy, p0=p_Italy)

print("In the case of Italy")

print("Predicted L (the maximum number of confirmed cases): " + str(int(popt_I[0])))

print("Predicted k (growth rate): " + str(float(popt_I[1])))

print("Predicted x0 (the day of the inflection): " + str(int(popt_I[2])))