# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import plotly.express as px

import plotly.graph_objects as go



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the csv file as a DataFrame called calls

calls = pd.read_csv("/kaggle/input/montcoalert/911.csv")
calls.head()
#Checking what the data consists of

calls.info() 
calls.rename(columns={"lat":"Latitude","lng":"Longitude","desc":"Description","zip":"Zipcode","title":"Title","timeStamp":"Time","twp":"Township","addr":"Address"},inplace= True)


calls["Zipcode"].value_counts().head(10)
calls["Township"].value_counts().head()
#Number of unique emergencies

calls["Title"].nunique()

#Reason for calling

reason = calls["Title"].apply(lambda x: x.split(":")[0])

reason.value_counts()
calls["Reason"] = reason

plt.figure(figsize=(10,5))

sns.countplot(x="Reason", data = calls, hue="Reason")

from pandas.api.types import is_string_dtype

import datetime as dt


is_string_dtype(calls["Time"])
#Convert the Time collumn from a str to DataTime object

calls["Time"] = pd.to_datetime(calls["Time"])
calls["Hour"] = calls["Time"].apply(lambda x:x.time())

calls["Date"] = calls["Time"].apply(lambda x:x.date())

calls["Day of Week"]= calls["Time"].dt.dayofweek



calls
# Map the date to its correspondent day of the week

day_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}

calls["Week Day"]=calls["Day of Week"].map(day_map)



calls.drop(columns=["Day of Week"], inplace = True)
# Reason for calling for each day of the week

plt.figure(figsize=(20,10))



sns.countplot(x="Week Day", data= calls, hue= "Reason")

plt.tight_layout()
# Reason for calling for each month

month= calls["Time"].apply(lambda x: x.month)

plt.figure(figsize=(20,10))



sns.countplot(x=month, data= calls , hue= "Reason")
calls.groupby("Date").count()["Township"].plot(figsize= (20,10))

plt.tight_layout()
#For each reason:

#Traffic

calls[calls["Reason"]=="Traffic"].groupby("Date").count()["Township"].plot(figsize= (20,10))

plt.title("Traffic",fontsize=30)
#EMS

calls[calls["Reason"]=="EMS"].groupby("Date").count()["Township"].plot(figsize=(20,10))

plt.title("EMS",fontsize=30)
#Fire

calls[calls["Reason"]=="Fire"].groupby("Date").count()["Township"].plot(figsize=(20,10))

plt.title("Fire",fontsize=30)
calls["The Hour"] = calls["Hour"].apply(lambda x:x.hour)

calls
#Grouping the data by Day and Hour

dayHour = calls.groupby(by=["Week Day","The Hour"]).count()["Reason"].unstack()

dayHour.head()
#Creating Heatmaps for dayHour

plt.figure(figsize= (20,10))

sns.heatmap(dayHour,cmap="coolwarm")
sns.clustermap(dayHour,cmap="coolwarm", figsize= (20,10))