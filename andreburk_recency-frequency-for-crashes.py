# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt',\
                   infer_datetime_format = True, parse_dates = ["Date"])
data.info()
data.head()
# Top10 cases by "Route"
data["Route"].value_counts().nlargest(10).plot(kind = "barh").invert_yaxis()
# Top10 airplane type
data["Type"].value_counts().nlargest(10).plot(kind = "barh").invert_yaxis()
# Top10 Location
data["Location"].value_counts().nlargest(10).plot(kind = "barh").invert_yaxis()
# fatalities over time
plt.figure()
data.set_index(data["Date"])["Fatalities"].plot()
# create date for recency time difference
today = datetime.datetime(2016, 7, 16)

# get recency
data["Recency"] = data["Date"].apply(lambda x: (today - x).days)
# aggregation per airline for RFF
airline = data.groupby("Operator").agg({"Recency": "min",
                                        "Date": "count",
                                        "Fatalities": lambda x: x.sum(skipna = True)})
airline["Fatalities"] = airline["Fatalities"].astype(np.int32)
airline.rename(columns = {"Date": "Frequency", "Fatalities": "TotalFatal"}, inplace = True)
airline["FatPerIncident"] = round(airline["TotalFatal"] / airline["Frequency"], 0).astype(np.int32)
airline.head()
sns.barplot(y = "Operator", x = "Recency", data = airline.nsmallest(15, columns = "Recency").reset_index())
plt.xlabel("Days since last incident")
airline.nsmallest(15, columns = "Recency").reset_index()
sns.barplot(y = "Operator", x = "Frequency", data = airline.nlargest(15, columns = "Frequency").reset_index())
plt.xlabel("total # of incidents")
airline.nlargest(15, columns = "Frequency").reset_index()
sns.barplot(y = "Operator", x = "TotalFatal", data = airline.nlargest(15, columns = "TotalFatal").reset_index())
plt.xlabel("Total # of fatalities")
airline.nlargest(15, columns = "TotalFatal").reset_index()
sns.barplot(y = "Operator", x = "FatPerIncident", data = airline.nlargest(15, columns = "FatPerIncident").reset_index())
plt.xlabel("Average # of fatalities per incident")
airline.nlargest(15, columns = "FatPerIncident").reset_index()
# checking for airlines with more than 1 total incidents
airline.loc[airline["Frequency"] > 1, :].nlargest(15, columns = "FatPerIncident").reset_index()
# Top20 airplane types for "Aeroflot" incidents
aero20 = data.loc[data["Operator"] == "Aeroflot", "Type"]
aero20 = aero20.loc[aero20.isin(aero20.value_counts().nlargest(20).index)]
sns.countplot(y = aero20)
# Top20 location for "Aeroflot" incidents
aero20l = data.loc[data["Operator"] == "Aeroflot", "Location"]
aero20l = aero20l.loc[aero20l.isin(aero20l.value_counts().nlargest(20).index)]
sns.countplot(y = aero20l)
# Aeroflot incidents over time
plt.figure(figsize = (17, 6))
sns.countplot(x = data.loc[data["Operator"] == "Aeroflot", "Date"].apply(lambda x: x.year))
plt.title("Aeroflot incidents over time")
# Top20 airplane types for "Military - U.S. Air Force" incidents
us20 = data.loc[data["Operator"] == "Military - U.S. Air Force", "Type"]
us20 = us20.loc[us20.isin(us20.value_counts().nlargest(20).index)]
sns.countplot(y = us20)
# Top20 location for "Military - U.S. Air Force" incidents
us20l = data.loc[data["Operator"] == "Military - U.S. Air Force", "Location"]
us20l = us20l.loc[us20l.isin(us20l.value_counts().nlargest(20).index)]
sns.countplot(y = us20l)
# Military - U.S. Air Force incidents over time
plt.figure(figsize = (17, 6))
sns.countplot(x = data.loc[data["Operator"] == "Military - U.S. Air Force", "Date"].apply(lambda x: x.year))
plt.title("Military - U.S. Air Force incidents over time")