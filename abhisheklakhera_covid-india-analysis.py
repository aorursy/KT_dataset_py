# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
age_data=pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
age_data.head()
age_data.set_index("Sno")

age_data.head()
plt.figure(figsize=(10,8))
age_data=age_data.sort_values(by="TotalCases",ascending=False)
sns.barplot(x=age_data.AgeGroup,y=age_data.TotalCases)
plt.title("Cases by age-group")
covid_data=pd.read_csv("../input/covid19-in-india/covid_19_india.csv",parse_dates=["Date"])
covid_data.head()
plt.figure(figsize=(12,5))
plt.title("State wise Confirmed vs Cured cases")
covid_data.groupby('State/UnionTerritory')['Confirmed'].max().sort_values(ascending=False).plot.bar(color="skyblue",edgecolor="black")
covid_data.groupby('State/UnionTerritory')['Cured'].max().sort_values(ascending=False).plot.bar(color="lightgreen",edgecolor="black")


plt.figure(figsize=(12,5))
plt.title("State wise Confirmed vs Deceased cases")
covid_data.groupby('State/UnionTerritory')['Confirmed'].max().sort_values(ascending=False).plot.bar(color="skyblue",edgecolor="black")
covid_data.groupby('State/UnionTerritory')['Deaths'].max().sort_values(ascending=False).plot.bar(color="red",edgecolor="black")


#covid_data['cure rate']
confirmedCases=covid_data.groupby('State/UnionTerritory')['Confirmed'].max()
curedCases=covid_data.groupby('State/UnionTerritory')['Cured'].max()
CureRateData=pd.concat([confirmedCases,curedCases],axis=1)
CureRateData['Cure_Rate']=(CureRateData['Cured']/CureRateData['Confirmed'])*100
CureRateData['Cure_Rate']=CureRateData['Cure_Rate'].fillna(0)
CureRateDataGraph=CureRateData.drop(["Confirmed","Cured"],axis=1)

CureRateDataGraph.plot.bar(color="orange",figsize=(12,5))
plt.title("Performance of all states ")
#CureRateDataGraph
hospital_data= pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
hospital_data.shape
statewiseTesting_data= pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv",parse_dates=["Date"])
statewiseTesting_data.head()
statewiseTesting_data.groupby('State')["TotalSamples"].max().plot.bar(color="skyblue",edgecolor="black")
statewiseTesting_data.groupby('State')["Positive"].max().plot.bar(color="red",edgecolor="black")
plt.title("Total Samples v/s positive Cases")
ICMRTestingLabs_data= pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")
plt.figure(figsize=(8,5))
sns.countplot(x='type',  data=ICMRTestingLabs_data)
plt.title("Total types of testing labs")
plt.show()
