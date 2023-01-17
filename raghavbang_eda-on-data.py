
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("../input/covid-19-world-dataset/time_series_2019-ncov-Confirmed.csv")
database=data.copy()
aa=[]
bb=[]
for i in range(4,65):
    m=database.iloc[:,i].value_counts()
    if(m[0]>341):
          aa.append(i)
data.drop(data.columns[aa],axis=1,inplace=True)
data=data.rename(columns={"Province/State":"State","Country/Region":"Country"})
data
data.info()
data.describe()
data.isnull().sum()

plt.figure(figsize=(6,14))
plt.barh(data["Country"],data["3/18/20"])
plt.yticks(fontsize=7.5)
plt.ylabel("Country")
plt.xlabel("Cases")
plt.title("Corona cases on 3/18/20")
plt.show()
plt.title("Cases in Japan")
plt.xlabel("Cases")
plt.ylabel("Date")
col=list(set(data.columns)-set(['State', 'Country', 'Lat', 'Long']))
plt.plot(data.iloc[1,4:23],col)
asia = pd.read_csv("../input/covid-19-world-dataset/as_countries.csv")
eu = pd.read_csv("../input/covid-19-world-dataset/eu_countries.csv")
af = pd.read_csv("../input/covid-19-world-dataset/af_countries.csv")
na = pd.read_csv("../input/covid-19-world-dataset/na_countries.csv")
sa = pd.read_csv("../input/covid-19-world-dataset/sa_countries.csv")
as_count = asia['name'].values.tolist()
eu_count = eu['name'].values.tolist()
af_count = af['name'].values.tolist()
na_count = na['name'].values.tolist()
sa_count = sa['name'].values.tolist()
cases = data["3/18/20"]
countries = data["Country"]

ind = 0
as_cases = 0
eu_cases = 0
af_cases = 0
na_cases = 0
sa_cases = 0
for i in countries:
    if i in as_count:
        as_cases+=cases[ind]
    if i in eu_count:
        eu_cases+=cases[ind]
    if i in af_count:
        af_cases+=cases[ind]
    if i in na_count:
        na_cases+=cases[ind]
    if i in sa_count:
        sa_cases+=cases[ind]
    ind=ind+1
x=[as_cases,eu_cases,af_cases,na_cases,sa_cases]
y=["Asia","Europe","Africa","NA","SA"]
plt.bar(y,x,color="RED")
