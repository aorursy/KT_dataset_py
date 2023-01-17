import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
df = pd.read_csv("../input/startup_funding.csv")
df.shape
del df["Remarks"]
df.tail(10)
df["CityLocation"].unique()
city = df.groupby(df["CityLocation"],as_index = False).count()
city.head()
city = city.sort_values(["SNo","CityLocation"],ascending = False)
city.head()
city_clear = pd.concat([city["CityLocation"], city["StartupName"]],join ="outer",axis = 1)
# city_clear.index = city_clear["CityLocation"]
# city_clear.pop("CityLocation")
city_clear = city_clear[0:10]
city_clear

plt.figure(figsize=(15,10))
g =sns.barplot(x=city_clear["CityLocation"], y=city_clear["StartupName"], data=city_clear)
g.set_xticklabels(city_clear["CityLocation"],rotation=45)
g.set_title("Top 10 Startup cities")
# g.set_ylabel('lololo')
df.head()
df["AmountInUSD"] = df["AmountInUSD"].apply(lambda x : float(str(x).replace(",","")))
df["AmountInUSD"] = pd.to_numeric(df["AmountInUSD"])
df.head()
fa = df[["CityLocation" ,"AmountInUSD"]].groupby(df["CityLocation"] , as_index = True).sum()
fa = fa.sort_values(["AmountInUSD"], ascending = False)
fa = fa[0:10]
fa
plt.figure(figsize=(15,10))
z = sns.barplot(x=fa.index , y=fa["AmountInUSD"])
z.set_xticklabels(fa.index,rotation = 45)
z.set_title("Top 10 Startup Funding cities")

df.head()
dr = df.dropna()
dr.head()
dr["AmountInUSD"].sort_values().max()
dr[dr.AmountInUSD == 1400000000.0 ]

fo = dr.sort_values("AmountInUSD",ascending =False )
fo.head()
plt.figure(figsize=(15,10))
b =sns.barplot(x=fo["StartupName"][0:10],y = fo["AmountInUSD"][0:10])
b.set_xticklabels(fo["StartupName"][0:10],rotation =45)
b.set_title("Top 10 Most Funding Companies ")
df.head(10)
df["Date"].dtypes
pa = df["AmountInUSD"] / df["AmountInUSD"].sum()
pa.head()
plt.figure(figsize=(15,10))
q = sns.boxplot(x = df["Date"], y = pa)
q.set_xticklabels(df["Date"], rotation = 45)
q.set_title("Investment 2015 -17")

yi = df[["Date","AmountInUSD"]].groupby(df["Date"],as_index= True).sum()
yi["Dates"] = yi.index 
yi.head()

yi["Dates"].dtypes
s = yi["Dates"]
yi["Dates"] = yi["Dates"].apply(lambda x : x[-4:])
yi.head()

yi = yi.sort_values(["Dates"] ,ascending =True)
yi.head()
y15 = yi["Dates"].str.contains("2015")
y16 = yi["Dates"].str.contains("2016")
y17 = yi["Dates"].str.contains("2017")
a15 = yi[y15].sort_values(["AmountInUSD"],ascending = False)
a16 = yi[y16].sort_values(["AmountInUSD"],ascending = False)
a17 = yi[y17].sort_values(["AmountInUSD"],ascending = False)
a17.head()
as15 = a15["AmountInUSD"].sum()
as16 = a16["AmountInUSD"].sum()
as17 = a17["AmountInUSD"].sum()
as15
year_as = ["2015","2016","2017"]
fund_as = [as15,as16,as17]
plt.figure(figsize=(15,10))
asf = sns.barplot(x = year_as , y = fund_as)
asf.set_xticklabels(year_as, rotation = 45)
asf.set_title("Year wise Investment")
df.head()
it = df.sort_values(["InvestmentType"],ascending = False)
it.head()
it = it[["InvestmentType","AmountInUSD"]]
it.head()
it["InvestmentType"] = it["InvestmentType"].str.replace("Crowd funding","Crowd Funding")
it["InvestmentType"] = it["InvestmentType"].str.replace("PrivateEquity","Private Equity")
it["InvestmentType"] = it["InvestmentType"].str.replace("SeedFunding","Seed Funding")

it = it.groupby(["InvestmentType"],as_index = False ).sum()
it
plt.figure(figsize=(15,10))
itg = sns.barplot(x = it["InvestmentType"] , y = it["AmountInUSD"])
itg.set_xticklabels(it["InvestmentType"], rotation = 45)
itg.set_title("Investment Type Funding Amounts")
