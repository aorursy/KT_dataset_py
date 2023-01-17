import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

rain=pd.read_csv('/kaggle/input/rain-fall-1/chennai_reservoir_rainfall.csv')

res=pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv')
rain.head()
rain.dtypes
res.dtypes
res.head()
#df=rain.join(res,lsuffix="_Rain",rsuffix="_Res")
df=rain.merge(res,on='Date',how='inner',suffixes=("_Rain","_Res"))
df.head()
# updating rain data

# convert date to datetime object

rain.Date=pd.to_datetime(rain.Date,format="%d-%m-%Y")

#update the index with datetime

rain.index=rain.Date

#drop the Date column as it is unnecessary

rain=rain.drop(['Date'],axis=1)
rain.head()
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(12,5))

plt.plot(rain.POONDI,color='#2403ff')

plt.title("Rain Fall At POONDI")

plt.xlabel("Date")

plt.ylabel("Rainfail in CM")
plt.figure(figsize=(12,5))

plt.plot(rain.POONDI,color='#2403ff',label="Poondi")

plt.plot(rain.CHOLAVARAM,color='r',label="Chola")

plt.plot(rain.REDHILLS,color='g',label="RedHills")

plt.plot(rain.CHEMBARAMBAKKAM,color='y',label="Chembar")

plt.legend()

plt.title("Rain Fall At POONDI")

plt.xlabel("Date")

plt.ylabel("Rainfail in CM")
rain['Total']=(rain.POONDI+rain.CHOLAVARAM+rain.REDHILLS+rain.CHEMBARAMBAKKAM)/4
rain.head(1)
plt.figure(figsize=(12,5))

plt.plot(rain.Total,color='#2403ff',label="Total Rain")

plt.legend()

plt.title("Avg Rain Fall")

plt.xlabel("Date")

plt.ylabel("Rainfail in CM")
rain["Date"]=rain.index
rain.Date[0].month
rain['year']=rain.Date.map(lambda x: x.year)

rain['month']=rain.Date.map(lambda x: x.month)
rain.head()
yearly_rainfall_total=rain.groupby(['year'])["Total"].sum()
plt.figure(figsize=(10,4.5))

plt.plot(yearly_rainfall_total)

plt.title("Yearly Total Rain")
plt.figure(figsize=(10,4))

plt.bar(yearly_rainfall_total.index,yearly_rainfall_total)

plt.xlabel('Year')

plt.ylabel("Total Rain in CM")

plt.title("Rainfall Per year")
calan={1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",  7:"July",

      8:"Aug",   9:"Sep",10:"Oct", 11:"Nov", 12: "Dec"}

rain["month"]=rain.month.map(lambda x: calan[x])

monthy_avg_rain=rain.groupby(["month"])["Total"].mean()
plt.figure(figsize=(10,4))

plt.bar(calan.values(),monthy_avg_rain.loc[calan.values()])

plt.title("Monthly average rainfall")
res.Date=pd.to_datetime(res.Date)

res.index=res.Date
res["month"]=res.Date.map(lambda x: x.month)

res["year"]=res.Date.map(lambda x: x.year)
res["total"]=(res['POONDI']+res['CHOLAVARAM']\

            +res['REDHILLS']+res['CHEMBARAMBAKKAM'])/4
avg_water=res.groupby(["year"])['total'].mean()
plt.figure(figsize=(10,3))

plt.plot(avg_water,label="Avg Water Level")

plt.plot(yearly_rainfall_total,label="Total Rain Fall")

plt.legend()
plt.scatter(avg_water,yearly_rainfall_total)
plt.scatter(avg_water,yearly_rainfall_total.shift(1))

plt.xlabel('avg water level')

plt.ylabel("yearly rain fall with 1 year lag")
from scipy.stats import pearsonr,spearmanr

print("pearson test",pearsonr(avg_water.iloc[1:],yearly_rainfall_total.shift(1).iloc[1:]))

print("Spearman test",spearmanr(avg_water.iloc[1:],yearly_rainfall_total.shift(1).iloc[1:]))
plt.figure(figsize=(10,3))

plt.plot(avg_water,label="Avg Water Level")

plt.plot(yearly_rainfall_total.shift(1),label="Total Rain Fall")

plt.legend()
d=pd.DataFrame(data={"Rainfall":yearly_rainfall_total,\

                      "Avg_Water":avg_water})

d.to_csv("chennai_water.csv")
plt.figure(figsize=(10,4))

plt.bar(avg_water.index,avg_water)

plt.title("Average Water in chennai resorvoir")

plt.xlabel("Avg Water in Feet")

plt.ylabel("Year")
plt.hist(rain.Total,bins=100)
plt.hist(yearly_rainfall_total[:-1],bins=4)
yearly_rainfall_total[:-1].describe()