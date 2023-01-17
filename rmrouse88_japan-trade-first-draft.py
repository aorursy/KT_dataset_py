import numpy as np 



import pandas as pd



from pandas import Series,DataFrame
import matplotlib as mpl



import matplotlib.pyplot as plt



import seaborn as sns
%matplotlib inline
countries = pd.read_csv("../input/country_eng.csv")
countries.head()
c = countries["Country"]
n = countries["Country_name"]
cdict = {c:n for c,n in zip(c,n)}
dframe = pd.read_csv("../input/exp_custom_latest_ym.csv")
dframe.head()
hs2 = pd.read_csv("../input/hs2_eng.csv")

hs2.head()
hs2dict = {k:v for k,v in zip(hs2["hs2"],hs2["hs2_name"])}
hs4 = pd.read_csv("../input/hs4_eng.csv")

hs4.head()
hs4dict = {k:v for k,v in zip(hs4["hs4"],hs4["hs4_name"])}
hs6 = pd.read_csv("../input/hs6_eng.csv")

hs6.head()
hs6dict = {k:v for k,v in zip(hs6["hs6"],hs6["hs6_name"])}
hs9 = pd.read_csv("../input/hs9_eng.csv")

hs9.head()
hs9dict = {k:v for k,v in zip(hs9["hs9"],hs9["hs9_name"])}
year_latest = pd.read_csv("../input/year_latest.csv")
year_latest.head()
year_latest.groupby("exp_imp").size()
year_latest["hs2"]=year_latest["hs2"].map(hs2dict)
year_latest["hs4"]=year_latest["hs4"].map(hs4dict)
year_latest["hs6"]=year_latest["hs6"].map(hs6dict)
year_latest["hs9"] = year_latest["hs9"].map(hs9dict)
year_latest["Country"]=year_latest["Country"].map(cdict)
year_latest.groupby("exp_imp")["VY"].agg(sum)
year_latest.head()
area = {k:v for k,v in zip(countries["Country_name"],countries["Area"])}
year_latest["Area"] = year_latest["Country"].map(area)
year_latest.head()
grouped = year_latest.groupby(["Year","Area"])["VY"].agg(sum).to_frame().unstack()

grouped.head()
grouped["VY"].plot(figsize=(10,10))
grouped_export = year_latest[year_latest.exp_imp==1].groupby(["Year","Area"])["VY"].agg(sum).unstack()

grouped_export.head()
grouped_import = year_latest[year_latest.exp_imp==2].groupby(["Year","Area"])["VY"].agg(sum).unstack()

grouped_import.head()
fig,ax = plt.subplots(2,figsize=(10,10),sharex=True)



grouped_export.plot(ax=ax[0])

ax[0].set_title("Exports")

ax[0].set_ylabel("Volume")



grouped_import.plot(ax=ax[1])

ax[1].set_title("Imports")

ax[1].set_xlabel("Year")

ax[1].set_ylabel("Volume")
totals = pd.pivot_table(year_latest,"VY","Year","exp_imp",aggfunc=sum).rename(columns = {1:"Exports",2:"Imports"})

totals.head()
fig,ax = plt.subplots(figsize=(10,10))



totals.plot(kind="bar", ax=ax)

ax.set_ylabel("Value")

ax.set_title("Japanese Import/Export Value Totals by Year")
totals = pd.pivot_table(year_latest,"VY","Year",["exp_imp","Area"],aggfunc=sum)

totals.head()
exports = totals[1]

imports = totals[2]
fig3,ax3 = plt.subplots(2,figsize=(10,20))

exports.plot(kind="bar",stacked=True,ax=ax3[0])

ax3[0].set_title("Exports", fontsize=20)

ax3[0].set_ylabel("Value", fontsize=15)



imports.plot(kind="bar",stacked=True,ax=ax3[1])

ax3[1].set_title("Imports", fontsize=20)

ax3[1].set_ylabel("Value", fontsize =15)

ax3[1].set_xlabel(ax3[1].get_xlabel(),fontsize=15)



yticks0 = plt.setp(ax3[0].get_yticklabels(),fontsize=10)

yticks1 = plt.setp(ax3[1].get_yticklabels(),fontsize=10)

xticks = plt.setp(ax3[1].get_xticklabels(),fontsize=10)
fig4,ax4 = plt.subplots(figsize=(10,10))



year_latest[(year_latest.exp_imp==2) & (year_latest.Area == "Asia")].groupby(["Country"])["VY"].agg(sum).sort_values(ascending=True).plot(kind="barh",ax=ax4)



ax4.set_ylabel(ax4.get_ylabel(),fontsize=15)

ax4.set_xlabel("Total Import Value",fontsize=15)





yticks = plt.setp(ax4.get_yticklabels(),fontsize=10)

xticks = plt.setp(ax4.get_xticklabels(),fontsize=10)
region_dive = year_latest[year_latest.Area=="Asia"].groupby(["Year","Country"])["VY"].agg(sum).to_frame()
region_dive.replace(np.nan, 0, inplace=True)
region_dive.head()
fig5,ax5 = plt.subplots(figsize=(10,10))



region_dive.unstack()["VY"].plot(kind = "bar",stacked = True,ax=ax5)

ax5.set_ylabel("Value", fontsize=15)

ax5.set_xlabel(ax5.get_xlabel(),fontsize=15)

ax5.set_title("Asian Region Trade Value By Year",fontsize=20)







yticks = plt.setp(ax5.get_yticklabels(),fontsize=10)

xticks = plt.setp(ax5.get_xticklabels(),fontsize=10)