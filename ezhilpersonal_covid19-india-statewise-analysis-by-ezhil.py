#Created by Ezhilarasan Kannaiyan

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#covid = pd.read_csv("G:\Python\Session6_26-Apr-2020\Exercise\covid19-corona-virus-india-dataset\complete.csv")

covid = pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")

#covid.head()
covid.rename(columns={"Date":"Reported_Dt","Name of State / UT":"State_UT", "Total Confirmed cases (Indian National)":"Confirmed_Indian","Total Confirmed cases ( Foreign National )": "Confirmed_Forigners","Total Confirmed cases" : "Confirmed_Total","Cured/Discharged/Migrated" : "Cure_Discharge_Migrated" }, inplace=True)

#covid.head()
state_group = covid.groupby(covid.State_UT)

st_agg = state_group.aggregate({"Confirmed_Total": "sum"}).sort_values(by="Confirmed_Total", ascending=False)

#st_agg.iloc[:5,:].plot()

#plt.show()

x=list(st_agg.iloc[:5,:].index)

y = list(st_agg.iloc[:5,0])

#plt.xkcd()

#plt.style.use("ggplot")

#print(plt.style.available)

#mpl.rcParams.update(mpl.rcParamsDefault)

plt.plot(x,y,'ro-', linewidth=3,label='Statewise Confirmed Cases (Top 5)')

plt.legend()

plt.xlabel('State Name')

plt.ylabel('Total Conrfirmed Cases (till 27-Apr-2020)')

plt.grid(True)

plt.show()

#To Show top 5 States affected by Corona Virus
state_group = covid.groupby(covid.State_UT)

st_agg = state_group.aggregate({"Death": "sum"}).sort_values(by="Death", ascending=False)

st_names=list(st_agg.iloc[:5,:].index)

cured_count = list(st_agg.iloc[:5,0])

explode_capital = [0,0,0,0.1,0] # to show out capital exclusively 

plt.pie(cured_count,labels=st_names,explode=explode_capital, startangle=90, autopct='%1.1f%%', wedgeprops={'edgecolor':'black'})

plt.title('Statewise Cured Pie Chart')

plt.show()

# Conclusion : Capital Delhi is in second place in confirmed cases. But in fourth place in Death percentage
covid.columns

date_group = covid.groupby(covid.Reported_Dt)



dt_agg = date_group.aggregate({"Confirmed_Total": "sum","Cure_Discharge_Migrated":"sum","Death":"sum"}).sort_values(by="Reported_Dt", ascending=False)

x=list(dt_agg.iloc[:10,:].index)

Con = list(dt_agg.iloc[:10,0])

Cure = list(dt_agg.iloc[:10,1])

Died = list(dt_agg.iloc[:10,2])

x.reverse()

Con.reverse()

Cure.reverse()

Died.reverse()

plt.stackplot(x,Con,Cure,Died, colors=['blue','green','red'], labels=['Confirmed','Cured','Died'])



plt.legend(loc='upper left')

plt.xlabel('Date')

plt.ylabel('Count (till 27-Apr-2020)')

plt.xticks(rotation='vertical')

plt.show()

#Confirmed/Cured/Death cases
date_group = covid.groupby(covid.Reported_Dt)

dt_agg = date_group.aggregate({"Death": "sum"}).sort_values(by="Reported_Dt", ascending=False)

x=list(dt_agg.iloc[:30,:].index)

y = list(dt_agg.iloc[:30,0])

x.reverse()

y.reverse()

plt.figure(figsize=(15,10))

plt.bar(x,y,label='Datewise Death Cases')

plt.legend()

plt.xlabel('Date', fontsize=22)

plt.ylabel('Death Cases (till 27-Apr-2020)', fontsize=22)

plt.xticks(rotation='vertical', fontsize=18)

plt.show()

#Death rate is increasing daily (in the last 1 month)