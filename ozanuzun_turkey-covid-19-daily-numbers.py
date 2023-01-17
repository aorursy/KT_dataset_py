import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

covid_turkey=pd.read_csv("../input/covid19-in-turkey/covid_19_data_tr.csv")
covid_turkey
new_values={"New_cases":[],"New_Deaths":[],"New_Recovered":[]}
last_case=0
last_death=0
last_recorered=0

aa=covid_turkey[["Confirmed","Deaths","Recovered"]]

for i,j,k in zip(covid_turkey["Confirmed"],covid_turkey["Deaths"],covid_turkey["Recovered"]):
    new_values["New_cases"].append(i-last_case)
    new_values["New_Deaths"].append(j-last_death)
    new_values["New_Recovered"].append(k-last_recorered)
    last_case=i
    last_death=j
    last_recorered=k


covid_turkey.insert(3,"New_cases",new_values["New_cases"])
covid_turkey.insert(5,"New_deaths",new_values["New_Deaths"])
covid_turkey.insert(7,"New_Recovered",new_values["New_Recovered"])
covid_turkey.head(55)
dflen=len(covid_turkey)-1
index_values=np.arange(0,dflen,int(dflen/10))
          
cases_index=np.append(index_values,[covid_turkey["New_cases"].idxmax(),dflen])
cases_index=np.sort(cases_index)

cases_dates=covid_turkey["Last_Update"][cases_index]
cases_values=np.append(np.arange(0,covid_turkey["New_cases"].max(),1000),covid_turkey["New_cases"].max())

deaht_index=np.append(index_values,[covid_turkey["New_deaths"].idxmax(),dflen])
deaht_index=np.sort(deaht_index)

deaht_dates=covid_turkey["Last_Update"][deaht_index]
deaht_values=np.append(np.arange(0,covid_turkey["New_deaths"].max(),30),covid_turkey["New_deaths"].max())

recovered_index=np.append(index_values,[covid_turkey["New_Recovered"].idxmax(),dflen])
recovered_index=np.sort(recovered_index)

recovered_dates=covid_turkey["Last_Update"][recovered_index]
recovered_values=np.append(np.arange(0,covid_turkey["New_Recovered"].max(),1000),covid_turkey["New_Recovered"].max())



fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=[25,30])
ax1.plot(covid_turkey["Last_Update"],covid_turkey["New_cases"])
covid_turkey["Last_Update"][0]
ax1.set_xticks(cases_dates)
ax1.set_yticks(cases_values)
ax1.set_title("Türkiye Günlük Vaka Sayısı")
ax1.set_xlabel("Tarih")
ax1.set_ylabel("Vaka Miktarı")
ax1.legend(["Günlük vaka miktarı"])
for label in ax1.get_xticklabels():
    label.set_rotation(65)
ax1.grid(True)

ax2.plot(covid_turkey["Last_Update"],covid_turkey["New_deaths"])
ax2.set_xticks(deaht_dates)
ax2.set_yticks(deaht_values)
ax2.set_title("Türkiye Günlük Ölüm Sayısı")
ax2.set_xlabel("Tarih")
ax2.set_ylabel("Ölüm miktarı")
ax2.legend(["Günlük Ölüm Miktarı"])
for label in ax2.get_xticklabels():
    label.set_rotation(65)
ax2.grid(True)

ax3.plot(covid_turkey["Last_Update"],covid_turkey["New_Recovered"])
ax3.set_xticks(recovered_dates)
ax3.set_yticks(recovered_values)
ax3.set_xlabel("Tarih")
ax3.set_ylabel("İyileşme sayıları")
ax3.set_title("Türkiye Günlük İyileşme Sayısı")
ax3.legend(["Günlük İyileşme Sayıları"])
for label in ax3.get_xticklabels():
    label.set_rotation(65)

ax3.grid(True);
between_dates=covid_turkey["Last_Update"][np.append(index_values,dflen)]
x=covid_turkey["Last_Update"]
y1=covid_turkey["New_cases"]
y2=covid_turkey["New_Recovered"]
fig,ax=plt.subplots(figsize=[25,9])
ax.plot(x,y1,"red")
ax.plot(x,y2,"green")
ax.set_xticks(between_dates)
ax.legend(["Günlük Vaka Sayısı","Günlük İyileşme Sayısı"])
ax.fill_between(x,y1,y2,where=(y1>y2),color="red",alpha=0.15)
ax.fill_between(x,y1,y2,where=(y2>y1),color="green",alpha=0.15)
ax.set_title("Türkiye Günlük Vaka Ve İyileşme Sayılarının Karşılaştırılması")
ax.set_xlabel("Tarih")
ax.set_ylabel("Mevcut Günlük Vaka Ve İyileşme Sayıları")
ax.grid(True);
