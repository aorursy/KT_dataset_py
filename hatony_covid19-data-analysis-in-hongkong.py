import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from pandas import DataFrame as df
import os
import folium
from datetime import datetime
from IPython.display import Image
import datetime as dt
import pylab as pl
import matplotlib.dates as mdates
from fbprophet import Prophet
import pystan
from fbprophet import Prophet
sns.set(rc={"figure.figsize": (18,10)})
Image("../input/imagess/CovidDataAnalysisinHK.jpg")
Crown_data=pd.read_csv('../input/hkdata/enhanced_sur_covid_19_eng0716.csv')
Crown_data29=pd.read_csv('../input/hkdata/enhanced_sur_covid_19_eng0729.csv')
impexp_data=pd.read_csv('../input/covid-19-hk/hongkong_covid_(importExp).csv')
Crown_data.info()
months = mdates.MonthLocator()
x=np.arange(7)
Crown_data29.rename(columns={'Report date':'ReportDate'},inplace=True)
Crown_data29["ReportDate"] = pd.to_datetime(Crown_data29["ReportDate"], format='%d/%m/%Y', errors='ignore')
Crown_data29["xDate"]=Crown_data29["ReportDate"]

Crown_data29["xDate"]= Crown_data29['ReportDate'].dt.strftime('%b')
Crown_data29['month'] = pd.DatetimeIndex(Crown_data29['ReportDate']).month

data_cofim=pd.get_dummies(Crown_data29['Confirmed/probable'],prefix='Confirmed/probable')
data_cofim.rename(columns={'Confirmed/probable_Confirmed':'Confirmed'},inplace=True)
data_cofim.rename(columns={'Confirmed/probable_Probable':'Probable'},inplace=True)

mon_date=Crown_data29[['month','xDate']]

merge_data=pd.concat([mon_date,data_cofim],axis=1)

resuldata=merge_data.groupby(['month']).sum().reset_index() 

resuldata.set_index('month')

#setting
plt.subplots_adjust(bottom=0.18)
pal = sns.color_palette("Greens_d",len(resuldata))
#
rank = resuldata["month"].argsort().argsort()
#text
g = sns.barplot(x="month",y="Confirmed",data=resuldata,palette=np.array(pal[::-1])[rank])
for index,row in resuldata.iterrows():
 g.text(row.name,row.Confirmed,round(row.Confirmed,2),color="black",ha="right")
#title
plt.xlabel("Month")
plt.ylabel("Confirmed Cases Counts")
plt.xticks(x,('Jan','Feb','Mar','Apr','May','Jun','Jul'))
plt.title("From 23Jan To 29Jul Convid19 Confirmed Cases in HongKong")
plt.show()
#import case count with local case compare
impexp_data.tail()
impexp_data.rename(columns={'Report date':'inoutDate'},inplace=True)
impexp_data.rename(columns={'Case classification*':'CaseStatus'},inplace=True)
impexp_data['inoutDate']=pd.to_datetime(impexp_data['inoutDate'],format='%d/%m/%Y',errors='ignore')
impexp_data['month']=pd.DatetimeIndex(impexp_data['inoutDate']).month
dategroup=impexp_data['month']
dategroup.reset_index()
case_data=pd.get_dummies(impexp_data['CaseStatus'],prefix='CaseStatus')
case_data.rename(columns={'CaseStatus_Epidemiologically linked with imported case':'Epidemiologically_With_Imported_Case'},inplace=True)
case_data.rename(columns={'CaseStatus_Epidemiologically linked with local case':'Epidemiologically_With_Local_Case'},inplace=True)
case_data.rename(columns={'CaseStatus_Epidemiologically linked with possibly local case':'Epidemiologically_Possibly_Local_Case'},inplace=True)
case_data.rename(columns={'CaseStatus_Imported case':'Imported_Case'},inplace=True)
case_data.rename(columns={'CaseStatus_Local case':'Local_Case'},inplace=True)
case_data.rename(columns={'CaseStatus_Possibly local case':'Possibly_Local_Case'},inplace=True)

case_data1=case_data[['Imported_Case','Local_Case']]
casegroup=pd.concat([dategroup,case_data1],axis=1)
case_data1.reset_index()
alldata=pd.concat([dategroup,case_data1],axis=1)
displaydata=alldata.groupby(alldata['month']).sum()
displaydata.plot()
plt.title("Imported Case & Local Case Counts Compare")
plt.xlabel("Month")
resuldata=pd.DataFrame(resuldata["Confirmed"])
resuldata.describe().round()

# Data sorce of Hospital just update to 16/Jul
guest_data=Crown_data[['Name of hospital admitted','Hospitalised/Discharged/Deceased']]

Hos_inout_data=pd.get_dummies(guest_data['Hospitalised/Discharged/Deceased'],prefix='Hospitalised/Discharged/Deceased')
Hos_inout_data.rename(columns={'Hospitalised/Discharged/Deceased_Discharged':'Discharged'},inplace=True)
Hos_inout_data.rename(columns={'Hospitalised/Discharged/Deceased_Hospitalised':'Hospitalised'},inplace=True)
Hos_inout_data.rename(columns={'Hospitalised/Discharged/Deceased_Deceased':'Deceased'},inplace=True)
Hos_inout_data.rename(columns={'Hospitalised/Discharged/Deceased_No admission':'No admission'},inplace=True)
Hos_inout_data.rename(columns={'Hospitalised/Discharged/Deceased_Pending admission':'Pending admission'},inplace=True)
guest_dataH=Crown_data[['Name of hospital admitted']]
guest_dataH.rename(columns={'Name of hospital admitted':'HospitalName'},inplace=True)
Inout_status=pd.concat([guest_dataH,Hos_inout_data],axis=1)
view_data=Inout_status.groupby(by='HospitalName').sum()
view_data.reset_index(inplace=True)
view_data_hos=view_data[['HospitalName','Hospitalised']]
sns.barplot(x='Hospitalised',y='HospitalName',data=view_data_hos)
view_data
view_data_dec=view_data[['HospitalName','Deceased']]
sns.barplot(x='Deceased',y='HospitalName',data=view_data_dec)
#by sex for comfired
sexdata=Crown_data29['Gender']
comparemf=pd.concat([sexdata,data_cofim],axis=1)
mf=comparemf.groupby(['Gender']).sum().reset_index()
mfrank = mf["Gender"].argsort().argsort()
mfg = sns.barplot(x="Gender",y="Confirmed",data=mf,palette=np.array(pal[::-1])[mfrank])
for index,row in mf.iterrows():
  mfg.text(row.name,row.Confirmed,round(row.Confirmed,2),color="blue",ha="right")
plt.xticks(np.arange(2),('Female','Male'),rotation=360)
plt.title("Male & Female Confirmed Cases Compare")
plt.show()
print("23Jan-29Jul Confirmed Cases Total:",int(mf['Confirmed'].sum()))
guest_age=Crown_data['Age']
colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']
cutdata=pd.concat([guest_age,Hos_inout_data],axis=1)
cutdata['splitdata']=pd.cut(cutdata['Age'],[0,20,40,60,80,100],
                            labels=['under20AgeGroup','20-40AgeGroup','40-60AgeGroup','60-80AgeGrpup','80-100AgeGroup'])
disdata=cutdata['Discharged'].groupby(cutdata['splitdata']).sum()
disdata.reset_index()
plt.subplot(2,2,1)
disdata.plot(kind='pie',title='Discharged Percentage',autopct='%.1f%%',colors=colors,explode=(0,0.1,0,0,0))

decdata=cutdata['Deceased'].groupby(cutdata['splitdata']).sum()
decdata.reset_index()
plt.subplot(2,2,2)
decdata.plot(kind='pie',title='Deceased Percentage',autopct='%1.1f%%',colors=colors,explode=(0,0,0,0.1,0))

hosdata=cutdata['Hospitalised'].groupby(cutdata['splitdata']).sum()
hosdata.reset_index()
plt.subplot(2,2,3)
hosdata.plot(kind='pie',title='Hospitalised Percentage',startangle=70,colors=colors,autopct='%1.1f%%',explode=(0,0,0.1,0,0))

cutdata2=pd.concat([guest_age,data_cofim],axis=1)
cutdata2['splitdata2']=pd.cut(cutdata2['Age'],[0,20,40,60,80,100],labels=['under20AgeGroup','20-40AgeGroup','40-60AgeGroup','60-80AgeGrpup','80-100AgeGroup'])
comdata=cutdata2['Confirmed'].groupby(cutdata2['splitdata2']).sum()
comdata.reset_index()
plt.subplot(2,2,4)
comdata.plot(kind='pie',title='Confirmed Percentage',autopct='%1.1f%%',colors=colors,explode=(0,0.1,0,0,0))
plt.show()
## Age with hospitalised & Deceased heamap
Crown_Age=Crown_data[['Age']]
view_data_Thr=Hos_inout_data[['Hospitalised','Discharged','Deceased']]
groupAge=pd.concat([Crown_Age,view_data_Thr],axis=1)
sns.pairplot(groupAge,kind="reg",vars=["Age","Hospitalised","Discharged","Deceased"])
#sns.pairplot(groupAge,kind="reg",hue="Discharged",vars=["Age","Hospitalised","Discharged","Deceased"])
agedata=view_data.corr()
sns.heatmap(agedata,annot=True)
# 18 districts in which more than 2 probable/confirmed cases have resided in the past 14 days
build_data=pd.read_csv('../input/hkdata/building_list_eng29.csv')
build_data.rename(columns={'Related probable/confirmed cases':'Confirmed_Case_Counts'},inplace=True)
build_data=build_data[['District','Confirmed_Case_Counts']]
info_case=build_data['Confirmed_Case_Counts'].str.split(',',expand=True)
info_case=info_case.stack()
info_case=info_case.reset_index(level=1,drop=True).rename('Case_Counts')
info_new=build_data.drop(['Confirmed_Case_Counts'],axis=1).join(info_case)
District=info_new.groupby("District")
Disdata=District.size()
Disdata
#with 2 or more probable/confirmed cases have resided in the past 14 days(25/07/2020)
longitude = 114.1693611
latitude = 22.3193039

tooltip ='Please click for informations'
case_map = folium.Map(location=[latitude, longitude],zoom_start=11)
case_map.add_child(folium.Marker(location=[22.286667,114.155],tooltip=tooltip,popup='Central&Western CaseCounts 44',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.284167,114.224167],tooltip=tooltip,popup='Eastern CaseCounts 82',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.261111,113.946111],tooltip=tooltip,popup='Islands CaseCounts 25',icon=folium.Icon(color='blue'))) 
case_map.add_child(folium.Marker(location=[22.328333,114.181667],tooltip=tooltip,popup='Kowloon City CaseCounts 159',icon=folium.Icon(color='pink'))) 
case_map.add_child(folium.Marker(location=[22.355,114.083889],tooltip=tooltip,popup='Kwai Tsing CaseCounts 121',icon=folium.Icon(color='blue'))) 
case_map.add_child(folium.Marker(location=[22.313333,114.225833],tooltip=tooltip,popup='Kwun Tong CaseCounts 241',icon=folium.Icon(color='pink')))
case_map.add_child(folium.Marker(location=[22.494722,114.138056],tooltip=tooltip,popup='North CaseCounts 101',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.381389,114.270556],tooltip=tooltip,popup='Sai Kung CaseCounts 116',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.385556,114.193056],tooltip=tooltip,popup='Sha Tin CaseCounts 163',icon=folium.Icon(color='pink')))
case_map.add_child(folium.Marker(location=[22.330833,114.162222],tooltip=tooltip,popup='Sham Shui Po CaseCounts 168',icon=folium.Icon(color='pink')))
case_map.add_child(folium.Marker(location=[22.247222,114.158889],tooltip=tooltip,popup='Southern CaseCounts 29',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.441178,114.164772],tooltip=tooltip,popup='Tai Po CaseCounts 48',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.374444,114.115],tooltip=tooltip,popup='Tsuen Wan CaseCounts 59',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.390264,113.976255],tooltip=tooltip,popup='Tuen Mun CaseCounts 234',icon=folium.Icon(color='pink')))
case_map.add_child(folium.Marker(location=[22.390264,113.976255],tooltip=tooltip,popup='Wan Chai CaseCounts 88',icon=folium.Icon(color='blue')))
case_map.add_child(folium.Marker(location=[22.333611,114.196944],tooltip=tooltip,popup='Wong Tai Sin  CaseCounts 538',icon=folium.Icon(color='red')))
case_map.add_child(folium.CircleMarker(location=[22.333611,114.196944],tooltip=tooltip,popup='Wong Tai Sin  CaseCounts 538',max_width=1000,min_width=1000,radius=25,fill=True,fill_color='crimson',icon=folium.Icon(color='red')))
case_map.add_child(folium.Marker(location=[22.321389,114.1725],tooltip=tooltip,popup='Yau Tsim Mong CaseCounts 256',icon=folium.Icon(color='red')))
case_map.add_child(folium.CircleMarker(location=[22.321389,114.1725],tooltip=tooltip,popup='Yau Tsim Mong CaseCounts 256',radius=25,fill=True,fill_color='crimson',icon=folium.Icon(color='red')))
case_map.add_child(folium.Marker(location=[22.445556,114.022222],tooltip=tooltip,popup='Yuen Long CaseCounts 82',icon=folium.Icon(color='blue')))

case_map
#future
futrue_data=pd.read_csv('../input/hkdata/enhanced_sur_covid_19_eng0729.csv')
futrue_data=futrue_data[['Report date','Confirmed/probable']]
spilt_data=pd.get_dummies(futrue_data['Confirmed/probable'],prefix='Confirmed/probable')
spilt_data.rename(columns={'Confirmed/probable_Confirmed':'Confirmed'},inplace=True)
spilt_data=spilt_data[['Confirmed']]

Comtraining_data=pd.concat([futrue_data,spilt_data],axis=1)
Comtraining_data.rename(columns={'Report date':'ds','Confirmed':'y'},inplace=True)
Comtraining_data=Comtraining_data[['ds','y']]
Comtraining_data['ds']=Comtraining_data['ds'].apply(pd.to_datetime)
Comtraining_data.reset_index()
plt.xticks(rotation=90)
record=Comtraining_data['y'].groupby(Comtraining_data['ds']).value_counts()
record.plot()
#Comtraining_data
prophet = Prophet()
prophet.fit(Comtraining_data)
future = prophet.make_future_dataframe(periods=180)
future.tail()
forecast = prophet.predict(future)
forecast[['ds','trend','yhat', 'yhat_lower','yhat_upper']].tail()
fig1 = prophet.plot(forecast)
print(fig1)
fig2 = prophet.plot_components(forecast)
print(fig2)