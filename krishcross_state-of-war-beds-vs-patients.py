import plotly.express as px
import pandas as pd

frame=pd.read_csv('../input/corona-complete/covid_19_clean_complete.csv')

df=pd.read_csv('../input/covid19inindia/HospitalBedsIndia.csv')
frame.columns
dead=0
actv=0
rec=0
conf=0
res=frame['Date']
for x in range(len(res)):
    if(frame['Date'][x]=='6/13/2020' or frame['Country/Region'][x]=='India'):
        dead=frame['Deaths'][x]
        actv=frame['Active'][x]
        rec=frame['Recovered'][x]
        conf=frame['Confirmed'][x]
import matplotlib.pyplot as plt
dead=(str)((dead/conf)*100)
actv=(str)((actv/conf)*100)
rec=(str)((rec/conf)*100)
names=['DEATHS'+"-"+dead[:4]+"%",'ACTIVE CASES'+"-"+actv[:5]+"%",'RECOVERED CASES'+"-"+rec[:5]+"%"]
my_circle=plt.Circle( (0,0), 0.7, color='pink')
size=[dead,actv,rec]
fig = plt.figure()
fig.patch.set_facecolor('pink')

plt.pie(size,  colors=['black','red','green'],radius=1,startangle=200)
p=plt.gcf()
patches, texts = plt.pie(size, colors=['black','red','green'], shadow=True, startangle=90)
plt.legend(patches, labels=names, loc="best")
p.gca().add_artist(my_circle)
plt.axis('equal')
plt.tight_layout()
plt.title("NUMBER OF COVID-19 PATIENTS IN INDIA AS ON 19th JUNE 2020")
plt.show()


total_beds={}
total_urban_beds={}
sum=0
for x in range(len(df['State/UT'])):
    total_beds[df['State/UT'][x]]=df['NumPublicBeds_HMIS'][x]+df['NumRuralBeds_NHP18'][x]+df['NumUrbanBeds_NHP18'][x]
    total_urban_beds[df['State/UT'][x]]=df['NumUrbanBeds_NHP18'][x]
print(total_beds)

    
import pandas as pd
df=pd.read_csv('../input/covid19inindia/HospitalBedsIndia.csv')
df.describe
# Getting the total number of bedsavailable in each state so as to find the threat number according
# complete corona world dataset analysis of USA,INDIA and ITALY corona tally from 22nd Jan 2020.
comp=pd.read_csv('../input/corona-complete/covid_19_clean_complete.csv')
comp.columns
import matplotlib.pyplot as plt
import numpy as np
dates=[]
cases=[]
for x in range(len(comp['Country/Region'])):
    if(comp['Country/Region'][x]=='India'):
        dates.append([comp['Date'][x]])
        cases.append(comp['Confirmed'][x])

dates=np.arange(len(dates))
f, ax = plt.subplots(figsize=(18,10))
plt.bar(dates,cases,width=0.8)
plt.xlabel('No. of days')

plt.ylabel('Total no. of Confirmed Cases in India')
plt.show()

dates=[]
cases=[]
for x in range(len(comp['Country/Region'])):
    if(comp['Country/Region'][x]=='US'):
        dates.append([comp['Date'][x]])
        cases.append(comp['Confirmed'][x])

dates=np.arange(len(dates))
f, ax = plt.subplots(figsize=(18,10))

plt.bar(dates,cases,color='Red',width=0.8)
plt.xlabel('No. of days')
plt.ylabel('Total no. of Confirmed Cases in US')
plt.show()
# Number of days between 22'Jan-19'June 2020
dates=[]
cases=[]
for x in range(len(comp['Country/Region'])):
    if(comp['Country/Region'][x]=='Italy'):
        dates.append([comp['Date'][x]])
        cases.append(comp['Confirmed'][x])

dates=np.arange(len(dates))
f, ax = plt.subplots(figsize=(18,10))

plt.bar(dates,cases,color='Green')
plt.xlabel('No. of days')
plt.ylabel('Total no. of Confirmed Cases in Italy')
plt.show()
# Number of days between 22'Jan-19'June 2020
# State Wise Analysis as on 19th June in India
state=pd.read_csv('../input/corona-latest-19june/19june.csv')
state.columns
states=[]
cases=[]
active=[]
for x in range(len(state['State'])):
    if(state['State'][x]=='Total'):
        continue
    states.append(state['State'][x])
    cases.append(state['Confirmed'][x])
    active.append(state['Active'][x])
    


fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(cases, labels=states, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True)
ax1.set_title("STATE WISE ANALYSIS")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
f, ax = plt.subplots(figsize=(18,10))

plt.bar(states,cases,color = list('rgbkymc'))
plt.xlabel("STATES")
plt.ylabel("CONFIRMED CASES")
plt.xticks(rotation=90)
plt.show()



threat_num=[]
state_list=[]
for x in range(len(states)):
    if(states[x]=='State Unassigned' or states[x]=='Ladakh' or states[x]=='Dadra and Nagar Haveli and Daman and Diu'):
        continue
    if(states[x]=='Jammu and Kashmir'):
        states[x]='Jammu & Kashmir'
    if(states[x]=='Andaman and Nicobar Islands'):
        states[x]='Andaman & Nicobar Islands'
    threat_num.append(-cases[x]+(total_urban_beds[states[x]]))
    state_list.append(states[x])
    if(total_urban_beds[states[x]]>=cases[x]):
        
        
        print(total_urban_beds[states[x]],cases[x],states[x],"ok")
    else:
        print(total_urban_beds[states[x]],cases[x],states[x],"red")

        
# Threat Number Visualization In different States
import folium

fg=folium.FeatureGroup("BED ANALYSIS")
fg.add_child(folium.GeoJson(data=(open('../input/folium-maps/webmap-using-folium-master/india_states.json','r',encoding='utf-8-sig').read())))
fg.add_child(folium.Marker(location=[19.7515, 75.7139],icon=folium.Icon(color='red'),radius=5,fill=False,popup="MAHARASHTRA"+" "+(str)(threat_num[0])))
fg.add_child(folium.Marker(location=[11.1271, 78.6569],icon=folium.Icon(color='red'),radius=5,fill=False,popup="TAMIL NADU"+" "+(str)(threat_num[1])))
fg.add_child(folium.Marker(location=[28.7041, 77.1025],icon=folium.Icon(color='red'),radius=5,fill=False,popup="DELHI"+" "+(str)(threat_num[2])))
fg.add_child(folium.Marker(location=[22.2587, 71.1924],icon=folium.Icon(color='red'),radius=5,fill=False,popup="GUJARAT"+" "+(str)(threat_num[3])))
fg.add_child(folium.Marker(location=[26.8467, 80.9462],icon=folium.Icon(color='green'),radius=5,fill=False,popup="UTTAR PRADESH"+" "+(str)(threat_num[4])))
fg.add_child(folium.Marker(location=[27.0238, 74.2179],icon=folium.Icon(color='red'),radius=5,fill=False,popup="RAJSATHAN"+" "+(str)(threat_num[5])))
fg.add_child(folium.Marker(location=[22.9734, 78.6569],icon=folium.Icon(color='green'),radius=5,fill=False,popup="MADHYA PRADESH"+" "+(str)(threat_num[6])))
fg.add_child(folium.Marker(location=[22.9868, 87.8550],icon=folium.Icon(color='green'),radius=5,fill=False,popup="BENGAL"+" "+(str)(threat_num[7])))
fg.add_child(folium.Marker(location=[15.3173, 75.7139],icon=folium.Icon(color='green'),radius=5,fill=False,popup="KARNATAKA"+" "+(str)(threat_num[8])))
fg.add_child(folium.Marker(location=[29.0588, 76.0856],icon=folium.Icon(color='red'),radius=5,fill=False,popup="HARYANA"+" "+(str)(threat_num[9])))
fg.add_child(folium.Marker(location=[25.0961, 85.3131],icon=folium.Icon(color='red'),radius=5,fill=False,popup="BIHAR"+" "+(str)(threat_num[10])))
fg.add_child(folium.Marker(location=[15.9129, 79.7400],icon=folium.Icon(color='green'),radius=5,fill=False,popup="ANDHRA PRADESH"+" "+(str)(threat_num[11])))
fg.add_child(folium.Marker(location=[33.7782, 76.5762],icon=folium.Icon(color='red'),radius=5,fill=False,popup="J & K"+" "+(str)(threat_num[12])))
fg.add_child(folium.Marker(location=[18.1124, 79.0193],icon=folium.Icon(color='green'),radius=5,fill=False,popup="TELANGANA"+" "+(str)(threat_num[13])))
fg.add_child(folium.Marker(location=[20.9517, 85.0985],icon=folium.Icon(color='green'),radius=5,fill=False,popup="ORISSA"+" "+(str)(threat_num[14])))
fg.add_child(folium.Marker(location=[26.2006, 92.9376],icon=folium.Icon(color='green'),radius=5,fill=False,popup="ASSAM"+" "+(str)(threat_num[15])))
fg.add_child(folium.Marker(location=[10.8505, 76.2711],icon=folium.Icon(color='green'),radius=5,fill=False,popup="KERELA"+" "+(str)(threat_num[16])))
fg.add_child(folium.Marker(location=[30.0668, 79.0193],icon=folium.Icon(color='green'),radius=5,fill=False,popup="U K"+" "+(str)(threat_num[17])))
fg.add_child(folium.Marker(location=[23.6102, 85.2799],icon=folium.Icon(color='green'),radius=5,fill=False,popup="JHARNKHAND"+" "+(str)(threat_num[18])))
fg.add_child(folium.Marker(location=[21.2787, 81.8661],icon=folium.Icon(color='green'),radius=5,fill=False,popup="CHATTISGARH"+" "+(str)(threat_num[19])))
fg.add_child(folium.Marker(location=[23.9408, 91.9882],icon=folium.Icon(color='green'),radius=5,fill=False,popup="TRIPURA"+" "+(str)(threat_num[20])))
fg.add_child(folium.Marker(location=[31.1048, 77.1734],icon=folium.Icon(color='green'),radius=5,fill=False,popup="HIMACHAL PRADESH"+" "+(str)(threat_num[21])))

fg.add_child(folium.Marker(location=[15.2993, 74.1240],icon=folium.Icon(color='green'),radius=5,fill=False,popup="GOA"+" "+(str)(threat_num[22])))
fg.add_child(folium.Marker(location=[24.6637, 93.9063],icon=folium.Icon(color='green'),radius=5,fill=False,popup="MANIPUR"+" "+(str)(threat_num[23])))
fg.add_child(folium.Marker(location=[30.7333, 76.7794],icon=folium.Icon(color='green'),radius=5,fill=False,popup="CHANDIGARH"+" "+(str)(threat_num[24])))
fg.add_child(folium.Marker(location=[11.9416, 79.8083],icon=folium.Icon(color='green'),radius=5,fill=False,popup="PONDICHERRY"+" "+(str)(threat_num[25])))
fg.add_child(folium.Marker(location=[26.1584, 94.5624],icon=folium.Icon(color='green'),radius=5,fill=False,popup="NAGALAND"+" "+(str)(threat_num[26])))
fg.add_child(folium.Marker(location=[23.1645, 92.9376],icon=folium.Icon(color='green'),radius=5,fill=False,popup="MIZORAM"+" "+(str)(threat_num[27])))
fg.add_child(folium.Marker(location=[15.9129, 79.7400],icon=folium.Icon(color='green'),radius=5,fill=False,popup="ARUNACHAL PRADESH"+" "+(str)(threat_num[28])))
fg.add_child(folium.Marker(location=[25.4670, 91.3662],icon=folium.Icon(color='green'),radius=5,fill=False,popup="MEGHALAYA"+" "+(str)(threat_num[29])))
fg.add_child(folium.Marker(location=[11.7401, 92.6586],icon=folium.Icon(color='green'),radius=5,fill=False,popup="ANDAMAN ISLANDS"+" "+(str)(threat_num[30])))
fg.add_child(folium.Marker(location=[27.5330, 88.5122],icon=folium.Icon(color='green'),radius=5,fill=False,popup="SIKKIM"+" "+(str)(threat_num[31])))
map=folium.Map(location=[21.1458,79.0882],tiles='CartoDB dark_matter',zoom_start=5)


map.add_child(fg)
display(map)

import numpy as np
def bar_color(df,color1,color2):
    return np.where(df.values>0,color1,color2).T


dft = pd.DataFrame (threat_num)
clr=[]
for x in threat_num:
    if(x<=0):
        clr.append('red')
    else:
        clr.append('green')
f, ax = plt.subplots(figsize=(18,10))

plt.bar(state_list,threat_num,color =clr)
plt.xlabel("STATES")
plt.ylabel("THREAT NUMBER")
plt.xticks(rotation=90)
plt.show()