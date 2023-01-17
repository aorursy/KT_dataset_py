from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
#checking donations top five rows
donations.head()

#checking schools top five rows
schools.head()
#checking donors top five rows
donors.head()
#checking teachers top five rows
teachers.head()
#checking projects top five rows
projects.head()
#checking resources top five rows
resources.head()
#checking for missing data in donation
missTot=donations.isnull().sum().sort_values(ascending=False)

missPer=(donations.isnull().sum()/donations.isnull().count()*100).sort_values(ascending=False)

missing_don_data=pd.concat([missTot,missPer],axis=1,keys=['Total','Percentage'])

#donation missing data
missing_don_data.head()
#checking for missing data in donors
missTot=donors.isnull().sum().sort_values(ascending=True)
missPer=(donors.isnull().sum()/donors.isnull().count()*100).sort_values(ascending=False)
missing_don_data=pd.concat([missTot,missPer],axis=1,keys=['Total','Percentage'])
#donors missing data
missing_don_data.head()
#checking for missing data in projects
missTot=projects.isnull().sum().sort_values(ascending=False)

missPer=(projects.isnull().sum()/projects.isnull().count()*100).sort_values(ascending=False)

missing_don_data=pd.concat([missTot,missPer],axis=1,keys=['Total','Percentage'])

#projects missing data
missing_don_data.head()
#checking for missing data in resources
missTot=resources.isnull().sum().sort_values(ascending=False)

missPer=(resources.isnull().sum()/resources.isnull().count()*100).sort_values(ascending=False)

missing_don_data=pd.concat([missTot,missPer],axis=1,keys=['Total','Percentage'])

#resources missing data
missing_don_data.head()
#checking for missing data in schools
missTot=schools.isnull().sum().sort_values(ascending=False)

missPer=(schools.isnull().sum()/schools.isnull().count()*100).sort_values(ascending=False)

missing_don_data=pd.concat([missTot,missPer],axis=1,keys=['Total','Percentage'])

#schools missing data
missing_don_data.head()
#checking for missing data in teachers
missTot=teachers.isnull().sum().sort_values(ascending=False)

missPer=(teachers.isnull().sum()/teachers.isnull().count()*100).sort_values(ascending=False)

missing_don_data=pd.concat([missTot,missPer],axis=1,keys=['Total','Percentage'])

#teachers missing data
missing_don_data.head()
#dividing donation received date in donations into year, month etc and looking at donation
#data based on year, month etc
donations['Donation Date']=pd.to_datetime(donations['Donation Received Date'])
donations['Donation Year']=donations['Donation Date'].dt.year
donations['Donation Month']=donations['Donation Date'].dt.month
donations['Donation Day']=donations['Donation Date'].dt.day
donations['Donation Hour']=donations['Donation Date'].dt.hour
donations['Donation Weekday']=donations['Donation Date'].dt.weekday

donY=donations.groupby('Donation Year').agg({'Donation Month':'count','Donation Amount':'sum'}).reset_index().rename(columns={'Donation Month':'Total Donation','Donation Amount':'Total Amount'})
x = donY['Donation Year']
y1 = donY['Total Donation']
y2 = donY['Total Amount']
trace1=go.Scatter(x=x[:-1],y=y1[:-1],fillcolor='#fcc45f',mode='none')
trace2=go.Scatter(x=x[:-1],y=y2[:-1],fillcolor='#e993f9',mode='none')
fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Total Donation per year','Total donation amount per year'])
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)

fig['layout'].update(height=300,yaxis=dict(autorange=True),yaxis2=dict(autorange=True))

iplot(fig)


#Donations per weekday
wkmp={0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'}
donations['Donation Weekday']=donations['Donation Weekday'].map(wkmp)
#print(donations['Donation Weekday'])
donY=donations.groupby('Donation Weekday').agg({'Donation Day':'count','Donation Amount':'sum'}).reset_index().rename(columns={'Donation Day':'Total Donation','Donation Amount':'Total Amount'})
trace1=go.Scatter(x=donY['Donation Weekday'],y=donY['Total Donation'],fillcolor='r')
trace2=go.Scatter(x=donY['Donation Weekday'],y=donY['Total Amount'],fillcolor='g')
#data=[trace1, trace2]
fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Total Donation per weekday','Total donation amount per weekday'])
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
#data=[trace1, trace2]
fig['layout'].update(height=300,yaxis=dict(autorange=True),yaxis2=dict(autorange=True),title='Donation Count and Total Weekday')
#fig=go.Figure(data=data,layout=layout)
iplot(fig)
#Donations per month
mnmp={1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
donations['Donation Month']=donations['Donation Month'].map(mnmp)
donY=donations.groupby('Donation Month').agg({'Donation Day':'count','Donation Amount':'sum'}).reset_index().rename(columns={'Donation Day':'Total Donation','Donation Amount':'Total Amount'})
trace1=go.Scatter(x=donY['Donation Month'][:-1],y=donY['Total Donation'][:-1],fillcolor='r')
trace2=go.Scatter(x=donY['Donation Month'][:-1],y=donY['Total Amount'][:-1],fillcolor='g')
fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Total Donation per month','Total donation amount per month'])
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)

fig['layout'].update(height=300,yaxis=dict(autorange=True),yaxis2=dict(autorange=True))
iplot(fig)
#Donation amount Included Optional to donorschoose
#1.number of yes/no for 15% donation to dc
donNum=donations['Donation Included Optional Donation'].value_counts()
print(donNum)
#2.total amount of 15% donation to dc by those who said Yes
dcDon=donations.groupby('Donation Included Optional Donation').agg({'Donation Amount':'sum'}).reset_index()
per2dc=dcDon[dcDon['Donation Included Optional Donation']=='Yes']['Donation Amount']*15/100
print(float(per2dc))
#total amount of 15% donation to dc by those who said Yes
#dcamount=donations[donations['Donation Included Optional']=='Y]['Donation Amount'].sum()*15/100
#donors and donations merge
dondona=donations.merge(donors,on='Donor ID',how='inner')
#statewisedonor=dondona.groupby('Donor State').agg({'Donor ID':'count'})
top15statedonors=dondona['Donor State'].value_counts().head(15)
plt.figure(figsize=(20,4))
top15statedonors.plot(kind='bar')
plt.xlabel('Donor States')
plt.ylabel('Count')#,xTitle='States',yTitle = "Donors Count", title = 'Top Donor states')
plt.title('Top Donor States')

#Statewise donation amount
swdonationamt=dondona.groupby('Donor State').agg({'Donation Amount':'sum'}).reset_index().sort_values(by='Donor State',ascending=False)
plt.figure(figsize=(15,4))
trace1=plt.barh(swdonationamt['Donor State'][:5],swdonationamt['Donation Amount'][:5],color='r')

plt.xlabel('Donation Amount ($)', fontsize=12)
plt.ylabel('Donor States')
plt.title("Distribution of Donation Amount statewise")
plt.show(trace1)
#top 15 donor cities
city=dondona['Donor City'].value_counts().head(15)
print(city)
plt.figure(figsize=(20,4))
plt.bar(city.index,height=city.values)
plt.xlabel('Donor Cities')
plt.ylabel('Count')
plt.title('Top donor cities')
plt.show()

#top 15 donor state
state=dondona['Donor Is Teacher']
sns.countplot(state)
#top donor cart sequence
cart=dondona['Donor Cart Sequence'].value_counts().head()
carts=pd.DataFrame()
carts['labels']=cart.index
carts['values']=cart.values
plt.figure(figsize=(6,6))
plt.pie(cart.values,labels=cart.index,autopct='%1.1f%%', shadow=True)
plt.title('Donor Cart Sequence')
plt.show()


#merging schools and projects together

schprojects=projects.merge(schools,how='inner',on='School ID')
#states with highest donor to population ratio
censusact2013 = {'Mississippi': 2991207, 'Iowa': 3090416, 'Oklahoma': 3850568, 'Delaware': 925749, 'Minnesota': 5420380, 'Alaska': 735132, 'Illinois': 12882135, 'Arkansas': 2959373, 'New Mexico': 2085287, 'Indiana': 6570902, 'Maryland': 5928814, 'Louisiana': 4625470, 'Texas': 26448193, 'Wyoming': 582658, 'Arizona': 6626624, 'Wisconsin': 5742713, 'Michigan': 9895622, 'Kansas': 2893957, 'Utah': 2900872, 'Virginia': 8260405, 'Oregon': 3930065, 'Connecticut': 3596080, 'New York': 19651127, 'California': 38332521, 'Massachusetts': 6692824, 'West Virginia': 1854304, 'South Carolina': 4774839, 'New Hampshire': 1323459, 'Vermont': 626630, 'Georgia': 9992167, 'North Dakota': 723393, 'Pennsylvania': 12773801, 'Florida': 19552860, 'Hawaii': 1404054, 'Kentucky': 4395295, 'Rhode Island': 1051511, 'Nebraska': 1868516, 'Missouri': 6044171, 'Ohio': 11570808, 'Alabama': 4833722, 'South Dakota': 844877, 'Colorado': 5268367, 'Idaho': 1612136, 'New Jersey': 8899339, 'Washington': 6971406, 'North Carolina': 9848060, 'Tennessee': 6495978, 'Montana': 1015165, 'District of Columbia': 646449, 'Nevada': 2790136, 'Maine': 1328302}
donorfromstates=dict(donors['Donor State'].value_counts())
donorpopulation={}
for state, donr in donorfromstates.items():
    if state not in censusact2013:
        continue
    donorpopulation[state]=float(donr)*100000/censusact2013[state]
#donorpopulation=pd.DataFrame(donorpopulation)
#donorpopulation.columns=['Donor State', 'Population Ratio']
import operator
donorpopulation = sorted(donorpopulation.items(), key=operator.itemgetter(1), reverse = True)
xx = [x[0] for x in (donorpopulation)][1:16]
yy = [x[1] for x in (donorpopulation)][1:16]
#topratio=donorpopulation.sort_values(by='Population Ratio',ascending =False).head(15)
#x=topratio['Donor State']
#y=topratio['Population Ratio']
trace=go.Bar(x=xx,y=yy, name='Donors to population',opacity=.3)
data=[trace]
layout=go.Layout(barmode='group',
    legend=dict(dict(x=-.1, y=1.2)),
    margin=dict(b=120),
    title = 'States with highest Donors to Population Ratio')
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='population ratio')

import folium
from folium import plugins
from io import StringIO
#import folium 


statesll=StringIO("""State,Latitude,Longitude
Alabama,32.806671,-86.791130
Alaska,61.370716,-152.404419
Arizona,33.729759,-111.431221
Arkansas,34.969704,-92.373123
California,36.116203,-119.681564
Colorado,39.059811,-105.311104
Connecticut,41.597782,-72.755371
Delaware,39.318523,-75.507141
District of Columbia,38.897438,-77.026817
Florida,27.766279,-81.686783
Georgia,33.040619,-83.643074
Hawaii,21.094318,-157.498337
Idaho,44.240459,-114.478828
Illinois,40.349457,-88.986137
Indiana,39.849426,-86.258278
Iowa,42.011539,-93.210526
Kansas,38.526600,-96.726486
Kentucky,37.668140,-84.670067
Louisiana,31.169546,-91.867805
Maine,44.693947,-69.381927
Maryland,39.063946,-76.802101
Massachusetts,42.230171,-71.530106
Michigan,43.326618,-84.536095
Minnesota,45.694454,-93.900192
Mississippi,32.741646,-89.678696
Missouri,38.456085,-92.288368
Montana,46.921925,-110.454353
Nebraska,41.125370,-98.268082
Nevada,38.313515,-117.055374
New Hampshire,43.452492,-71.563896
New Jersey,40.298904,-74.521011
New Mexico,34.840515,-106.248482
New York,42.165726,-74.948051
North Carolina,35.630066,-79.806419
North Dakota,47.528912,-99.784012
Ohio,40.388783,-82.764915
Oklahoma,35.565342,-96.928917
Oregon,44.572021,-122.070938
Pennsylvania,40.590752,-77.209755
Rhode Island,41.680893,-71.511780
South Carolina,33.856892,-80.945007
South Dakota,44.299782,-99.438828
Tennessee,35.747845,-86.692345
Texas,31.054487,-97.563461
Utah,40.150032,-111.862434
Vermont,44.045876,-72.710686
Virginia,37.769337,-78.169968
Washington,47.400902,-121.490494
West Virginia,38.491226,-80.954453
Wisconsin,44.268543,-89.616508
Wyoming,42.755966,-107.302490""")
temp=donors['Donor State'].value_counts()
temp1=pd.DataFrame()
temp1['Donor State']=temp.index
temp1['Donor Count']=temp.values
state=pd.read_csv(statesll).rename(columns={'State':'Donor State'})
state=state.merge(temp1,on='Donor State', how='inner')
mapp=folium.Map(location=[39.5,-98.35])
for j, row in state.iterrows():
    rown=list(row)
    folium.CircleMarker([float(rown[1]), float(rown[2])], popup="<b>State:</b>" + rown[0].title() +"<br> <b>Donors:</b> "+str(int(rown[3])), radius=float(rown[3])*0.0001, color='#be0eef', fill=True).add_to(mapp)
mapp
#projects contributed by top 15 cities in California
sp=schprojects[schprojects['School State']=='California']['School City'].value_counts().head(15)
print(sp)
sp1=pd.DataFrame()
sp1['City']=sp.index
sp1['Count']=sp.values
#print(sp.City)
#print(sp.Count)
trace=go.Bar(x=sp1.City,y=sp1.Count,text=sp1.Count,
            textposition = 'auto',
            marker=dict( color='rgb(158,202,225)', line=dict( color='rgb(8,48,107)',  width=1.5) ), opacity=0.6)
layout=dict(title='Distribution of School cities')
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='CitySchools')
#plt.plot(sp['City'],sp['Count'],color='r')
#plt.show()
#top 20 cities with highest projects counts in USA
sp=schprojects['School City'].value_counts().head(20)
sp1=pd.DataFrame()
sp1['City']=sp.index
sp1['Count']=sp.values
trace=go.Bar(x=sp1.City,y=sp1.Count,text=sp1.Count,textposition = 'auto',
            marker=dict( color='rgb(158,202,225)', line=dict( color='rgb(8,48,107)',  width=1.5) ), opacity=0.6)
layout=dict(title='Distribution of Top 15 Cities with schools contributing highest number of projects in USA',xaxis=dict(tickangle=-45))
#layout={title='Distribution of School cities'}
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='CitySchools')
#plt.plot(sp['City'],sp['Count'],color='r')
#plt.show()
#school metro type
smt=schools['School Metro Type'].value_counts()
#smt.iplot(kind='pie',labels='labels',values='values',title='Distribution of SMT')
plt.figure(figsize=(10,4))
plt.bar(smt.index[::-1],smt.values[::-1],color='r')

plt.show()
#Project Grade Level Category
pg=projects['Project Grade Level Category'].value_counts()
plt.figure(figsize=(10,4))
plt.bar(pg.index[::-1],pg.values[::-1])

plt.show()
#project types
pr=schprojects['Project Type'].value_counts()
print(pr)
plt.figure(figsize=(8,6))
plt.pie(pr.values,labels=pr.index, autopct='%1.1f%%', shadow=True)
#plt.xlabel('Project Name')
#plt.ylabel('Count')
plt.title('Project Types')
#merging teachers and projects
projteach=teachers.merge(projects,on='Teacher ID',how='inner')
#finding projects count based on Teacher prefix
pt=projteach.groupby('Teacher Prefix').agg({'Project Subject Category Tree':'count'}).reset_index()
sns.barplot(x=pt['Teacher Prefix'],y=pt['Project Subject Category Tree'])

#teacher wtih specific project category tree and its count:
pt=projteach.groupby(['Teacher Prefix','Project Subject Category Tree']).agg({'Project ID':'count'}).sort_values(by=['Teacher Prefix','Project ID'],ascending=False).reset_index().rename(columns={'Project ID': 'Prefix Count'})
pt1=pt[pt['Teacher Prefix']=='Teacher'].head(5)
pt2=pt[pt['Teacher Prefix']=='Mr.'].head(5)
pt3=pt[pt['Teacher Prefix']=='Mrs.'].head(5)
import matplotlib.pyplot as plt

plt.pie(pt1['Prefix Count'],labels=pt1['Project Subject Category Tree'], autopct='%1.1f%%', shadow=True)#, label='one',c='r',figsize=(4,20))
plt.title('Prefix = Teachers, Top five')
plt.show()

plt.pie(pt2['Prefix Count'],labels=pt2['Project Subject Category Tree'], autopct='%1.1f%%', shadow=True)#', label='two',c='g',figsize=(4,20))
plt.title('Prefix = Mr., Top five')
plt.show()

plt.pie(pt3['Prefix Count'],labels=pt3['Project Subject Category Tree'] ,autopct='%1.1f%%', shadow=True)#, label='three',c='b',figsize=(4,20))
plt.title('Prefix = Mrs., Top five')
plt.show()
#projects per year
projects['Project Posted Date']=pd.to_datetime(projects['Project Posted Date'])
projects['PPY']=projects['Project Posted Date'].dt.year
projects['PPM']=projects['Project Posted Date'].dt.month
projects['PPW']=projects['Project Posted Date'].dt.weekday
projects['PPQ']=projects['Project Posted Date'].dt.quarter

projects['Project Funded Date']=pd.to_datetime(projects['Project Fully Funded Date'])
projects['PFY']=projects['Project Funded Date'].dt.year
projects['PFM']=projects['Project Funded Date'].dt.month
projects['PFW']=projects['Project Funded Date'].dt.weekday
projects['PFQ']=projects['Project Funded Date'].dt.quarter

projects['Funding Time']=projects['Project Funded Date']-projects['Project Posted Date']
projects['Funding Time'].value_counts().head()


#projects Posted on month 

mnmp={1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
projects['PPM']=projects['PPM'].map(mnmp)
ppm=projects['PPM'].value_counts()
#projects Funded on weekdays
projects['PFM']=projects['PFM'].map(mnmp)
pfm=projects['PFM'].value_counts()
t2=pd.DataFrame()
t1=pd.DataFrame()
t1['Month']=ppm.index
t1['Count']=ppm.values
t1=t1.sort_values(by='Month', ascending=False)
t2['Month']=pfm.index
t2['Count']=pfm.values
t2=t2.sort_values(by='Month', ascending=False)
fig, ax = plt.subplots(1,2,figsize=(15, 4))
ax[0].bar(t1['Month'],t1['Count'])
ax[0].set_title("Project Posted Month")
ax[1].bar(t2['Month'],t2['Count'])
ax[1].set_title("Project Funded Month")
#projects Posted on weekdays 

days={0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
projects['PPW']=projects['PPW'].map(days)
ppd=projects['PPW'].value_counts()

#projects Funded on weekdays
projects['PFW']=projects['PFW'].map(days)
pfd=projects['PFW'].value_counts()
t2=pd.DataFrame()
t1=pd.DataFrame()
t1['Days']=ppd.index
t1['Count']=ppd.values
t2['Days']=pfd.index
t2['Count']=pfd.values
fig, ax = plt.subplots(1,2,figsize=(10, 4))
ax[0].bar(t1['Days'],t1['Count'])
ax[0].set_title("Project Posted Days")
ax[1].bar(t2['Days'],t2['Count'])
ax[1].set_title("Project Funded Days")

#most projects were posted as well as funded on weekends
#Number of projects posted per year
proyear=projects.groupby('PPY').agg({'Project ID':'count'}).reset_index()
sns.barplot(x='PPY',y='Project ID', data=proyear)

#most project funding time
pft=projects['Funding Time'].value_counts()
t1=pd.DataFrame()
t1['Time']=pft.index
t1['Count']=pft.values
#longest project funding Time
print('Longest project funding time & number of projects:\n'+str(t1.sort_values(by='Time',ascending=False).head(1)))
print('\n')
#Shortest projects funding Time:
print('Shortest project funding time & number of projects:\n'+str(t1.sort_values(by='Time',ascending=False).tail(1)))
print('\n')
#most projects funding time:
print('Most project funding time & number of projects:\n'+str(t1.sort_values(by='Count',ascending=False).head(1)))
#free lunch percentage in states
fl=schools.groupby('School State').agg({'School ID':'count','School Percentage Free Lunch':'mean'}).reset_index().rename(columns={'School Percentage Free Lunch':'Mean FL PercSW','School ID': 'Count'}).sort_values(by='Count',ascending=False).head(5)
top5=fl.head(5)
plt.pie(fl['Count'],labels=fl['School State'] ,autopct='%1.1f%%', shadow=True)
plt.title('Top 5 States in school no.')
plt.show()


#checking correlation between resource quantity and resources unit price
rescor=resources.corr()
sns.heatmap(rescor,annot=True,cmap='coolwarm')


#resources supplying vendor in terms of resources quantity
vendor=resources.groupby('Resource Vendor Name').agg({'Resource Quantity':'sum'}).sort_values(by='Resource Quantity',ascending =False).reset_index().head(15)

trace=go.Bar(x=vendor['Resource Vendor Name'],y=vendor['Resource Quantity'], marker=dict( color='rgb(158,202,225)') , opacity=0.6)
layout=dict(title='Quantity supplied w.r.t. Vendor Names  ',xaxis=dict(tickangle=-45))
#layout={title='Distribution of School cities'}
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='vendor-distribution')
#most expensive item
t = resources.sort_values('Resource Unit Price', ascending=False).head(15)
t1 = pd.DataFrame()
t1['item'] = t['Resource Item Name']
t1['price'] = t['Resource Unit Price']
t1['standardized'] = """Handicapped Playground,Playground,CommercialStructure,Telescopic Gym,Playground+Playsystem,Sound System,CommercialStructure,Wood Playground,Playground+Playsystem,10 Alpha,Leveled Bookroom,Daktronic GS6,Contra Bassoon,Fencing,Playground+Playsystem""".split(",")
t2 = t1.groupby('standardized').agg({'price' : 'max'}).reset_index()
t2 = t2.sort_values('price', ascending = False)
trace=go.Bar(x=t2.standardized,y=t2.price, marker=dict( color='rgb(158,202,225)') , opacity=0.6)
layout=dict(title='Resource\'s Items with maximum price  ',xaxis=dict(tickangle=-45))
#layout={title='Distribution of School cities'}
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig, filename='vendor-distribution')
