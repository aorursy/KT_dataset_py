import pandas as pd

import matplotlib.pyplot as plt
data=pd.read_csv('../input/UberRequest.csv')
data.info()  


d=data[data.duplicated()]

d.head()
#pICKUP POINT , REQUEST ID AND  TIME STAMP IS IMPORTANT



data['Pickup point'].isnull().sum()

data['Request id'].isnull().sum()

data['Request timestamp'].isnull().sum()
## Complete trip should have pickup, drop ,time stamp and request id

completed=data.loc[data['Status']=='Trip Completed']

completed.isnull().sum()
## Cancelled trip should have request id , pickup point,driver id.

completed=data.loc[data['Status']=='Cancelled']

completed.isnull().sum() # no missing values for column which is important.
data['Request timestamp'] = pd.to_datetime(data['Request timestamp'])


data['Drop timestamp']=pd.to_datetime(data['Drop timestamp'])
data['Part of Day']=(data['Request timestamp'].dt.hour).apply(lambda x: 'Early Morning' if x in [4,5,6] else ('Morning' if x in [7,8,9] else ('Late Morning' if x in [10,11] else ('Afternoon' if x in [12,13,14] else ('Early Evening' if x in [15,16,17] else ('Late Evening' if x in [18,19,20] else ( 'Night' if x in [21,22,23] else ('Late Night' if x in [0,1,2,3] else None))))))))


##Comparing number of request accoridng to pickup point

b=data['Pickup point'].value_counts().plot(kind='bar',figsize=(10,5),title='Number of Request according to location',color=['Blue','Violet'])

b.set_xlabel('Pickup Point')

b.set_ylabel('Counts')

plt.show()



print("Number of Request from City: ",len(data[data['Pickup point']=='City']))

print("Number of Request from Airport: ",len(data[data['Pickup point']=='Airport']))




a=data['Status'].value_counts().plot(kind='bar',figsize=(10,6),title='Frequency of Request',color=['Red','Green','Blue'])

a.set_xlabel('Status')

a.set_ylabel('Frequency')

plt.show()
d=data['Part of Day'].value_counts().plot(kind='bar',figsize=(10,6),title='Request According to Part of Day')

d.set_xlabel('Part of Day')

d.set_ylabel('Frequency')

plt.show()



#Number of Requests

print("\nNumber of Request in Early Morning:",len(data[data['Part of Day']=='Early Morning']))

print("\nNumber of Request in Morning:",len(data[data['Part of Day']=='Morning']))

print("\nNumber of Request in Late Morning:",len(data[data['Part of Day']=='Late Morning']))

print("\nNumber of Request in Afternoon:",len(data[data['Part of Day']=='Afternoon']))

print("\nNumber of Request in Early Evening:",len(data[data['Part of Day']=='Early Evening']))

print("\nNumber of Request in Late Evening: ",len(data[data['Part of Day']=='Late Evening']))

print("\nNumber of Request in Night: ",len(data[data['Part of Day']=='Night']))

print("\nNumber of Request in Late Night: ",len(data[data['Part of Day']=='Late Night']))

a=data.loc[data['Part of Day']=='Early Morning']

b=data.loc[data['Part of Day']=='Morning']

c=data.loc[data['Part of Day']=='Late Morning']

d=data.loc[data['Part of Day']=='Afternoon']

e=data.loc[data['Part of Day']=='Early Evening']

f=data.loc[data['Part of Day']=='Late Evening']

g=data.loc[data['Part of Day']=='Night']

h=data.loc[data['Part of Day']=='Late Night']

fig,ax=plt.subplots(2,4)

fig.set_size_inches(15,7)

fig.subplots_adjust(wspace=0.5,hspace=1)



a['Status'].value_counts().plot(ax=ax[0,0],kind='bar',title='Early Morning',color=['Red','Green','Blue'])

b['Status'].value_counts().plot(ax=ax[0,1],kind='bar',title='Morning',color=['Green','Red','Blue'])

c['Status'].value_counts().plot(ax=ax[0,2],kind='bar',title='Late Morning',color=['Red','Blue','Green'])

d['Status'].value_counts().plot(ax=ax[0,3],kind='bar',title='Afternoon',color=['Red','Blue','Green'])

e['Status'].value_counts().plot(ax=ax[1,0],kind='bar',title='Early Evening',color=['Red','Blue','Green'])

f['Status'].value_counts().plot(ax=ax[1,1],kind='bar',title='Late Evening',color=['Blue','Red','Green'])

g['Status'].value_counts().plot(ax=ax[1,2],kind='bar',title='Night',color=['Blue','Red','Green'])

h['Status'].value_counts().plot(ax=ax[1,3],kind='bar',title='Late Night',color=['Blue','Red','Green'])

print("------Demand gap of early Morning------")

print("\n Number of Request By Customer: ",len(a))

print("\n Number of Supply by Uber",len(a[a['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(a)-len(a[a['Status']=='Trip Completed']))





print("\n------Demand gap of Morning------")

print("\n Number of Request By Customer: ",len(b))

print("\n Number of Supply by Uber",len(b[b['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(b)-len(b[b['Status']=='Trip Completed']))



print("\n------Demand gap of early Late Morning------")

print("\n Number of Request By Customer: ",len(c))

print("\n Number of Supply by Uber",len(c[c['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(c)-len(c[c['Status']=='Trip Completed']))





print("\n------Demand gap of early Afternoon------")

print("\n Number of Request By Customer: ",len(d))

print("\n Number of Supply by Uber",len(d[d['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(d)-len(d[d['Status']=='Trip Completed']))





print("\n------Demand gap of early Early Evening------")

print("\n Number of Request By Customer: ",len(e))

print("\n Number of Supply by Uber",len(e[e['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(e)-len(e[e['Status']=='Trip Completed']))





print("\n------Demand gap of early Late Evening------")

print("\n Number of Request By Customer: ",len(f))

print("\n Number of Supply by Uber",len(f[f['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(f)-len(f[f['Status']=='Trip Completed']))





print("\n------Demand gap of early Night------")

print("\n Number of Request By Customer: ",len(g))

print("\n Number of Supply by Uber",len(g[g['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(g)-len(g[g['Status']=='Trip Completed']))







print("\n------Demand gap of early Late Night------")

print("\n Number of Request By Customer: ",len(h))

print("\n Number of Supply by Uber",len(h[h['Status']=='Trip Completed']))

print("\n Demand Gap: ",len(h)-len(h[h['Status']=='Trip Completed']))
fig,ax=plt.subplots(2,4)

fig.set_size_inches(15,7)

fig.subplots_adjust(wspace=0.5,hspace=1)



a['Pickup point'].value_counts().plot(ax=ax[0,0],kind='bar',title='Early Morning',color=['Red','Blue'])

b['Pickup point'].value_counts().plot(ax=ax[0,1],kind='bar',title='Morning',color=['Red','Blue'])

c['Pickup point'].value_counts().plot(ax=ax[0,2],kind='bar',title='Late Morning',color=['Red','Blue'])

d['Pickup point'].value_counts().plot(ax=ax[0,3],kind='bar',title='Afternoon',color=['Red','Blue'])

e['Pickup point'].value_counts().plot(ax=ax[1,0],kind='bar',title='Early Evening',color=['Blue','Red'])

f['Pickup point'].value_counts().plot(ax=ax[1,1],kind='bar',title='Late Evening',color=['Blue','Red'])

g['Pickup point'].value_counts().plot(ax=ax[1,2],kind='bar',title='Night',color=['Blue','Red'])

h['Pickup point'].value_counts().plot(ax=ax[1,3],kind='bar',title='Late Night',color=['Red','Blue'])


#Seprating data of City and Airport

m=data.loc[data['Pickup point']=='City']

n=data.loc[data['Pickup point']=='Airport']

fig,ax=plt.subplots(1,2)

fig.set_size_inches(10,4)

fig.subplots_adjust(wspace=0.3,hspace=1)





m['Part of Day'].value_counts().plot(ax=ax[0],kind='bar',title='City')

n['Part of Day'].value_counts().plot(ax=ax[1],kind='bar',title='Airport')
print("\n --------------City--------------")

print("\n Trip Completed:",len(m[m['Status']=='Trip Completed']))

print("\n No Car Available:",len(m[m['Status']=='No Cars Available']))

print("\n Cancelled:",len(m[m['Status']=='Cancelled']))





print("\n --------------Airport--------------")

print("\n Trip Completed:",len(n[n['Status']=='Trip Completed']))

print("\n No Car Available:",len(n[n['Status']=='No Cars Available']))

print("\n Cancelled:",len(n[n['Status']=='Cancelled']))
fig,ax=plt.subplots(1,2)

fig.set_size_inches(10,4)

fig.subplots_adjust(wspace=0.3,hspace=1)



y=[1504,937,1066]

m['Status'].value_counts().plot(ax=ax[0],kind='bar',title='City')

n['Status'].value_counts().plot(ax=ax[1],kind='bar',title='Airport')

plt.legend()
fig,ax=plt.subplots(1,2)

fig.set_size_inches(10,4)

fig.subplots_adjust(wspace=0.3,hspace=1)



ax[0].pie(m['Status'].value_counts(),autopct='%1.1f%%',labels=['Trip Completed','Cancelled','No Cars Available'],colors=['Blue','Orange','Green'],shadow=True, startangle=90)

d=ax[0].set_title("CITY")

plt.setp(d, size=15, weight="bold",color='RED')



ax[0].axis('equal')

ax[1].pie(n['Status'].value_counts(),autopct='%1.1f%%',labels=['No Cars Available','Trip Completed','Cancelled'],colors=['Green','Blue','Orange'],shadow=True, startangle=90)

c=ax[1].set_title("AIRPORT")

ax[1].axis('equal')

plt.setp(c, size=15, weight="bold",color='RED')

fig1, ax1 = plt.subplots()

ax1.pie(data['Pickup point'].value_counts(),labels=['City','Airport'],autopct=lambda p: '{:.0f}'.format(p * len(data) / 100),shadow=True, startangle=90)

ax1.axis('equal')

plt.show()