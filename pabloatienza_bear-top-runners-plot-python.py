import pandas as pd
df1 = pd.read_excel('../input/splits-carreras/desnivel_hardrock_clockwise.xlsx')
df2 = pd.read_excel('../input/splits-carreras/desnivel_hardrock_counter.xlsx')
df3 = pd.read_excel('../input/splits-carreras/desnivel_bear.xlsx')
import matplotlib.pyplot as plt
import time
from IPython import display
from xml.dom import minidom
import math 
#dfr = pd.read_excel('HardRock_Candi(mod).xlsx')
#dfr = pd.read_excel('hard-rock-counter-primercuartil.xlsx')
#dfr3 = pd.read_excel('CounterClockwise_top1.xlsx')
dfr3=pd.read_excel('../input/top-runners/bear_topporao.xlsx')
maxvels=0
minvels=0
velocidades={}
for i in range(10,23):
    for j in range(7):
        if i not in velocidades:
            velocidades[i]=dfr3.iloc[j,i]
            if maxvels<dfr3.iloc[j,i]:
                maxvels=dfr3.iloc[j,i]
        else:
            velocidades[i]+=dfr3.iloc[j,i]
            if maxvels<dfr3.iloc[j,i]:
                maxvels=dfr3.iloc[j,i]

print(velocidades)

for v in range(10,23):
    velocidades[v]=velocidades[v]/7
    
print(velocidades)
print(maxvels)
maxdesc=0
mindesc=0
descansos={}
for i in range(24,37):
    for j in range(7):
        if i not in descansos:
            descansos[i]=dfr3.iloc[j,i]
            if maxdesc<dfr3.iloc[j,i]:
                maxdesc=dfr3.iloc[j,i]
        else:
            descansos[i]+=dfr3.iloc[j,i]
            if maxdesc<dfr3.iloc[j,i]:
                maxdesc=dfr3.iloc[j,i]

print(descansos)

for d in range(24,37):
    descansos[d]=descansos[d]/7
    
print(descansos)
print(maxdesc)
velolist=[0]
desclist=[0]
for v1 in velocidades:
    velolist.append(velocidades[v1])
velolist.append(5.23)
#velolist.append(0)

for d1 in descansos:
    desclist.append(descansos[d1])
desclist.append(0)
#READ GPX FILE
data=open('../input/tracks-gpx/The Bear 100 Ultra.gpx')
xmldoc = minidom.parse(data)
track = xmldoc.getElementsByTagName('trkpt')
elevation=xmldoc.getElementsByTagName('ele')
#datetime=xmldoc.getElementsByTagName('time')
n_track=len(track)

#PARSING GPX ELEMENT
lon_list=[]
lat_list=[]
h_list=[]
#time_list=[]
for s in range(n_track):
    lon,lat=track[s].attributes['lon'].value,track[s].attributes['lat'].value
    elev=elevation[s].firstChild.nodeValue
    lon_list.append(float(lon))
    lat_list.append(float(lat))
    h_list.append(float(elev))
    # PARSING TIME ELEMENT
    #dt=datetime[s].firstChild.nodeValue
    #time_split=dt.split('T')
    #hms_split=time_split[1].split(':')
    #time_hour=int(hms_split[0])
    #time_minute=int(hms_split[1])
    #time_second=int(hms_split[2].split('Z')[0])
    #total_second=time_hour*3600+time_minute*60+time_second
    #time_list.append(total_second)
    
#GEODETIC TO CARTERSIAN FUNCTION
def geo2cart(lon,lat,h):
    a=6378137 #WGS 84 Major axis
    b=6356752.3142 #WGS 84 Minor axis
    e2=1-(b**2/a**2)
    N=float(a/math.sqrt(1-e2*(math.sin(math.radians(abs(lat)))**2)))
    X=(N+h)*math.cos(math.radians(lat))*math.cos(math.radians(lon))
    Y=(N+h)*math.cos(math.radians(lat))*math.sin(math.radians(lon))
    return X,Y

#DISTANCE FUNCTION
def distance(x1,y1,x2,y2):
    d=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

#SPEED FUNCTION
#def speed(x0,y0,x1,y1,t0,t1):
#    d=math.sqrt((x0-x1)**2+(y0-y1)**2)
#    delta_t=t1-t0
#    s=float(d/delta_t)
#    return s

#POPULATE DISTANCE AND SPEED LIST
d_list=[0.0]
speed_list=[0.0]
l=0
for k in range(n_track-1):
    if k<(n_track-1):
        l=k+1
    else:
        l=k
    XY0=geo2cart(lon_list[k],lat_list[k],h_list[k])
    XY1=geo2cart(lon_list[l],lat_list[l],h_list[l])
    
    #DISTANCE
    d=distance(XY0[0],XY0[1],XY1[0],XY1[1])
    sum_d=d+d_list[-1]
    d_list.append(sum_d)
    
    #SPEED
    #s=speed(XY0[0],XY0[1],XY1[0],XY1[1],time_list[k],time_list[l])
    #speed_list.append(s)
#Normalized Data
#dlist_norm = (x-min(x))/(max(x)-min(x))
dlist_norm=[]
for d in d_list:
    norm=(d-min(d_list))/(max(d_list)-min(d_list))
    #print(norm)
    dlist_norm.append(norm*15)
    
hlist_norm=[]
for h in h_list:
    norm=(h-min(h_list))/(max(h_list)-min(h_list))
    hlist_norm.append(norm)
    
#speed_norm=[]
#for s in df2['Speed']:
#    norm=(s-min(df2['Speed']))/(max(df2['Speed'])-min(df2['Speed']))
#    speed_norm.append(norm)
    
mile_norm=[]
for m in df3['Mile']:
    norm=(m-min(df3['Mile']))/(max(df3['Mile'])-min(df3['Mile']))
    mile_norm.append(norm*15)
    
nivel_norm=[]
for n in df3['Desnivel']:
    norm=(n-(df3['Desnivel'].mean()))/(df3['Desnivel'].std())
    nivel_norm.append(norm)
    
#speed_norm_2=[]
#for s in df2['Speed']:
#    norm=(s-min(df2['Speed']))/(max(df2['Speed'])-min(df2['Speed']))
#for s in df2['Speed']:
#    norm=(s-(df2['Speed'].mean()))/(df2['Speed'].std())
#    speed_norm_2.append(norm*1.75)

vel_norm=[]
for v in velolist:
    norm=(v-minvels)/(maxvels-minvels)
    vel_norm.append(norm)
    
desc_norm=[]
for d in desclist:
    norm=(d-mindesc)/(maxdesc-mindesc)
    desc_norm.append(norm)
#import seaborn as sns

plt.figure(figsize=(15,5))
base_reg=0

plt.plot(dlist_norm,hlist_norm,c="#34495e", alpha=0.6, label='Perfil Carrera')  #,c='#FF5733')
plt.fill_between(dlist_norm,hlist_norm,base_reg, alpha=0.6, color="#34495e")  #,alpha=0.6,color='#FF5733')

plt.plot(mile_norm, vel_norm, c="#e74c3c",alpha=0.6, label='Ritmo')  #, c='#33A2FF')
plt.fill_between(mile_norm, vel_norm,base_reg, alpha=0.6, color="#e74c3c") #,alpha=0.4,color='lightblue')

plt.plot(mile_norm,desc_norm, color="#3F7FBF",alpha=0.5, label='Descansos')
plt.fill_between(mile_norm, desc_norm,base_reg, alpha=0.5, color="#3F7FBF")
nn=1
#plt.axis('off')
for i in mile_norm:
    plt.axvline(x=i, ymin=0, ymax=1,c='#747474', linewidth=0.5)
    #plt.xlabel(df3['Name'][nn])
    nn+=1
    #plt.axis('off')

#plt.set_yticklabels(df3['Name'])    
plt.legend(loc='upper left')
plt.title('The Bear TOP 10: Promedio Ritmo y Descansos')
plt.axis('off')
plt.xticks(rotation='vertical')
#plt.savefig('Beartop10profile.png')
plt.show()