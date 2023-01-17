import matplotlib.pyplot as plt
import time
from IPython import display
from xml.dom import minidom
import math 
data1=open('../input/tracks-gpx/hardrock-100-endurance-run.gpx')
data2=open('../input/tracks-gpx/Hardrock-100 GPX.gpx')
data3=open('../input/tracks-gpx/The Bear 100 Ultra.gpx')
xmldoc = minidom.parse(data3)
track = xmldoc.getElementsByTagName('trkpt')
elevation=xmldoc.getElementsByTagName('ele')
datetime=xmldoc.getElementsByTagName('time')
n_track=len(track)
#PARSEAR GPX
lon_list=[]
lat_list=[]
h_list=[]
time_list=[]
for s in range(n_track):
    lon,lat=track[s].attributes['lon'].value,track[s].attributes['lat'].value
    elev=elevation[s].firstChild.nodeValue
    lon_list.append(float(lon))
    lat_list.append(float(lat))
    h_list.append(float(elev))
    
    # PARSEAR ELEMENTO TIEMPO (PARA LA VELOCIDAD, DESCARTADO)
    #dt=datetime[s].firstChild.nodeValue
    #time_split=dt.split('T')
    #hms_split=time_split[1].split(':')
    #time_hour=int(hms_split[0])
    #time_minute=int(hms_split[1])
    #time_second=int(hms_split[2].split('Z')[0])
    #total_second=time_hour*3600+time_minute*60+time_second
    #time_list.append(total_second)
#PASAR COORDENADAS GEODÉSICAS A CARTESIANAS
def geo2cart(lon,lat,h):
    a=6378137 #WGS 84 Major axis
    b=6356752.3142 #WGS 84 Minor axis
    e2=1-(b**2/a**2)
    N=float(a/math.sqrt(1-e2*(math.sin(math.radians(abs(lat)))**2)))
    X=(N+h)*math.cos(math.radians(lat))*math.cos(math.radians(lon))
    Y=(N+h)*math.cos(math.radians(lat))*math.sin(math.radians(lon))
    return X,Y

#FUNCION DISTANCIA
def distance(x1,y1,x2,y2):
    d=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

#FUNCION VELOCIDAD (DESCARTADA)
def speed(x0,y0,x1,y1,t0,t1):
    d=math.sqrt((x0-x1)**2+(y0-y1)**2)
    delta_t=t1-t0
    s=float(d/delta_t)
    return s
#LISTA VELOCIDADES Y DISTANCIAS
d_list=[0.0]
#speed_list=[0.0]
l=0
for k in range(n_track-1):
    if k<(n_track-1):
        l=k+1
    else:
        l=k
    XY0=geo2cart(lon_list[k],lat_list[k],h_list[k])
    XY1=geo2cart(lon_list[l],lat_list[l],h_list[l])
    
    #DISTANCIA
    d=distance(XY0[0],XY0[1],XY1[0],XY1[1])
    sum_d=d+d_list[-1]
    d_list.append(sum_d)
    
    #VELOCIDAD (DESCARTADA)
    #s=speed(XY0[0],XY0[1],XY1[0],XY1[1],time_list[k],time_list[l])
    #speed_list.append(s)
#PARA HARDROCK CLOCKWISE Y COUNTER CLOCKWISE

#PLOT TRACK
f,(track,elevation)=plt.subplots(2,1)
f.set_figheight(8*1.5)
f.set_figwidth(5*1.5)
plt.subplots_adjust(hspace=0.3)
track.plot(lon_list,lat_list,'k')
track.set_ylabel("Latitude")
track.set_xlabel("Longitude")
track.set_title("Hardrock 100 Counter-Clockwise Track Plot")
#plt.hold(True)

#PLOT VELOCIDAD (DESCARTADO)
#speed.bar(d_list,speed_list,30,color='w',edgecolor='w')
#speed.set_title("Speed")
#speed.set_xlabel("Distance(m)")
#speed.set_ylabel("Speed(m/s)")

#PLOT PERFIL ELEVACIÓN
base_reg=0
elevation.plot(d_list,h_list)
elevation.fill_between(d_list,h_list,base_reg,alpha=0.1)
elevation.set_title("Hardrock 100 Counter-Clockwise Elevation Profile")
elevation.set_xlabel("Distance(m)")
elevation.set_ylabel("GPS Elevation(m)")
elevation.grid()

#EXPORTAR COMO PNG
#f.savefig('hardrockcounter.png')
#PARA THE BEAR

plt.figure(figsize=(5*1.5,8*1.05))
plt.plot(lon_list,lat_list,'k')
plt.ylabel("Latitude")
plt.xlabel("Longitude")
plt.title("The Bear 100 Track Plot")
#plt.savefig('Beartrack.png')
#PARA THE BEAR

plt.figure(figsize=(8*1.5,5*1.1))
plt.plot(d_list,h_list)
plt.fill_between(d_list,h_list,base_reg,alpha=0.1)
plt.title("The Bear 100 Elevation Profile")
plt.xlabel("Distance(m)")
plt.ylabel("GPS Elevation(m)")
plt.grid()
#plt.savefig('BearHeight.png')