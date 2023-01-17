import math

import datetime

import os, sys

import numpy as np

import pandas as pd

DATA_DIR = "/kaggle/input/earthquake-database" + os.sep



# read data file

earthquake = pd.read_csv(

      DATA_DIR+"database.csv",

      sep=",",

      parse_dates={'datetime':['Date', 'Time']},

      encoding="utf-8",

      error_bad_lines=False,

)



# treating irregular data

for idx in [3378,7512,20650]:

    earthquake.at[idx, "datetime"] = earthquake.at[idx, "datetime"].split(" ")[0]



earthquake["datetime"] = pd.to_datetime(earthquake["datetime"], utc=True)

earthquake.set_index(["datetime"], inplace=True)



earthquake.head()
import matplotlib.pyplot as plt

plt.style.use('dark_background')

plt.grid(False)

from mpl_toolkits.basemap import Basemap

%matplotlib inline
ti = "Map of Earthquake's epicenter duaring 1965-2016"

fig, ax = plt.subplots(figsize=(18, 18), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24

m = Basemap(projection='robin', lat_0=0, lon_0=-170, resolution='c')

m.drawcoastlines()

#m.drawcountries()

m.fillcontinents(color='#606060', zorder = 1)

#m.bluemarble()

#m.drawmapboundary(fill_color='lightblue')



for i in range(5,10,1):

    #print(i)

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)&(earthquake["Type"]=="Earthquake")]

    x, y = m(list(tmp.Longitude), list(tmp.Latitude))

    points = m.plot(x, y, "o", label=f"Mag.: {i}.x", markersize=0.02*float(i)**3.2, alpha=0.55+0.1*float(i-5))



plt.title(f"{ti}", fontsize=22)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=18)

ax_pos = ax.get_position()

fig.text(ax_pos.x1-0.1, ax_pos.y0, "created by boomin", fontsize=16)

plt.show()
ti = "Distribution of Earthquake's Depth"

fig, ax = plt.subplots(figsize=(16, 9), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24

for i in range(5,8,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)&(earthquake["Type"]=="Earthquake")]

    plt.hist(tmp["Depth"], bins=60, density=True, histtype='step', linewidth=2.5, label=f"Mag.: {i}.x")

tmp = earthquake[(earthquake["Magnitude"]>=8)]

plt.hist(tmp["Depth"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: >8.x")

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("Depth, km")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.show()
earthquake = earthquake[earthquake["Depth"]<80]

earthquake = earthquake[earthquake["Type"]=="Earthquake"]
plt.clf()

ti = "Distribution of Earthquake's Latitude with Magnitude"

fig, ax = plt.subplots(figsize=(16, 9), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24

#

for i in range(5,8,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)&(earthquake["Type"]=="Earthquake")]

    plt.hist(tmp["Latitude"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: {i}.x")

tmp = earthquake[(earthquake["Magnitude"]>=8)]

plt.hist(tmp["Latitude"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: >8.x")

#

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("Latitude, deg")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.show()
# install ephemeris provided by NASA JPL

!pip install jplephem 
from numpy import linalg as LA



from astropy.coordinates import EarthLocation, get_body, AltAz, Longitude, Latitude, Angle

from astropy.time import Time, TimezoneInfo

from astropy import units as u

from astropy.coordinates import solar_system_ephemeris

solar_system_ephemeris.set('de432s')



# epcenter of each earthquake

pos = EarthLocation(Longitude(earthquake["Longitude"], unit="deg"), Latitude(earthquake["Latitude"], unit="deg"))



# time list of occerrd earthquake

dts = Time(earthquake.index, format="datetime64")

# position of the Moon and transforming from equatorial coordinate system to horizontal coordinate system

Mpos = get_body("moon", dts).transform_to(AltAz(location=pos))

# position of the Sun and transforming from equatorial coordinate system to horizontal coordinate system

Spos = get_body("sun", dts).transform_to(AltAz(location=pos))

# phase angle between the Sun and the Moon (rad)

SM_angle = Mpos.position_angle(Spos)

# phase angle from 0 (New Moon) to 180 (Full Moon) in degree

earthquake["p_angle"] = [ deg if deg<180 else 360-deg for deg in SM_angle.degree ]



earthquake["moon_dist"] = Mpos.distance.value/3.8e8

earthquake["sun_dist"] = Spos.distance.value/1.5e11

earthquake["moon_az"] = Mpos.az.degree

earthquake["moon_alt"] = Mpos.alt.degree



earthquake.head(5)
plt.clf()

ti = "Earthquakes with Phase Angle"

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

plt.rcParams["font.size"] = 24

fig.patch.set_facecolor('black')

for i in range(5,10,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)]

    plt.scatter(tmp.index, tmp["p_angle"], label=f"Mag.: {i}.x", s=0.02*float(i)**4.5, alpha=0.2+0.2*float(i-5))

plt.xlabel("Occuerd Year")

plt.ylabel("Phase Angle (0:New Moon, 180:Full Moon), deg")

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.grid(False)

plt.show()
plt.clf()

ti = "Phase Angle Histogram"

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

plt.rcParams["font.size"] = 24

fig.patch.set_facecolor('black')

for i in range(5,8,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)]

    plt.hist(tmp["p_angle"], bins=60, density=True, histtype='step', linewidth=2.0, label=f"Mag.: {i}.x")

i=8

tmp = earthquake[(earthquake["Magnitude"]>=8)]

plt.hist(tmp["p_angle"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: >8.x")



plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("Phase Angle (0:New Moon, 180:Full Moon), deg")

plt.ylabel("Count of Earthquake after 1965 (Normarized)")

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.grid(False)

plt.show()
plt.clf()

ti = "Map of Earthquake's epicenter with lunar pahse angle"

fig, ax = plt.subplots(figsize=(18, 10), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



import matplotlib.cm as cm

from matplotlib.colors import Normalize



for i in range(5,10,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)]

    m=plt.scatter(

        tmp.Longitude, tmp.Latitude, c=tmp.p_angle, s=0.02*float(i)**4.5,

        linewidths=0.4, alpha=0.4+0.12*float(i-5), cmap=cm.jet, label=f"Mag.: {i}.x",

        norm=Normalize(vmin=0, vmax=180)

    )        



plt.title(f"{ti}", fontsize=22)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=18)

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)



m.set_array(tmp.p_angle)

pp = plt.colorbar(m, cax=fig.add_axes([0.92, 0.17, 0.02, 0.48]), ticks=[0,45,90,135,180] )

pp.set_label("Phase Angle, deg", fontsize=18)

pp.set_clim(0,180)

plt.show()
plt.clf()

ti = "Distribution of Earthquake's Latitude with Phase Angle"

fig, ax = plt.subplots(figsize=(16, 10), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



plt.hist(earthquake["Latitude"], bins=60, density=True, histtype='step', linewidth=6, label=f"Average", color="w")



for deg in range(0,180,10):

    tmp = earthquake[(earthquake["p_angle"]>=deg)&(earthquake["p_angle"]<deg+10)]

    plt.hist(

        tmp["Latitude"], bins=60, density=True, histtype='step', linewidth=1.5, 

        label=f"{deg}-{deg+10}", color=cm.jet(deg/180), alpha=0.8

    )



plt.legend(bbox_to_anchor=(1.02, 0.97), loc='upper left', borderaxespad=0, fontsize=16)

plt.xlabel("Latitude, deg")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.03, ax_pos.y1-0.01, "phase angle", fontsize=16)

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.show()
ti="Relationship between Magnitue and Distances of the Sun and the Moon"

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

plt.rcParams["font.size"] = 24

fig.patch.set_facecolor('black')

for i in range(5,10,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)]

    plt.scatter(tmp["moon_dist"], tmp["sun_dist"], label=f"Mag.: {i}.x", s=0.02*float(i)**4.4, alpha=0.2+0.2*float(i-5))

plt.xlabel("distance between the Moon and the Earth (Normarized)")

plt.ylabel("distance between the Sun and the Earth (Normarized)")

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.grid(False)

plt.show()
ti = 'Distribution of Distance to the Moon with Phase Angle'

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



plt.hist(earthquake["moon_dist"], bins=60, density=True, histtype='step', linewidth=6, label=f"Average", color="w")

for deg in range(0,180,20):

    tmp = earthquake[(earthquake["p_angle"]>=deg)&(earthquake["p_angle"]<deg+20)]

    plt.hist(

        tmp["moon_dist"], bins=60, density=True, histtype='step', linewidth=1.5, 

        label=f"{deg}-{deg+20}", color=cm.jet(deg/180), alpha=0.8

    )



plt.legend(bbox_to_anchor=(1.02, 0.97), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("Distance to the Moon (Normalized)")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.03, ax_pos.y1-0.01, "phase angle", fontsize=16)

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.grid(False)

plt.show()
ti = 'Distribution of Distance to the Moon with Magnitude'

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



plt.hist(earthquake["moon_dist"], bins=60, density=True, histtype='step', linewidth=6.0, label=f"Average")

for i in range(5,8,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)]

    plt.hist(tmp["moon_dist"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: {i}.x")

i=8

tmp = earthquake[(earthquake["Magnitude"]>=i)]

plt.hist(tmp["moon_dist"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: >={i}")



plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("Distance to the Moon (Normalized)")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

plt.title(f"{ti}")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.grid(False)

plt.show()
ti = "Distribution of azimuth of the Moon"

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



for i in range(5,8,1):

    tmp = earthquake[(earthquake["Magnitude"]>=i)&(earthquake["Magnitude"]<i+1)]

    plt.hist(tmp["moon_az"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: {i}.x")

i=8

tmp = earthquake[(earthquake["Magnitude"]>=i)]

plt.hist(tmp["moon_az"], bins=60, density=True, histtype='step', linewidth=1.5, label=f"Mag.: >={i}")



plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("azimuth of the Moon (South:180)")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.title(f"{ti}")

plt.grid(False)

plt.show()
ti = "Distribution of azimuth of the Moon with Phase Angle"

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



plt.hist(earthquake["moon_az"], bins=60, density=True, histtype='step', linewidth=6, label=f"Average", color="w")

w=10

for deg in range(0,180,w):

    tmp = earthquake[(earthquake["p_angle"]>=deg)&(earthquake["p_angle"]<deg+w)]

    plt.hist(

        tmp["moon_az"], bins=60, density=True, histtype='step', linewidth=1.5, 

        label=f"{deg}-{deg+w}", color=cm.jet(deg/180), alpha=0.8

    )



plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("azimuth of the Moon (South:180)")

plt.ylabel("Count of Earthquake \n (Normarized at Total surface=1)")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.title(f"{ti}")

plt.grid(False)

plt.show()
ti = "Distribution of Moon's Altitude with Phase Angle"

fig, ax = plt.subplots(figsize=(18, 12), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24



plt.hist(earthquake["moon_alt"], bins=60, density=True, histtype='step', linewidth=6, label=f"Average", color="w")

w=10

for deg in range(0,180,w):

    tmp = earthquake[(earthquake["p_angle"]>=deg)&(earthquake["p_angle"]<deg+w)]

    plt.hist(

        tmp["moon_alt"], bins=60, density=True, histtype='step', linewidth=1.5, 

        label=f"{deg}-{deg+w}", color=cm.jet(deg/180), alpha=0.8

    )





plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

plt.xlabel("Altitude of the Moon")

plt.ylabel("Count of Earthquake after 1965 \n (Normarized at Total surface=1)")

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

plt.title(f"{ti}")

plt.grid(False)

plt.show()
import calendar



df = earthquake[earthquake["Type"]=="Earthquake"]

df2 = pd.pivot_table(df[df.index.year!=2011], index=df[df.index.year!=2011].index.month, aggfunc="count")["ID"]

df2 = df2/calendar.monthrange(2019,i)[1]/(max(df.index.year)-min(df.index.year)+1)



df = df[df.index.year==2011]

df3 = pd.pivot_table(df, index=df.index.month, aggfunc="count")["ID"]

df3 = df3/calendar.monthrange(2011,i)[1]



df4 = pd.concat([df2,df3], axis=1)

df4.columns=["except 2011","2011"]

df4.index=[ calendar.month_abbr[i] for i in range(1,13,1)]



left = np.arange(12)

labels = [ calendar.month_abbr[i] for i in range(1,13,1)]

width = 0.3



ti = "Seasonal Trend"

fig, ax = plt.subplots(figsize=(16, 8), dpi=96)

fig.patch.set_facecolor('black')

plt.rcParams["font.size"] = 24

for i,col in enumerate(df4.columns):

    plt.bar(left+width*i, df4[col], width=0.3, label=col)

plt.xticks(left + width/2, labels)

plt.ylabel("Count of Earthquake per day")

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

ax_pos = ax.get_position()

fig.text(ax_pos.x1+0.01, ax_pos.y0, "created by boomin", fontsize=16)

fig.text(ax_pos.x1-0.25, ax_pos.y1-0.04, f"Mean of except 2011: {df4.mean()[0]:1.3f}", fontsize=18)

fig.text(ax_pos.x1-0.25, ax_pos.y1-0.08, f"Std. of except 2011  : {df4.std()[0]:1.3f}", fontsize=18)

plt.title(f"{ti}")

plt.grid(False)

plt.show()

#

df4 = pd.concat([

    df4.T, 

    pd.DataFrame(df4.mean(),columns=["MEAN"]), 

    pd.DataFrame(df4.std(),columns=["STD"])

    ], axis=1)

print(df4.T)