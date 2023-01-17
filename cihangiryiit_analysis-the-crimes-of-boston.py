# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import folium as fl

from folium import plugins

import ipywidgets

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
unemp = pd.read_csv('../input/boston-unemployment-rate/Bostonrate.csv')

unemp
crime = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')

crime.head(8)
r_crime_type = crime.groupby(["OFFENSE_CODE_GROUP"]).mean()

crime_type = list(r_crime_type.index)

print(r_crime_type.index)

print(len(r_crime_type.index))
sns.catplot(y="YEAR",data=crime,kind="count")
winter_data =crime.loc[(crime.MONTH) >9]



c_of_year = ["#0affe6","#8b658b","#ffc1c1" ]



sns.catplot(x="YEAR",data=winter_data,hue="MONTH",height=4,kind="count",palette=c_of_year)
sns.catplot(y="YEAR",data=crime,hue="UCR_PART",height=4,kind="count")
plt.bar(unemp.Mounth,unemp["2016"] ,label= "2016")   

plt.legend(loc='upper right')    

plt.xlabel(' ')

plt.ylabel("rate %")

plt.show()
Y_2016 = crime.loc[(crime.YEAR)==2016]



sns.countplot(x="MONTH",data=Y_2016)

plt.show()
plt.bar(unemp.Mounth,unemp["2017"] ,label= "2017")   

plt.legend(loc='upper right')   

plt.xlabel(' ')

plt.ylabel("rate %")

plt.show()
Y_2017 = crime.loc[(crime.YEAR)==2017]



sns.countplot(x="MONTH",data=Y_2017)
def maxer (days,arr1,arr2,df):

    lap = 0 

    mx = 0

    ind= " "

    

    while lap < 7 :

        l = 0

        for i in df[days[lap]]:

            if i > mx :

                mx =  i

                ind = df.index[l]

            l += 1 

            

        arr2.append(str(ind)+"\n"+ str(days[lap]))

        lap += 1 

        arr1.append(mx)

        mx = 0

        ind = " "
d_crime_time = crime.groupby(["OFFENSE_CODE_GROUP","DAY_OF_WEEK"]).count().unstack().INCIDENT_NUMBER



Day = ["Friday","Monday","Saturday","Sunday","Thursday","Tuesday","Wednesday"]

day = []

crm=  []



maxer(Day,day,crm,d_crime_time)
d_crime_time.loc['Motor Vehicle Accident Response']
r_crime_time = crime.groupby(["INCIDENT_NUMBER"]).count()

r_time = []

x = [2,3,4,5,6,7,8,9,10,11,12,13]

for i in range(1,13):

    r_time.append(len(r_crime_time.loc[(r_crime_time.YEAR) > i]))

    



list_label = ["time","inicident"]

list_col = [x,r_time]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

crime_1 = pd.DataFrame(data_dict)

        

sns.catplot(x="time",y="inicident",kind="point",data=crime_1 ,markers=["o"] ,linestyle=["-"])
def parter(x):

    if i in other :

        return "Other"

    elif i in part1 :

        return "Part One"

    elif i in part2:

        return "Part Two"

    elif i in part3:

        return "Part Three"

    

def adder(arr,earr):

    

    for i in arr:   

        i = i.split(" ")

        empty = []

        L = 0

        while L < len(i):

            empty.append((i[L])[0])

            L += 1

        earr.append("".join(empty))

        empty=[]
dist = crime.groupby(["OFFENSE_CODE_GROUP","DISTRICT"]).count().unstack().INCIDENT_NUMBER

dıstrıct = [i for i in dist]

dist["ındex"]=dist.index



ucr= []

cap = []



other =['Arson','Auto Theft Recovery','Burglary - No Property Taken','License Plate Related Incidents','Manslaughter','Other']

part1 =['Aggravated Assault', 'Auto Theft', 'Commercial Burglary', 'Homicide', 'Larceny', 'Larceny From Motor Vehicle', 'Other Burglary', 'Residential Burglary', 'Robbery']

part2 =['Ballistics', 'Biological Threat', 'Bomb Hoax', 'Confidence Games', 'Counterfeiting', 'Criminal Harassment', 'Disorderly Conduct', 'Drug Violation', 'Embezzlement', 'Evading Fare', 'Explosives', 'Fire Related Reports', 'Firearm Violations', 'Fraud', 'Gambling', 'Harassment', 'Liquor Violation', 'Missing Person Reported', 'Offenses Against Child / Family', 'Operating Under the Influence', 'Other', 'Phone Call Complaints', 'Prisoner Related Incidents', 'Prostitution', 'Recovered Stolen Property', 'Restraining Order Violations', 'Simple Assault', 'Vandalism', 'Violations']

part3 =['Aircraft', 'Assembly or Gathering Violations', 'Explosives', 'Fire Related Reports', 'Firearm Discovery', 'Harbor Related Incidents', 'Investigate Person', 'Investigate Property', 'Landlord/Tenant Disputes', 'License Plate Related Incidents', 'License Violation', 'Medical Assistance', 'Missing Person Located', 'Missing Person Reported', 'Motor Vehicle Accident Response', 'Other', 'Police Service Incidents', 'Prisoner Related Incidents', 'Property Found', 'Property Lost', 'Property Related Damage', 'Search Warrants', 'Service', 'Towed', 'Verbal Disputes', 'Warrant Arrests']







for i in dist.ındex:

    ucr.append(parter(i))

    

    

adder(dist.index,cap)

    

dist["shortınd"] = cap   

dist["ucrpart"]=ucr



Other = dist.loc[(dist.ucrpart) == "Other" ]

PartOne = dist.loc[(dist.ucrpart) == "Part One" ]

PartTwo = dist.loc[(dist.ucrpart) == "Part Two" ]

PartThree = dist.loc[(dist.ucrpart) == "Part Three" ]



Other
col = [Other.A1 , Other.A15 , Other.A7 , Other.B2 , Other.B3 , Other.C11 , Other.C6 , Other.D14 , Other.D4 ,Other.E13,Other.E18,Other.E5]

fig, ax = plt.subplots(nrows=3,ncols=4,figsize=(15,10)) 

lap = 0

f= 0

s= 0

while lap < 12:

    pl = ax[f][s]

    pl.bar(Other.shortınd,col[lap],label= dıstrıct[lap])   

    pl.legend(loc='upper left')    

    pl.set_xlabel(' ')

    pl.set_ylabel(" ")

    

    lap += 1 

    s += 1 

    if s == 4:

        s=0

        f+=1

    

plt.show()

print(list(zip(Other.ındex,Other.shortınd)))
col = [PartOne.A1 , PartOne.A15 , PartOne.A7 , PartOne.B2 , PartOne.B3 , PartOne.C11 , PartOne.C6 , PartOne.D14 , PartOne.D4 ,PartOne.E13,PartOne.E18,PartOne.E5]

fig, ax = plt.subplots(nrows=3,ncols=4,figsize=(15,10)) 

lap = 0

f= 0

s= 0

while lap < 12:

    pl = ax[f][s]

    pl.bar(PartOne.shortınd,col[lap],label=dıstrıct[lap],align="center")   

    pl.legend(loc='upper left')    

    pl.set_xlabel(' ')

    pl.set_ylabel(" ")

    

    lap += 1 

    s += 1 

    if s == 4:

        s=0

        f+=1

    

plt.show()

print(list(zip(PartOne.ındex,PartOne.shortınd)))
col = [PartTwo.D14 , PartTwo.A15 , PartTwo.A7 ,  PartTwo.B3 ,PartTwo.B2 , PartTwo.C11 , PartTwo.C6 ,PartTwo.A1 , PartTwo.D4 ,PartTwo.E13,PartTwo.E18,PartTwo.E5]

fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(20,20)) 

lap = 0

f= 0

x_index = np.arange(len(PartTwo.shortınd))

width = 0.3



while lap < 4:

    pl = ax[lap]

    pl.bar(x_index-width,col[f],width = 0.25,label=dıstrıct[f],color="#444444")

    f+=1

    pl.bar(x_index,col[f],width = 0.25,label=dıstrıct[f],color="#008fd5")

    f+=1

    pl.bar(x_index+width,col[f],width = 0.25,label=dıstrıct[f],color="#e5ae38")

    f+=1

    pl.set_xticks(ticks=x_index,minor=False)

    pl.set_xticklabels(PartTwo.shortınd,fontdict={'fontsize':13})

    pl.legend(loc='upper left')    

    pl.set_xlabel(' ')

    pl.set_ylabel(" Part Two ")

    

    lap += 1

               

plt.show()

print(list(zip(PartTwo.ındex,PartTwo.shortınd)))
col = [PartThree.D14 , PartThree.A15 , PartThree.A7 ,  PartThree.B3 ,PartThree.B2 , PartThree.C11 , PartThree.C6 ,PartThree.A1 , PartThree.D4 ,PartThree.E13,PartThree.E18,PartThree.E5]

fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(20,20)) 

lap = 0

f= 0

x_index = np.arange(len(PartThree.shortınd))

width = 0.3



while lap < 4:

    pl = ax[lap]

    pl.bar(x_index-width,col[f],width = 0.25,label=dıstrıct[f],color="#444444")

    f+=1

    pl.bar(x_index,col[f],width = 0.25,label=dıstrıct[f],color="#008fd5")

    f+=1

    pl.bar(x_index+width,col[f],width = 0.25,label=dıstrıct[f],color="#e5ae38")

    f+=1

    pl.set_xticks(ticks=x_index,minor=False)

    pl.set_xticklabels(PartThree.shortınd,fontdict={'fontsize':12})

    pl.legend(loc='upper left')    

    pl.set_xlabel(' ')

    pl.set_ylabel(" Part Three ")

    

    lap += 1

               

plt.show()

print(list(zip(PartThree.ındex,PartThree.shortınd)))
def steeter (di,arr1,arr2,df):

    lap = 0 

    mx = 0

    stre= " "

    

    while lap < 12 :

        l = 0

        for i in df.loc[di[lap]].INCIDENT_NUMBER :

            if i > mx :

                mx =  i

                stre = df.loc[di[lap]].INCIDENT_NUMBER.index[l]

            l += 1 

            

        

        data = str(di[lap]) + "\n" + str(stre)

        arr2.append(mx)

        lap += 1 

        arr1.append(data)

        mx = 0

        stre = " "
street = crime.groupby(["DISTRICT","STREET"]).count()



col1= []

col2= []



dst = ["A1","A15","A7","B2","B3","C11","C6","D14","D4","E13","E18","E5"]  



steeter(dst,col1,col2,street)



fig, ax = plt.subplots(figsize=(26,8)) 

plt.bar(col1,col2,label="hello")   

plt.legend(loc='upper right')    

plt.xlabel('Count')

plt.ylabel("Street ")

plt.show()
index=[]

loc = []

colr = ['green', 'red', 'purple', 'lightred', 'gray', 'beige', 'lightgreen', 'cadetblue', 'darkred', 'white', 'darkgreen', 'lightgray', 'blue', 'black', 'orange', 'pink', 'darkblue', 'darkpurple', 'lightblue']



steer = crime.loc[((crime.STREET) == "BLUE HILL AVE")& ((crime.DISTRICT)== "B2")&((crime.YEAR)==2016)]



ster = steer.groupby(["OFFENSE_CODE_GROUP"]).count()



data = ster.loc[(ster.OFFENSE_CODE)> 10].index



for i in data :

    x = steer.loc[(steer.OFFENSE_CODE_GROUP)== i]

    lat = x["Lat"].dropna(inplace=False)

    long = x["Long"].dropna(inplace=False)

    Loc = zip(lat,long)

    index.append(i)

    loc.append(list(Loc))

    



street = fl.Map(location=[42.30584602, -71.08470215],zoom_start=14)

fl.raster_layers.TileLayer('Stamen Terrain').add_to(street)

fl.LayerControl().add_to(street)



rad = 50

lap = 0

while lap < 16 :

    

    for i in loc[lap]:

        fl.Circle(location=[i[0],i[1]],radius=rad, color=colr[lap],popup=index[lap] ,fill_color=colr[lap],fill_opacity=0.7).add_to(street)

    

    lap += 1

    rad -= 3



street

index=[]

loc = []

colr = ['green', 'red', 'purple', 'lightred', 'gray', 'beige', 'lightgreen', 'cadetblue', 'darkred', 'white', 'darkgreen', 'lightgray', 'blue', 'black', 'orange', 'pink', 'darkblue', 'darkpurple', 'lightblue']



steer = crime.loc[((crime.STREET) == "BLUE HILL AVE")& ((crime.DISTRICT)== "B3")&((crime.YEAR)==2016)]

ster = steer.groupby(["OFFENSE_CODE_GROUP"]).count()

data = ster.loc[(ster.OFFENSE_CODE)> 10].index



for i in data :

    x = steer.loc[(steer.OFFENSE_CODE_GROUP)== i]

    lat = x["Lat"].dropna(inplace=False)

    long = x["Long"].dropna(inplace=False)

    Loc = zip(lat,long)

    index.append(i)

    loc.append(list(Loc))

    



street = fl.Map(location=[42.305846 ,-71.077763],zoom_start=14)

fl.raster_layers.TileLayer('Stamen Terrain').add_to(street)

fl.LayerControl().add_to(street)



rad = 50

lap = 0

while lap < 16 :

    

    for i in loc[lap]:

        fl.Circle(location=[i[0],i[1]],radius=rad, color=colr[lap],popup=index[lap] ,fill_color=colr[lap],fill_opacity=0.7).add_to(street)

    

    lap += 1

    rad -= 3





street  
index=[]

loc = []

colr = ['green', 'red', 'purple', 'lightred', 'gray', 'beige', 'lightgreen', 'cadetblue', 'darkred', 'white', 'darkgreen', 'lightgray', 'blue', 'black', 'orange', 'pink', 'darkblue', 'darkpurple', 'lightblue']



steer = crime.loc[((crime.STREET) == "DORCHESTER AVE")& ((crime.DISTRICT)== "C11")&((crime.YEAR)==2016)]

ster = steer.groupby(["OFFENSE_CODE_GROUP"]).count()

data = ster.loc[(ster.OFFENSE_CODE)> 20].index



for i in data :

    x = steer.loc[(steer.OFFENSE_CODE_GROUP)== i]

    lat = x["Lat"].dropna(inplace=False)

    long = x["Long"].dropna(inplace=False)

    Loc = zip(lat,long)

    index.append(i)

    loc.append(list(Loc))

    



street = fl.Map(location=[42.305846 ,-71.077763],zoom_start=13)

fl.raster_layers.TileLayer('Stamen Terrain').add_to(street)

fl.LayerControl().add_to(street)



rad = 50

lap = 0

while lap < 16 :

    

    for i in loc[lap]:

        fl.Circle(location=[i[0],i[1]],radius=rad, color=colr[lap],popup=index[lap] ,fill_color=colr[lap],fill_opacity=0.5).add_to(street)

    

    lap += 1

    rad -= 3



street

index=[]

loc = []

colr = ['green', 'red', 'purple', 'lightred', 'gray', 'beige', 'lightgreen', 'cadetblue', 'darkred', 'white', 'darkgreen', 'lightgray', 'blue', 'black', 'orange', 'pink', 'darkblue', 'darkpurple', 'lightblue']



steer = crime.loc[((crime.STREET) == "BOYLSTON ST")& ((crime.DISTRICT)== "D4")&((crime.YEAR)==2016)]



ster = steer.groupby(["OFFENSE_CODE_GROUP"]).count()



data = ster.loc[(ster.OFFENSE_CODE)> 20].index



for i in data :

    x = steer.loc[(steer.OFFENSE_CODE_GROUP)== i]

    lat = x["Lat"].dropna(inplace=False)

    long = x["Long"].dropna(inplace=False)

    Loc = zip(lat,long)

    index.append(i)

    loc.append(list(Loc))

    



street = fl.Map(location=[42.34489594, -71.09659186],zoom_start=14)

fl.raster_layers.TileLayer('Stamen Terrain').add_to(street)

fl.LayerControl().add_to(street)



rad = 50

lap = 0

while lap < 16 :

    

    for i in loc[lap]:

        fl.Circle(location=[i[0],i[1]],radius=rad, color=colr[lap],popup=index[lap] ,fill_color=colr[lap],fill_opacity=0.7).add_to(street)

    

    lap += 1

    rad -= 3



street
dıst = crime.loc[((crime.DISTRICT) == "B3")]

lar = dıst.loc[((dıst.OFFENSE_CODE_GROUP) == "Larceny")&((dıst.YEAR)==2016)]



Larloc = []

lat = lar["Lat"].dropna(inplace=False)

long = lar["Long"].dropna(inplace=False)

Larloc.append(zip(lat,long))



Map = fl.Map(location=[42.305846 ,-71.077763], zoom_start=12)



for i in list(Larloc[0]):

    fl.Circle(location=i,radius=30,color="red",fill_color="red",fill_opacity=0.7).add_to(Map)
dıst = crime.loc[((crime.DISTRICT) == "B2")]

lar = dıst.loc[((dıst.OFFENSE_CODE_GROUP) == "Larceny")&((dıst.YEAR)==2016)]



Larloc = []

lat = lar["Lat"].dropna(inplace=False)

long = lar["Long"].dropna(inplace=False)

Larloc.append(zip(lat,long))





for i in list(Larloc[0]):

    fl.Circle(location=i,radius=30,color="blue",fill_color="blue",fill_opacity=0.7).add_to(Map) 
dıst = crime.loc[((crime.DISTRICT) == "C11")]

lar = dıst.loc[((dıst.OFFENSE_CODE_GROUP) == "Larceny")&((dıst.YEAR)==2016)]



Larloc = []

lat = lar["Lat"].dropna(inplace=False)

long = lar["Long"].dropna(inplace=False)

Larloc.append(zip(lat,long))





for i in list(Larloc[0]):

    fl.Circle(location=i,radius=30,color="orange",fill_color="orange",fill_opacity=0.7).add_to(Map)

Map