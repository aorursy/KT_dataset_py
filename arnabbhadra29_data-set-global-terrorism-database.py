import csv
import pandas as pd
filePath="/kaggle/input/gtd/globalterrorismdb_0718dist.csv"
globalterroristData=pd.read_csv(filePath,encoding='ISO-8859–1')
globalterroristData=pd.read_csv(filePath,encoding='ISO-8859–1')
r,c=globalterroristData.shape
print("Total number of rows ",r,"and attributes",c)
print("Column Names")
print(globalterroristData.columns.values)
print("Number of Years not available",globalterroristData['iyear'].isnull().sum())
print("Number of month details not available",globalterroristData['imonth'].isnull().sum())
print("Number of Apprximate date not available",globalterroristData['approxdate'].isnull().sum())
print("Location Unknown",globalterroristData['location'].isnull().sum())
print("Motive not Known",globalterroristData['motive'].isnull().sum())
print("Terrorist Group details Unknow",globalterroristData['gname'].isnull().sum())
globalterroristData.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
globalterroristData=globalterroristData[['Year','Month','Day','Country','Region','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
globalterroristData.head(3)
dateOfTerrorisam=globalterroristData[['Year','Month','Day']].apply(lambda x : '{}/{}/{}'.format(x[0],x[1],x[2]), axis=1).unique()
(numberOfDaysattackedOccured,)=dateOfTerrorisam.shape
print("number of Days attack occured between ",numberOfDaysattackedOccured)
numberOFpeopleKilled=globalterroristData['Killed'].sum()
print("Number of people Killed",int(numberOFpeopleKilled))
numberOFpeopleWounded=globalterroristData['Wounded'].sum()
print("Number of people Killed",int(numberOFpeopleWounded))
print("Total number of People affected",int(numberOFpeopleKilled+numberOFpeopleWounded))
terroristGroup=list(globalterroristData['Group'][globalterroristData['Group']!='Unknown'].unique())
print("Total Number of Know Terrorist group",len(terroristGroup),"\n")

#globalterroristData=pd.read_csv(filePath,encoding='ISO-8859–1')
#globalterroristData.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
#globalterroristData=globalterroristData[['Year','Month','Day','Country','Region','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
globalterroristData=globalterroristData[globalterroristData['Group']!='Unknown']
terrorist=globalterroristData[['Year','Group']].groupby('Group').count().sort_values(by=['Year'],ascending=False)
print("Top Terrorist Group\n------------------")
terroristGroup=terrorist.head(20).index.tolist()
for name in terroristGroup:
    print(name)
globalterroristData=globalterroristData[globalterroristData['Weapon_type']!='Unknown']
terrorist=globalterroristData[['Year','Weapon_type']].groupby('Weapon_type').count().sort_values(by=['Year'],ascending=False)
print("Top Five Weapon Used\n------------------")
weaponUsed=terrorist.head(5).index.tolist()
for name in weaponUsed:
    print(name)
globalterroristData=pd.read_csv(filePath,encoding='ISO-8859–1')

print("Column Names")
print(globalterroristData.columns.values)
import matplotlib.pyplot as plt
YearWiseTerrorisAttack=globalterroristData[['iyear','eventid']].groupby('iyear').count().sort_values(by=['iyear'],ascending=True)
year=YearWiseTerrorisAttack.index.tolist()
countOfattacks=YearWiseTerrorisAttack['eventid'].tolist()
plt.bar(year,countOfattacks)
plt.xlabel('Year')
plt.ylabel('No of attacks')
plt.title('Year wise number of attacks')
plt.grid(alpha=0.7)

plt.show()
causalites=globalterroristData[['iyear','nkill','nwound']].groupby('iyear').sum().sort_values(by=['iyear'],ascending=True)
year=YearWiseTerrorisAttack.index.tolist()
numberofKill=causalites['nkill'].tolist()
numberOfWound=causalites['nwound'].tolist()
plt.bar(year,numberofKill,color='r',label="Killed")
plt.bar(year,numberOfWound,bottom=numberofKill,color='b',label="Wound")
plt.xlabel('Year')
plt.ylabel('No of Casualities')
plt.title('Year wise Casulity')
plt.legend(loc='best')
plt.grid(alpha=0.7)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
countryWiseTerrorisAttack=globalterroristData[['country_txt','eventid','nkill']].groupby('country_txt').count().sort_values(by=['eventid'],ascending=False)
country=countryWiseTerrorisAttack.index.tolist()[0:21]
index=np.arange(0,len(country),1)
countOfattacks=countryWiseTerrorisAttack['eventid'].tolist()[0:21]
nkills=countryWiseTerrorisAttack['nkill'].tolist()[0:21]
plt.plot(index,nkills,'-o',color='b',alpha=0.7,label='Death')
plt.plot(index,countOfattacks,'-o',color='g',alpha=0.7, label='Attack')
plt.xticks(index,country, rotation=90)
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Top Country wise Attack Data Over 30 years')
plt.grid(alpha=0.7)
plt.legend(loc='best')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
countryWiseTerrorisAttack=globalterroristData[['region_txt','eventid','nkill']].groupby('region_txt').count()
country=countryWiseTerrorisAttack.index.tolist()
index=np.arange(0,len(country),1)
countOfattacks=countryWiseTerrorisAttack['eventid'].tolist()
nkills=countryWiseTerrorisAttack['nkill'].tolist()
plt.plot(index,nkills,'-o',color='r',alpha=0.7,label='Death')
plt.plot(index,countOfattacks,'-o',color='black',alpha=0.7, label='Attack')
plt.xticks(index,country, rotation=90)
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('Region wise Attack Data Over 30 years')
plt.grid(alpha=0.7)
plt.legend(loc='best')
plt.show()
from sklearn import preprocessing
attackOnIndia=globalterroristData[['iyear','country_txt','latitude', 'longitude']][globalterroristData['country_txt']=='India'].dropna()
x=np.array(attackOnIndia['iyear']).reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

x_scaled=x_scaled.reshape(1,-1)[0]

import matplotlib.pyplot as plt
import seaborn as sns
import shapefile as snp
import shapefile as shp

IndiaMap="../input/indiamap/IND_adm1.csv"
sf = shp.Reader(IndiaMap)
fields = [x[0] for x in sf.fields][1:]
records = sf.records()
plt.figure(figsize=(10,10))
for j in range(len(sf.shapeRecords())):
    shape=sf.shapeRecords()[j]
    for i in range(len(shape.shape.parts)):
        i_start = shape.shape.parts[i]
        if i==len(shape.shape.parts)-1:
            i_end = len(shape.shape.points)
            x = [i[0] for i in shape.shape.points[i_start:i_end]]
            y = [i[1] for i in shape.shape.points[i_start:i_end]]
        else:
            i_end = shape.shape.parts[i+1]
            x = [i[0] for i in shape.shape.points[i_start:i_end]]
            y = [i[1] for i in shape.shape.points[i_start:i_end]]
        plt.plot(x,y,'black')
cm = plt.cm.get_cmap('RdYlBu')
plt.title("Similar to India Map (Not exactly)\nYear wise Terrorist Attacks in different parts of India")
plt.scatter(attackOnIndia['longitude'],attackOnIndia['latitude'],marker='.',c=attackOnIndia['iyear'],cmap=cm)
plt.axis('off')
plt.clim(attackOnIndia['iyear'].min(),attackOnIndia['iyear'].max())
plt.colorbar()
#plt.xlabel("")
plt.show()

differentTypeOFattacks=globalterroristData['attacktype1_txt'].unique()
differentTypeOFattacks=np.delete(differentTypeOFattacks, np.where(differentTypeOFattacks == "Unknown"))

print(", ".join(differentTypeOFattacks))
weaponType=globalterroristData['weaptype1_txt'][(globalterroristData['iyear']>2011) ].unique()
weaponType=np.delete(weaponType, np.where(weaponType == "Unknown"))
print(",".join(weaponType))
print("Average people died in Terrorist attack {0:.2f}".format(globalterroristData['nkill'].mean()))
print("Total Suicide attack",globalterroristData['suicide'].sum())

print("Total Number of successful attacks",globalterroristData['success'].sum(),"among",globalterroristData['eventid'].count(),"attacks")