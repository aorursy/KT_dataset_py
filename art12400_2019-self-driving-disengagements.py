import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/2019-autonomous-vehicle-disengagement-reports/2019AutonomousVehicleDisengagementReports.csv')

df1 = pd.read_csv('../input/2019-autonomous-vehicle-disengagement-reports/2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv')

df = pd.concat([df,df1],sort=False)

df.columns = ['MANUFACTURER','PERMIT NUMBER',"DATE",'VIN NUMBER','VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER','DRIVER PRESENT','DISENGAGEMENT INITIATED BY',"DISENGAGEMENT LOCATION","FACTS CAUSING DISENGAGEMENT","",""]



# preprocess data to standardize some fields 

df["DISENGAGEMENT INITIATED BY"] = df["DISENGAGEMENT INITIATED BY"].replace("Test driver", "Test Driver")

df["DISENGAGEMENT INITIATED BY"] = df["DISENGAGEMENT INITIATED BY"].replace("test driver", "Test Driver")

df["DISENGAGEMENT INITIATED BY"] = df["DISENGAGEMENT INITIATED BY"].replace("Safety Driver", "Test Driver")

df["DISENGAGEMENT INITIATED BY"] = df["DISENGAGEMENT INITIATED BY"].replace("Vehicle Operator", "Test Driver")



df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("street", "Street")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("STREET", "Street")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("street (high speed)", "Street")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace(" Downtown street", "Street")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("highway", "Highway")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("Rural", "Rural Road")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("parking facility", "Parking Facility")

df["DISENGAGEMENT LOCATION"] = df["DISENGAGEMENT LOCATION"].replace("Parking Lot", "Parking Facility")



df["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"] = df["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"].str.lower()

df["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"] = df["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"].replace("n", "no")

df["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"] = df["VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER"].replace("y", "yes")



df["DRIVER PRESENT"] = df["DRIVER PRESENT"].str.lower()

df["DRIVER PRESENT"] = df["DRIVER PRESENT"].replace("n", "no")

df["DRIVER PRESENT"] = df["DRIVER PRESENT"].replace("y", "yes")



df["DATE"] = df["DATE"].replace("1/30.2019", "1/30/2019")

df['DATE'] = pd.to_datetime(df['DATE'])

df = df.set_index('DATE')



df["FACTS CAUSING DISENGAGEMENT"] = df["FACTS CAUSING DISENGAGEMENT"].astype(str)
print("Number of reports: " + str(len(df.index)))

print("Number of unique vehicles: " +  str(df["VIN NUMBER"].nunique()))

print("Number of unique permtis: " +  str(df["PERMIT NUMBER"].nunique()))

print("Number of unique companies: " +  str(df["MANUFACTURER"].nunique()))

df.head()
plt.subplots(1,2,figsize=(12,5))

plt.subplot(1,2,1)

df['DRIVER PRESENT'].value_counts().plot.pie(autopct='%1.0f%%',fontsize=16)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.subplot(1,2,2)

df['VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER'].value_counts().plot.pie(autopct='%1.0f%%',fontsize=16)
plt.subplots(1,2,figsize=(12,5))

plt.suptitle('Disengagement Initiator and Location',fontsize=16)

plt.subplot(1,2,1)

df['DISENGAGEMENT INITIATED BY'].value_counts().plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.subplot(1,2,2)

df['DISENGAGEMENT LOCATION'].value_counts().plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)
plt.subplots(figsize=(12,5))

plt.suptitle('Number of Disengagements',fontsize=16)

df["MANUFACTURER"].value_counts().sort_values(ascending=False).plot.bar(width=0.5,color='b',edgecolor='k',align='center',linewidth=1)

plt.ylabel("Disengagements", fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)



plt.subplots(figsize=(12,5))

plt.suptitle('Number of Unique Vehicles with Disengagements',fontsize=16)

df.groupby('MANUFACTURER')["VIN NUMBER"].nunique().sort_values(ascending=False).plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)

plt.ylabel("Disengagements", fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)
d = df.groupby('DATE')["VIN NUMBER"].nunique()

plt.subplots(figsize=(12,5))

# there are 3 reports date to December of 2019

# these will be dropped since the dataset officially ends on November 30, 2019

d['2018-01-01':'2019-11-30'].resample('M').count().plot()

plt.xlabel("Month", fontsize=16)

plt.ylabel("Disengagements", fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)
descriptions = df["FACTS CAUSING DISENGAGEMENT"].str.lower().str.replace('[^\w\s]','').replace('[^\n]',' ')



counts = dict()

for description in descriptions:

    description = description.split(" ")

    for word in np.unique(description):

        if word in counts:

            counts[word] += 1

        else:

            counts[word] = 1



numTotalReports = len(df.index)*1.0



print("\nFraction of reports which referred to the keyword.\n")



software = counts["software"]/numTotalReports*100

hardware = counts["hardware"]/numTotalReports*100



print("Software %.2f%%" % software)

print("Hardware %.2f%%" % hardware)



sensor = (counts["sensor"]+counts["radar"]+counts["lidar"]+counts["camera"])/numTotalReports*100

camera = (counts["map"])/numTotalReports*100

lidar = (counts["map"])/numTotalReports*100

radar = (counts["map"])/numTotalReports*100

mapp = (counts["map"])/numTotalReports*100

gps = (counts["gps"])/numTotalReports*100



print("\nInformation Input Element:")

print("Sensor %.2f%%" % sensor)

print("Camera %.2f%%" % camera)

print("Lidar %.2f%%" % lidar)

print("Radar %.2f%%" % radar)

print("Map %.2f%%" % mapp)

print("GPS %.2f%%" % gps)



planning = (counts["planning"]+counts["planned"])/numTotalReports*100

perception = (counts["perception"])/numTotalReports*100

tracking = (counts["tracking"])/numTotalReports*100

trajectory = (counts["trajectory"])/numTotalReports*100

localization = (counts["localization"])/numTotalReports*100

control =(counts["control"]+counts["controller"]+counts["motionbehaviour"]+counts["oscillating"]+counts["closely"])/numTotalReports*100



print("\nSubsystem:")

print("Perception %.2f%%" % perception)

print("Tracking %.2f%%" % tracking)

print("Localization %.2f%%" % localization)

print("Planning %.2f%%" % planning)

print("Trajectory %.2f%%" % trajectory)

print("Control %.2f%%" % control)



car = (counts["car"]+counts["vehicle"])/numTotalReports*100

pedestrians = (counts["pedestrians"])/numTotalReports*100

bicyclist = (counts["bicyclist"])/numTotalReports*100

truck = (counts["truck"])/numTotalReports*100



print("\nObjects:")

print("Pedestrians %.2f%%" % pedestrians)

print("Bicyclist %.2f%%" % bicyclist)

print("Car %.2f%%" % car)

print("Truck %.2f%%" % truck)



light = (counts["green"]+counts["light"])/numTotalReports*100

construction = (counts["construction"])/numTotalReports*100

traffic = (counts["traffic"])/numTotalReports*100

intersection = (counts["intersection"])/numTotalReports*100

rain = (counts["rain"])/numTotalReports*100

weather = (counts["weather"])/numTotalReports*100

debris = (counts["debris"])/numTotalReports*100



print("\nEnvironment:")

print("Light %.2f%%" % light)

print("Construction %.2f%%" % construction)

print("Traffic %.2f%%" % traffic)

print("Intersection %.2f%%" % intersection)

print("Rain %.2f%%" % rain)

print("Weather %.2f%%" % weather)

print("Debris %.2f%%" % debris)




